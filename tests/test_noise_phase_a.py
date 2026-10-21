"""Generated noise/ECG and mocked inference only; repository raw/model I/O blocked."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from rhythmlens import noise_phase_a as phase
from rhythmlens import noise_phase_a_analysis as analysis
from rhythmlens.calibration import LABELS, probabilities, write_new
from rhythmlens.noise import prepare_pool, apply_plan, array_hash
from rhythmlens.preprocessing import (
    PreprocessingState,
    PREPROCESSING_VERSION,
    transform_physical,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/noise-phase-a-v1.yaml"


@pytest.fixture(autouse=True)
def no_real_artifacts(monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        for folder in (
            "data/raw",
            "outputs/rhythm",
            "outputs/temporal-internal-rhythm-evaluation-v1",
        ):
            if path.resolve().is_relative_to(ROOT / folder):
                pytest.fail(f"Protected real artifact access: {path}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)


def rows(labels=("N", "N", "A", "O", "~")):
    return [
        dict(
            record_id=f"synthetic-{i:03d}",
            label=k,
            duplicate_group="duplicate-N" if k == "N" else f"g{i}",
            samples=71 + i * 13,
            split="validation",
        )
        for i, k in enumerate(labels)
    ]


@pytest.fixture(scope="module")
def pools():
    t = np.arange(650000, dtype=np.int32)
    digital = np.column_stack((t % 201 - 100, (t * 7) % 301 - 150)).astype(np.int16)
    return {
        record: prepare_pool(
            digital, record=record, role="stress", source_sha256="a" * 64
        )
        for record in ("bw", "ma", "em")
    }


class GeneratedBackend:
    def __init__(self, rows, pools, destination):
        self.rows, self.pools, self.destination = rows, pools, destination
        self.state = PreprocessingState(
            PREPROCESSING_VERSION, 0.2, "synthetic-train", "synthetic-fit", 1, 0
        )
        self.calls = 0
        self.physical_calls = 0
        self.fail_at = None
        self.flat = False
        self.bad_historical = False
        self.model_seeds = []

    def physical(self, index):
        self.physical_calls += 1
        return (
            np.zeros(self.rows[index]["samples"])
            if self.flat
            else 0.3 + np.sin(np.arange(self.rows[index]["samples"]) / 11) * 0.2
        )

    @staticmethod
    def logits(items):
        return np.array(
            [[x.mean(), x.std(), -x.mean(), -x.std()] for _, x in items],
            dtype=np.float32,
        )

    def historical(self, seed):
        items = [
            (r, transform_physical(self.physical(i), self.state))
            for i, r in enumerate(self.rows)
        ]
        z = self.logits(items)
        if self.bad_historical:
            z[0, 0] += 2
        p = probabilities(z)
        return [
            dict(
                record_id=r["record_id"],
                true_label=r["label"],
                predicted_label=LABELS[int(z[i].argmax())],
                **{f"logit_{k}": float(z[i, j]) for j, k in enumerate(LABELS)},
                **{f"probability_{k}": float(p[i, j]) for j, k in enumerate(LABELS)},
            )
            for i, r in enumerate(self.rows)
        ]

    def state_and_model(self, seed):
        assert (self.destination / "plans.json").is_file()
        assert (self.destination / "registry.json").is_file()
        self.model_seeds.append(seed)
        return self.state, seed

    def predict(self, model, items):
        self.calls += 1
        if self.calls == self.fail_at:
            raise ValueError("synthetic technical failure")
        assert all(x.dtype == np.float32 for _, x in items)
        return self.logits(items)


@pytest.mark.parametrize(
    "key,value",
    [
        ("partition", "holdout"),
        ("extra", 1),
        ("output_root", "outputs/bypass"),
        ("execution_authorized", "false"),
        ("execution_authorized", 1),
        ("environment", {}),
        ("models", []),
        ("noise", {}),
        ("data_identity", {}),
    ],
)
def test_strict_contract_rejects_changes(tmp_path, key, value):
    c = yaml.safe_load(CONFIG.read_text())
    c[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):
        phase.load_config(path)


def test_authorization_variants_and_old_schemas(tmp_path):
    c = phase.load_config(CONFIG)
    assert c["execution_authorized"] is False
    for authorized in (False, True):
        c["execution_authorized"] = authorized
        p = tmp_path / "config.yaml"
        p.write_text(yaml.safe_dump(c))
        assert phase.load_config(p)["execution_authorized"] is authorized
    p.write_text(CONFIG.read_text() + "\nexecution_authorized: false\n")
    with pytest.raises(ValueError, match="Duplicate"):
        phase.load_config(p)
    for name in ("noise-robustness-v1.yaml", "noise-quality-datasets-v1.yaml"):
        with pytest.raises(ValueError, match="schema"):
            phase.load_config(ROOT / "configs" / name)


def test_gate_before_all_protected_io(tmp_path, monkeypatch):
    monkeypatch.setattr(
        phase,
        "metadata_preflight",
        lambda *a: pytest.fail("metadata I/O before authority"),
    )
    for runtime in (False, True):
        with pytest.raises(ValueError, match="execution_authorized"):
            phase.execute(tmp_path, CONFIG, runtime)
    c = phase.load_config(CONFIG)
    c["execution_authorized"] = True
    p = tmp_path / "authorized.yaml"
    p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match="authorize-execution"):
        phase.execute(tmp_path, p, False)
    monkeypatch.setattr(
        phase,
        "require_committed_source",
        lambda *a: (_ for _ in ()).throw(ValueError("commit gate")),
    )
    with pytest.raises(ValueError, match="commit gate"):
        phase.execute(tmp_path, p, True)
    assert not (tmp_path / "outputs").exists()


def test_metadata_hash_tamper_before_loader(tmp_path):
    p = tmp_path / "metadata.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="Changed metadata"):
        phase.metadata_preflight(
            tmp_path, {"metadata_sha256": {"metadata.json": "0" * 64}}
        )


@pytest.mark.parametrize("mutation", ("canonical", "audit_bytes", "roles", "pool"))
def test_noise_bindings_fail_closed(tmp_path, mutation):
    p = {"example": 1}
    a = {
        "status": "passed",
        "errors": [],
        "provenance_sha256": phase.canonical(p),
        "roles": [
            {"record": str(i), "role": "stress", "pool_sha256": "a" * 64}
            for i in range(9)
        ],
    }
    write_new(tmp_path / "p.json", p)
    write_new(tmp_path / "a.json", a)
    r = {
        "audit_sha256": phase.sha(tmp_path / "a.json"),
        "roles": copy.deepcopy(a["roles"]),
    }
    if mutation == "canonical":
        a["provenance_sha256"] = "0" * 64
        (tmp_path / "a.json").write_text(json.dumps(a))
        r["audit_sha256"] = phase.sha(tmp_path / "a.json")
    if mutation == "audit_bytes":
        r["audit_sha256"] = "0" * 64
    if mutation == "roles":
        r["roles"].pop()
    write_new(tmp_path / "r.json", r)
    c = {
        "metadata_sha256": {},
        "noise": {
            "provenance": "p.json",
            "audit": "a.json",
            "roles": "r.json",
            "pool_sha256": {
                f"{v['record']}/{v['role']}": v["pool_sha256"] for v in a["roles"]
            },
        },
    }
    if mutation == "pool":
        c["noise"]["pool_sha256"] = {}
    with pytest.raises(ValueError):
        phase.metadata_preflight(tmp_path, c)


def test_pairing_units_order_and_rng(pools):
    rs = rows()
    original = np.random.get_state()
    plans = phase.freeze_plans(rs, pools)
    assert len(plans) == len(rs) * 9 and len(analysis.conditions()) == 55
    for i, r in enumerate(rs):
        for noise in pools:
            for repeat in range(3):
                plan = plans[i, noise, repeat]
                assert plan == phase.freeze_plans(rs, pools)[i, noise, repeat]
                x = np.sin(np.arange(r["samples"]) / 13) * 0.2 + 0.6
                state = PreprocessingState(
                    PREPROCESSING_VERSION, 0.2, "train", "fit", 1, 0
                )
                shapes = []
                for snr in (24, 18, 12, 6, 0, -6):
                    corrupted, meta = apply_plan(
                        x, plan, pools, partition="validation", snr_db=snr
                    )
                    shapes.append((corrupted - x) / meta["added_rms_mv"])
                    expected = ((corrupted - np.median(corrupted)) / 0.2).astype(
                        np.float32
                    )
                    np.testing.assert_array_equal(
                        transform_physical(corrupted, state), expected
                    )
                    assert abs(meta["achieved_snr_db"] - snr) <= 1e-6
                for shape in shapes[1:]:
                    np.testing.assert_allclose(shapes[0], shape, atol=1e-10)
    after = np.random.get_state()
    np.testing.assert_array_equal(original[1], after[1])
    assert original[2:] == after[2:]
    altered = dict(pools)
    altered["bw"] = copy.copy(pools["bw"])
    object.__setattr__(altered["bw"], "role", "development-reserve")
    with pytest.raises(ValueError, match="stress"):
        phase.freeze_plans(rs, altered)


@pytest.mark.parametrize(
    "change", ("missing", "labels", "prediction", "logits", "probability", "duplicate")
)
def test_clean_gate_mismatches(tmp_path, pools, change):
    rs = rows()
    b = GeneratedBackend(rs, pools, tmp_path)
    historical = b.historical(42)
    z = np.array(
        [[r[f"logit_{k}"] for k in LABELS] for r in historical], dtype=np.float32
    )
    if change == "missing":
        historical.pop()
    elif change == "labels":
        historical[0]["true_label"] = "A"
    elif change == "prediction":
        historical[0]["predicted_label"] = "~"
    elif change == "logits":
        z[0, 0] += 1
    elif change == "probability":
        historical[0]["probability_N"] += 0.01
    else:
        historical[-1] = historical[0]
    with pytest.raises(ValueError):
        phase.baseline_gate(rs, z, historical)


def fast_bootstrap(monkeypatch):
    real = analysis.bootstrap
    monkeypatch.setattr(phase, "bootstrap", lambda rows, p: real(rows, p, replicates=8))


def test_complete_engine_seals_before_scoring(tmp_path, pools, monkeypatch):
    fast_bootstrap(monkeypatch)
    rs = rows()
    dest = tmp_path / "run"
    b = GeneratedBackend(rs, pools, dest)
    original = phase.score

    def checked(rows, z):
        assert len(json.loads((dest / "seal.json").read_text())["predictions"]) == 165
        assert b.calls == 165
        return original(rows, z)

    monkeypatch.setattr(phase, "score", checked)
    phase.run_verified(dest, rs, b, {"synthetic": True}, lambda: None)
    assert json.loads((dest / "status.json").read_text())["state"] == "completed"
    accounting = json.loads((dest / "accounting.json").read_text())
    assert len(accounting) == 3 * 55 * len(rs) and all(
        r["status"] == "completed" for r in accounting
    )
    z = phase.sealed_predictions(dest, rs)
    assert z.shape == (3, 55, len(rs), 4)
    result = json.loads((dest / "metrics.json").read_text())
    assert result["per_seed"]["42"]["clean"]["eligible"] == len(rs)
    assert b.model_seeds[:3] == [42, 43, 44]
    with pytest.raises(FileExistsError):
        phase.run_verified(dest, rs, b, {}, lambda: None)
    path = dest / "seed44/em-snr-6-repeat2/logits.npy"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Sealed bytes"):
        phase.sealed_predictions(dest, rs)


@pytest.mark.parametrize(
    "mode",
    (
        "model",
        "baseline",
        "metadata",
        "nan",
        "flat",
        "noise_flat",
        "invalid_length",
        "missing_export",
    ),
)
def test_failure_evidence_and_no_completed_metrics(tmp_path, pools, monkeypatch, mode):
    rs = rows()
    dest = tmp_path / "run"
    b = GeneratedBackend(rs, pools, dest)

    def verify():
        return None

    if mode == "model":
        b.fail_at = 5
    elif mode == "baseline":
        b.bad_historical = True
    elif mode == "metadata":

        def verify():
            raise ValueError("identity changed")
    elif mode == "nan":
        b.physical = lambda i: np.full(rs[i]["samples"], np.nan)
    elif mode == "flat":
        b.flat = True
        fast_bootstrap(monkeypatch)
        phase.run_verified(dest, rs, b, {}, verify)
        meta = json.loads((dest / "seed42/bw-snr24-repeat0/batch0.json").read_text())
        assert all(
            c["corruption"]["achieved_snr_db"] is None
            and not c["corruption"]["corruption_applied"]
            for c in meta["cases"]
        )
        return
    elif mode == "noise_flat":
        b.pools = {k: copy.copy(v) for k, v in pools.items()}
        for v in b.pools.values():
            zero = np.zeros_like(v.values)
            zero.flags.writeable = False
            object.__setattr__(v, "values", zero)
            object.__setattr__(v, "pool_sha256", array_hash(zero))
    elif mode == "invalid_length":
        rs[0]["samples"] = 99999999
    else:
        b.historical = lambda seed: (_ for _ in ()).throw(
            FileNotFoundError("missing export")
        )
    with pytest.raises((ValueError, FileNotFoundError)):
        phase.run_verified(dest, rs, b, {}, verify)
    assert json.loads((dest / "status.json").read_text())["state"] == "failed"
    accounting = json.loads((dest / "accounting.json").read_text())
    assert len(accounting) == 3 * 55 * len(rs)
    assert not (dest / "metrics.json").exists() and not (dest / "seal.json").exists()
    if mode == "model":
        assert any(r["status"] == "technical_failure" for r in accounting)
        assert any(r["status"] == "completed" for r in accounting)
        assert (dest / "seed42/clean/batch0.npy").is_file()
    if mode == "baseline":
        assert b.calls == 1
    if mode in ("metadata", "missing_export", "invalid_length"):
        assert not b.model_seeds


def test_primary_and_full_metrics_accounting():
    rs = rows(("A", "A", "N", "O", "~"))
    z = np.array(
        [[0, 0, 0, 0], [0, 5, 0, 0], [0, 0, 0, 5], [0, 5, 0, 0], [0, 5, 0, 0]],
        dtype=np.float32,
    )
    cases, m = analysis.score_condition(rs, z)
    assert m["automatic_af_sensitivity"]["value"] == 0.5
    assert m["automatic_error"]["value"] == 0.8
    assert m["false_af_on_N_O"]["value"] == 0.5
    assert m["true_noisy_predicted_A"] == 1
    assert m["rhythm_output_coverage"]["numerator"] == 4
    assert m["deferral_rate"]["numerator"] == 0
    assert np.asarray(m["outcome_matrix"]).shape == (4, 7)
    assert sum(m["af_outcomes"]) == 2
    assert m["probability_metrics"]["binary_af_eligible"] == 4
    assert m["per_class"]["A"]["tp"] == 1 and m["per_class"]["A"]["fp"] == 2
    _, missing = analysis.score_condition(
        rows(("N",)), np.zeros((1, 4), dtype=np.float32)
    )
    assert missing["automatic_af_sensitivity"]["value"] is None
    assert set(missing["zero_support_classes"]) == {"A", "O", "~"}
    assert len(cases) == 5


def test_bootstrap_duplicate_groups_shared_draws_and_null_af():
    rs = rows()
    n = len(rs)
    z = np.zeros((3, 55, n, 4), dtype=np.float32)
    for i, r in enumerate(rs):
        z[:, :, i, LABELS.index(r["label"])] = 4
    original = np.random.get_state()
    draws, stats, meta = analysis.bootstrap(rs, z, replicates=20)
    draws2, stats2, _ = analysis.bootstrap(rs, z, replicates=20)
    np.testing.assert_array_equal(draws, draws2)
    np.testing.assert_array_equal(stats, stats2)
    assert meta["groups"]["duplicate-N"] == ["synthetic-000", "synthetic-001"]
    assert all(
        len(v) == 1 or name == "duplicate-N" for name, v in meta["groups"].items()
    )
    np.testing.assert_array_equal(
        stats[..., 0, :], np.broadcast_to([1, 1, 0], stats[..., 0, :].shape)
    )
    assert not stats[..., 1, :].any()
    rng = np.random.Generator(np.random.PCG64(20261016))
    # Each of four strata has one group here, sampled once, retaining both N members.
    assert (draws == 1).all()
    assert rng.integers(0, 1) == 0
    np.testing.assert_array_equal(original[1], np.random.get_state()[1])
    noaf = rows(("N", "N", "O", "~"))
    _, s, m = analysis.bootstrap(noaf, np.zeros((3, 55, len(noaf), 4)), replicates=4)
    assert np.isnan(s[..., 1]).all()
    assert (
        m["intervals"]["equal_seed_mean"]["clean"]["absolute"][
            "automatic_af_sensitivity"
        ]["interval"]
        is None
    )
    bad = copy.deepcopy(rs)
    bad[2]["duplicate_group"] = "duplicate-N"
    with pytest.raises(ValueError, match="Conflicting"):
        analysis.bootstrap(bad, z, replicates=2)


def test_bootstrap_nontrivial_group_multiplicity_and_paired_repetition_oracle():
    rs = rows(("N", "N", "N", "A", "A", "O", "~"))
    rs[2]["duplicate_group"] = "single-N"
    z = np.zeros((3, 55, len(rs), 4), dtype=np.float32)
    guesses = np.array([0, 1, 1, 1, 0, 2, 3])
    for s in range(3):
        for c in range(55):
            g = guesses.copy()
            if c and c % 3 == 0:
                g[2] = 0
            if s == 1:
                g[3] = 0
            z[s, c, np.arange(len(rs)), g] = 3
    draws, stats, meta = analysis.bootstrap(rs, z, replicates=20)
    assert len(np.unique(draws, axis=0)) > 1
    names = meta["draw_columns"]
    for stratum in (["duplicate-N", "single-N"], ["g3", "g4"], ["g5"], ["g6"]):
        np.testing.assert_array_equal(
            draws[:, [names.index(g) for g in stratum]].sum(axis=1), len(stratum)
        )
    for b in (0, 3, 8):
        indices = [
            i
            for j, g in enumerate(names)
            for _ in range(draws[b, j])
            for i, r in enumerate(rs)
            if r["record_id"] in meta["groups"][g]
        ]
        manual = []
        for c in range(4):
            cm = np.zeros((4, 4), dtype=int)
            for i in indices:
                cm[LABELS.index(rs[i]["label"]), z[0, c, i].argmax()] += 1
            per_class = [
                2 * cm[k, k] / (cm[k].sum() + cm[:, k].sum())
                if cm[k].sum() + cm[:, k].sum()
                else 0
                for k in range(4)
            ]
            manual.append(
                [
                    sum(per_class) / 4,
                    cm[1, 1] / cm[1].sum(),
                    1 - np.trace(cm) / cm.sum(),
                ]
            )
        np.testing.assert_allclose(stats[b, 0, :4, 0], manual)
        np.testing.assert_allclose(stats[b, 0, 55, 0], np.mean(manual[1:4], axis=0))
        np.testing.assert_allclose(
            stats[b, 0, 55, 1], np.mean(manual[1:4], axis=0) - manual[0]
        )
    expected = np.quantile(stats[:, 3, 55, 1, 0], [0.025, 0.975], method="linear")
    np.testing.assert_array_equal(
        meta["intervals"]["equal_seed_mean"]["bw-snr24-mean-repetitions"][
            "paired_delta_from_clean"
        ]["macro_f1"]["interval"],
        expected,
    )


def test_backend_batch_masks_and_eval_mode():
    import torch

    class MaskModel(torch.nn.Module):
        def forward(self, signals, mask):
            assert signals.dtype == torch.float32 and mask.dtype == torch.bool
            assert mask.tolist() == [[True] * 3 + [False] * 4, [True] * 7]
            assert signals.shape == (2, 1, 7)
            assert not torch.is_grad_enabled()
            return torch.stack([signals[:, 0].sum(axis=1)] * 4, axis=1)

    model = MaskModel().eval()
    backend = object.__new__(phase.Backend)
    rs = rows(("N", "A"))
    items = [
        (rs[0], np.ones(3, dtype=np.float32)),
        (rs[1], np.ones(7, dtype=np.float32)),
    ]
    z = backend.predict(model, items)
    np.testing.assert_array_equal(z, np.array([[3] * 4, [7] * 4], dtype=np.float32))
    model.train()
    with pytest.raises(ValueError, match="evaluation"):
        backend.predict(model, items)


def test_backend_only_reconstructs_reviewed_stress_role(tmp_path, pools, monkeypatch):
    from rhythmlens import nstdb

    calls = []

    def load(raw, provenance, role):
        calls.append(role)
        return pools

    monkeypatch.setattr(nstdb, "load_pools", load)
    write_new(tmp_path / "p.json", {"generated": True})
    write_new(
        tmp_path / "roles.json", {"roles": [p.metadata() for p in pools.values()]}
    )
    c = {"noise": {"raw": "synthetic", "provenance": "p.json", "roles": "roles.json"}}
    phase.Backend(tmp_path, c, object())
    assert calls == ["stress"]
    r = json.loads((tmp_path / "roles.json").read_text())
    r["roles"][0]["pool_sha256"] = "f" * 64
    (tmp_path / "roles.json").write_text(json.dumps(r))
    with pytest.raises(ValueError, match="descriptor"):
        phase.Backend(tmp_path, c, object())


@pytest.mark.parametrize("failure", ("bootstrap", "sealed_tamper"))
def test_failure_after_sealing_never_publishes_metrics(
    tmp_path, pools, monkeypatch, failure
):
    rs = rows()
    dest = tmp_path / "run"
    backend = GeneratedBackend(rs, pools, dest)

    def failed(rows, predictions):
        assert (dest / "seal.json").is_file()
        if failure == "bootstrap":
            raise ValueError("synthetic bootstrap failure")
        (dest / "seed44/em-snr-6-repeat2/logits.npy").write_bytes(
            b"changed after scoring"
        )
        return analysis.bootstrap(rows, predictions, replicates=2)

    monkeypatch.setattr(phase, "bootstrap", failed)
    with pytest.raises(ValueError):
        phase.run_verified(dest, rs, backend, {}, lambda: None)
    assert (dest / "seal.json").is_file() and (dest / "failure-context.json").is_file()
    assert not (dest / "metrics.json").exists()
    assert json.loads((dest / "status.json").read_text())["state"] == "failed"


def test_no_scoring_missing_seed_seal(tmp_path):
    write_new(tmp_path / "seal.json", {"predictions": {}, "evidence": {}})
    write_new(tmp_path / "plans.json", {"rows": rows()})
    with pytest.raises(ValueError, match="all conditions"):
        phase.sealed_predictions(tmp_path, rows())


def test_length_bucket_order_matches_original_no_shuffle_contract(
    tmp_path, pools, monkeypatch
):
    # 19 records exercise the leading remainder and stable global length sort.
    rs = rows(tuple("NAO~"[i % 4] for i in range(19)))
    for i, r in enumerate(rs):
        r["samples"] = 110 + (18 - i) * 7
        r["duplicate_group"] = f"g{i}"
    dest = tmp_path / "run"
    b = GeneratedBackend(rs, pools, dest)
    b.fail_at = 3
    with pytest.raises(ValueError):
        phase.run_verified(dest, rs, b, {}, lambda: None)
    batch0 = json.loads((dest / "seed42/clean/batch0.json").read_text())
    batch1 = json.loads((dest / "seed42/clean/batch1.json").read_text())
    assert batch0["indices"] == [0, 1, 2]
    assert batch1["indices"] == list(range(18, 2, -1))


def test_generation_registry_is_exclusive_before_protected_factory(tmp_path):
    dest = tmp_path / "occupied"
    dest.mkdir()

    def factory():
        pytest.fail("protected backend accessed before collision check")

    with pytest.raises(FileExistsError):
        phase.run_verified(dest, rows(), factory, {}, lambda: None)


def test_frozen_plan_rows_reject_relabeling_after_seal(tmp_path, pools, monkeypatch):
    fast_bootstrap(monkeypatch)
    rs = rows()
    dest = tmp_path / "run"
    b = GeneratedBackend(rs, pools, dest)
    phase.run_verified(dest, rs, b, {}, lambda: None)
    changed = copy.deepcopy(rs)
    changed[0]["label"] = "A"
    with pytest.raises(ValueError, match="frozen plans"):
        phase.sealed_predictions(dest, changed)
    result = json.loads((dest / "metrics.json").read_text())
    assert result["paired_delta_equal_seed_summary"]["clean"]["macro_f1"]["mean"] == 0
    assert result["flat_cases"]["42"]["bw-snr24-repeat0"]["count"] == 0
    status = json.loads((dest / "status.json").read_text())
    assert status["artifacts_index_sha256"] == phase.sha(dest / "artifacts-sha256.json")


@pytest.mark.parametrize(
    "change", (None, "seed", "epoch", "kind", "scale", "config", "source")
)
def test_checkpoint_candidate_metadata_with_mocked_loader(
    tmp_path, monkeypatch, change
):
    from types import SimpleNamespace
    import torch
    from rhythmlens import rhythm_training

    state = PreprocessingState(
        PREPROCESSING_VERSION, 0.18008390945457742, "train", "fit", 1, 0
    )
    checkpoint = dict(
        kind="baseline",
        epoch=31,
        model_version="rhythm-dilated-v1",
        config={"seed": 42},
        metadata={
            "code": {"revision": "synthetic-revision", "source_bundle_sha256": "c" * 64}
        },
    )
    model = torch.nn.Identity()
    write_new(tmp_path / "preprocessing.json", state.to_dict())
    write_new(tmp_path / "config.json", {"seed": 42})
    write_new(
        tmp_path / "reports/rhythm-multiseed-v1.json",
        {
            "runs": [
                dict(
                    seed=42,
                    architecture="rhythm-dilated-v1",
                    source_revision="synthetic-revision",
                    source_bundle_sha256="c" * 64,
                )
            ]
        },
    )
    if change == "seed":
        checkpoint["config"]["seed"] = 43
    elif change == "epoch":
        checkpoint["epoch"] = 30
    elif change == "kind":
        checkpoint["kind"] = "smoke"
    elif change == "scale":
        state = PreprocessingState(PREPROCESSING_VERSION, 0.5, "train", "fit", 1, 0)
    elif change == "config":
        checkpoint["config"]["other"] = 1
    elif change == "source":
        checkpoint["metadata"]["code"]["revision"] = "changed"

    def loader(path, **kwargs):
        assert path == tmp_path / "synthetic-unopened.pt"
        assert kwargs["expected_data_identity"] == {"train_sha256": "train"}
        return model, state, checkpoint

    monkeypatch.setattr(rhythm_training, "load_checkpoint", loader)
    backend = object.__new__(phase.Backend)
    backend.root = tmp_path
    backend.records = SimpleNamespace(identity={"train_sha256": "train"})
    backend.loaded = {}
    backend.c = {
        "models": [
            dict(
                seed=42,
                selected_epoch=31,
                checkpoint="synthetic-unopened.pt",
                preprocessing="preprocessing.json",
                resolved_config="config.json",
            )
        ]
    }
    if change:
        with pytest.raises(ValueError):
            backend.state_and_model(42)
    else:
        _, observed = backend.state_and_model(42)
        assert observed is model and not model.training
