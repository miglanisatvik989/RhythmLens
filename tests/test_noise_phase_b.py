"""Generated signals/metadata and mocked model/optimizer interfaces only."""

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

from rhythmlens import noise_phase_a as a
from rhythmlens import noise_phase_a_analysis as aa
from rhythmlens import noise_phase_b as b
from rhythmlens import noise_phase_b_analysis as ba
from rhythmlens import noise_phase_b_training as bt
from rhythmlens.calibration import read_json, sha, write_new
from rhythmlens.noise import prepare_pool, training_plan, apply_plan
from rhythmlens.preprocessing import transform_physical
from test_noise_phase_a import GeneratedBackend, rows

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/noise-phase-b-v1.yaml"


@pytest.fixture(autouse=True)
def protected_io_forbidden(monkeypatch):
    original = Path.open

    def guard(path, *args, **kwargs):
        if any(
            path.resolve().is_relative_to(ROOT / f)
            for f in ("data/raw", "outputs", "checkpoints", "weights")
        ):
            pytest.fail(f"Forbidden real artifact access: {path}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)


@pytest.fixture(scope="module")
def pools():
    t = np.arange(650000, dtype=np.int32)
    digital = np.column_stack((t % 201 - 100, (t * 7) % 301 - 150)).astype(np.int16)
    return {
        role: {
            n: prepare_pool(digital, record=n, role=role, source_sha256="a" * 64)
            for n in ("bw", "ma", "em")
        }
        for role in ("augmentation-train", "stress")
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("extra", 0),
        ("partition", ["holdout"]),
        ("models", []),
        ("baseline", {}),
        ("environments", {}),
        ("training_configs", {}),
        ("output_root", "outputs/new"),
        ("policy", {}),
        ("execution_authorized", "false"),
        ("execution_authorized", 1),
    ],
)
def test_contract_rejects_changes(tmp_path, key, value):
    c = yaml.safe_load(CONFIG.read_text())
    c[key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):
        b.load_config(path)


def test_authorization_before_any_protected_or_metadata_io(tmp_path, monkeypatch):
    c = b.load_config(CONFIG)
    c["execution_authorized"] = False
    disabled = tmp_path / "disabled.yaml"
    disabled.write_text(yaml.safe_dump(c))
    for method in ("metadata_preflight", "verify_baseline", "require_committed_source"):
        monkeypatch.setattr(
            b, method, lambda *a, **kw: pytest.fail("I/O before authority")
        )
    for stage in ("train", "evaluate"):
        for flag in (False, True):
            with pytest.raises(ValueError, match="execution_authorized"):
                b.execute(ROOT, disabled, stage, flag)
    c = b.load_config(CONFIG)
    c["execution_authorized"] = True
    path = tmp_path / "authorized.yaml"
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match="Explicit --authorize"):
        b.execute(ROOT, path, "train", False)
    for old in (
        "noise-robustness-v1.yaml",
        "noise-quality-datasets-v1.yaml",
        "noise-phase-a-v1.yaml",
    ):
        with pytest.raises(ValueError, match="schema"):
            b.load_config(ROOT / "configs" / old)
    path.write_text(CONFIG.read_text() + "\nexecution_authorized: false\n")
    with pytest.raises(ValueError, match="Duplicate"):
        b.load_config(path)


def test_declarations_from_tracked_metadata_only():
    c = b.load_config(CONFIG)
    r = read_json(ROOT / "reports/noise-phase-a-v1-results.json")
    assert c["baseline"]["identities"] == r["identities"]
    assert c["baseline"]["source_revision"] == r["source_revision"]
    assert list(c["training_configs"]) == [42, 43, 44]
    for file, digest in c["metadata_sha256"].items():
        assert sha(ROOT / file) == digest
    assert c["environments"]["train"]["torch"] == "2.8.0+cu128"
    assert c["environments"]["evaluate"]["torch"] == "2.8.0+cpu"


class Records:
    partition = "train"
    identity = {"train_sha256": "synthetic-train"}

    def __init__(self):
        self.rows = [{**r, "split": "train"} for r in rows()]
        self.calls = 0

    def load_physical(self, index):
        self.calls += 1
        return 0.3 + np.sin(np.arange(self.rows[index]["samples"]) / 11) * 0.2


def state(pools):
    return GeneratedBackend(rows(), pools["stress"], Path(".")).state


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_exact_training_draw_order_rng_isolation_and_epoch_change(pools, seed):
    rs = Records()
    train = pools["augmentation-train"]
    random.seed(719)
    np.random.seed(818)
    torch.manual_seed(919)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    plans = bt.epoch_plans(rs.rows, train, seed, 1)
    for row, plan in zip(rs.rows, plans, strict=True):
        key = [20261014, "train", seed, 1, row["record_id"]]
        number = int.from_bytes(
            hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).digest()[
                :8
            ],
            "little",
        )
        rng = np.random.Generator(np.random.PCG64(number))
        assert plan.seed == number
        if rng.random() < 0.5:
            n = ("bw", "ma", "em")[int(rng.integers(0, 3))]
            ch = int(rng.integers(0, 2))
            snr = (24, 18, 12, 6)[int(rng.integers(0, 4))]
            offset = int(rng.integers(0, len(train[n].values) - row["samples"] + 1))
            assert (plan.noise_record, plan.channel, plan.snr_db, plan.offset) == (
                n,
                ch,
                snr,
                offset,
            )
        else:
            assert plan.noise_record is None
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert before[0] == after[0] and np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:] and torch.equal(before[2], after[2])
    assert plans != bt.epoch_plans(rs.rows, train, seed, 2)
    with pytest.raises(ValueError, match="role"):
        bt.epoch_plans(rs.rows, pools["stress"], seed, 1)
    with pytest.raises(ValueError, match="partition"):
        bt.epoch_plans(rows(), train, seed, 1)


def test_physical_order_clean_cache_and_saved_preprocessing(pools):
    rs = Records()
    s = state(pools)
    saved = copy.deepcopy(s.to_dict())
    ds = bt.AugmentedDataset(rs, s, pools["augmentation-train"], 42)
    with pytest.raises(ValueError, match="Freeze"):
        ds[0]
    outputs = []
    for epoch in (1, 2):
        ds.start_epoch(epoch)
        for i, row in enumerate(rs.rows):
            x = 0.3 + np.sin(np.arange(row["samples"]) / 11) * 0.2
            plan = training_plan(
                row["record_id"],
                42,
                epoch,
                row["samples"],
                pools["augmentation-train"],
                partition="train",
            )
            expected, meta = apply_plan(
                x, plan, pools["augmentation-train"], partition="train"
            )
            out = ds[i]["signal"].numpy()
            assert np.array_equal(out, transform_physical(expected, s))
            assert ds.details[-1]["corruption"] == meta
            outputs.append(out)
    assert rs.calls == len(rs.rows)
    assert all(not x.flags.writeable for x in ds.cache.values())
    assert s.to_dict() == saved
    assert any(
        not np.array_equal(outputs[i], outputs[i + len(rs.rows)])
        for i in range(len(rs.rows))
    )
    wrong = copy.copy(rs)
    wrong.identity = {"train_sha256": "other"}
    with pytest.raises(ValueError, match="preprocessing"):
        bt.AugmentedDataset(wrong, s, pools["augmentation-train"], 42)


def test_flat_and_no_redraw_invalid_noise(pools):
    rs = Records()
    rs.load_physical = lambda i: np.zeros(rs.rows[i]["samples"])
    ds = bt.AugmentedDataset(rs, state(pools), pools["augmentation-train"], 42)
    ds.start_epoch(1)
    selected = next(i for i, p in enumerate(ds.plans) if p.noise_record is not None)
    assert np.all(ds[selected]["signal"].numpy() == 0)
    assert ds.details[-1]["corruption"]["achieved_snr_db"] is None
    # A mutated pool fails before application instead of searching another offset.
    pool = ds.pools[ds.plans[selected].noise_record]
    ds.pools = {
        **ds.pools,
        pool.record: replace(pool, values=np.zeros_like(pool.values)),
    }
    with pytest.raises(ValueError, match="Pool changed"):
        ds[selected]


class MockTrainer:
    def __init__(self, pools, metrics=(0.8, 0.8, 0.7, 0.9)):
        self.dataset = bt.AugmentedDataset(
            Records(), state(pools), pools["augmentation-train"], 42
        )
        self.state = self.dataset.state
        self.config = dict(epochs=10, patience=2, batch_size=16, seed=42)
        from rhythmlens.rhythm_training import make_loader

        self.loader = make_loader(self.dataset, self.config, shuffle=True)
        self.metrics = iter(metrics)
        self.saved = []
        self.steps = 0
        self.fail = False

    def initialize(self, folder):
        self.folder = folder

    def step(self, batch):
        self.steps += 1
        assert (self.folder / f"epoch{self.steps}" / "plans.json").exists()
        if self.fail:
            raise ValueError("mock optimizer failure")
        return 0.4, 2

    def validate(self):
        return {"macro_f1_four_classes": next(self.metrics)}

    def save(self, path, epoch, steps, history, best, stale, best_epoch):
        path.write_text(str(epoch))
        self.saved.append((path.name, epoch))


def test_original_strict_selection_and_epoch_ledger(tmp_path, pools):
    trainer = MockTrainer(pools)
    stages = []
    report = bt.train_seed(
        tmp_path / "seed42", 42, trainer, lambda s, **kw: stages.append(s)
    )
    assert report["best_epoch"] == 1 and report["completed_epochs"] == 3
    assert [e for n, e in trainer.saved if n == "best.pt"] == [1]
    assert stages.count("clean_validation") == 3
    assert read_json(tmp_path / "seed42/status.json")["state"] == "completed"
    for e in range(1, 4):
        meta = read_json(tmp_path / f"seed42/epoch{e}/batch0.json")
        assert len(meta["record_ids"]) == len(set(meta["record_ids"])) == 5
    with pytest.raises(FileExistsError):
        bt.train_seed(tmp_path / "seed42", 42, trainer, lambda *a, **kw: None)


def test_training_failure_preserves_plans_and_partial_context(tmp_path, pools):
    trainer = MockTrainer(pools)
    trainer.fail = True
    with pytest.raises(ValueError, match="mock optimizer"):
        bt.train_seed(tmp_path / "seed42", 42, trainer, lambda *a, **kw: None)
    assert read_json(tmp_path / "seed42/status.json")["state"] == "failed"
    assert (tmp_path / "seed42/epoch1/plans.json").exists()
    assert (
        len(
            read_json(tmp_path / "seed42/failure-context.json")[
                "available_corruption_metadata"
            ]
        )
        == 5
    )
    assert not (tmp_path / "seed42/run-report.json").exists()


@pytest.fixture
def generated_run(tmp_path, pools, monkeypatch):
    rs = rows()
    dest = tmp_path / "original"
    backend = GeneratedBackend(rs, pools["stress"], dest)
    monkeypatch.setattr(a, "bootstrap", lambda r, z: aa.bootstrap(r, z, replicates=3))
    a.run_verified(dest, rs, backend, {"configuration": "generated"}, lambda: None)
    augmented = tmp_path / "augmented"
    augmented.mkdir()
    write_new(augmented / "registry.json", {"generated": True})
    plans = read_json(dest / "plans.json")
    write_new(augmented / "plans.json", plans)
    augbackend = GeneratedBackend(rs, pools["stress"], augmented)
    return (
        rs,
        dest,
        augmented,
        augbackend,
        b.compatible_plans(rs, pools["stress"], plans),
    )


def test_paired_maps_full_evidence_and_comparison(generated_run, tmp_path, monkeypatch):
    rs, original, augmented, backend, plans = generated_run
    stages = []
    b.evaluate_augmented(
        augmented, rs, backend, plans, lambda s, **kw: stages.append(s), lambda: None
    )
    assert "clean_evaluation" in stages and "stress_evaluation" in stages
    z = b.sealed_augmented(augmented, rs)
    assert z.shape == (3, 55, 5, 4)
    monkeypatch.setattr(b, "completed_training", lambda folder: {})
    predictions, seal = b.comparative_seal(
        augmented, rs, original, {"generated": True}, original, lambda: None
    )
    assert predictions.shape == (2, 3, 55, 5, 4) and seal["seed_pairing"] == [
        42,
        43,
        44,
    ]
    assert (augmented / "comparison-seal.json").exists()
    result = ba.comparison(rs, predictions)
    assert result["augmented_minus_original_per_seed"]["42"]["clean"]["macro_f1"] == 0
    draws, stats, meta = ba.paired_bootstrap(rs, predictions, replicates=5)
    baseline_draws, baseline_stats, _ = aa.bootstrap(rs, predictions[0], replicates=5)
    assert np.array_equal(draws, baseline_draws)
    assert np.array_equal(stats[:, 0], baseline_stats)
    assert np.all(stats[:, 2] == 0)
    assert meta["shared_draws"] and len(meta["groups"]) == 4
    for group, row in zip(meta["draw_columns"], draws[0], strict=True):
        assert row >= 0
    # Mutation of a partial-batch evidence file blocks all comparative reads.
    p = augmented / "seed44/em-snr-6-repeat2/batch0.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="Input changed"):
        b.sealed_augmented(augmented, rs)


def test_missing_candidate_or_ledger_blocks_scoring(generated_run):
    rs, _, augmented, backend, plans = generated_run
    b.evaluate_augmented(
        augmented, rs, backend, plans, lambda *a, **kw: None, lambda: None
    )
    p = augmented / "seal.json"
    seal = read_json(p)
    complete_seal = copy.deepcopy(seal)
    del seal["predictions"]["seed44/clean/logits.npy"]
    a.atomic(p, seal)
    with pytest.raises(ValueError, match="All augmented"):
        b.sealed_augmented(augmented, rs)
    with pytest.raises(ValueError, match="All six"):
        ba.comparison(rs, np.zeros((1, 3, 55, 5, 4)))
    # A complete hash inventory cannot substitute for complete eligible accounting.
    ledger_path = augmented / "accounting.json"
    ledger = read_json(ledger_path)
    a.atomic(ledger_path, ledger[:-1])
    complete_seal["evidence"]["accounting.json"] = sha(ledger_path)
    a.atomic(p, complete_seal)
    with pytest.raises(ValueError, match="Incomplete augmented accounting"):
        b.sealed_augmented(augmented, rs)


def test_evaluation_failure_no_seal_or_metrics(generated_run):
    rs, _, augmented, backend, plans = generated_run
    backend.fail_at = 2
    with pytest.raises(ValueError, match="synthetic technical"):
        b.evaluate_augmented(
            augmented, rs, backend, plans, lambda *a, **kw: None, lambda: None
        )
    assert (
        not (augmented / "seal.json").exists()
        and not (augmented / "metrics.json").exists()
    )
    ledger = read_json(augmented / "accounting.json")
    assert len(ledger) == 3 * 55 * len(rs)
    assert any(r["status"] == "technical_failure" for r in ledger)
    assert (augmented / "seed42/clean/batch0.npy").exists()


def test_baseline_reuse_identity_and_collision(generated_run, tmp_path, monkeypatch):
    rs, original, _, _, _ = generated_run
    registry = read_json(original / "registry.json")
    registry.update(
        source_revision="generated",
        configuration={"generated": True},
        input_byte_sha256={},
        committed_files_sha256={},
    )
    a.atomic(original / "registry.json", registry)
    # Re-seal synthetic baseline after deliberately constructing its fixture registry.
    seal = read_json(original / "seal.json")
    seal["evidence"]["registry.json"] = sha(original / "registry.json")
    a.atomic(original / "seal.json", seal)
    a.atomic(
        original / "artifacts-sha256.json",
        b.inventory(original, exclude=("status.json", "artifacts-sha256.json")),
    )
    status = read_json(original / "status.json")
    status["artifacts_index_sha256"] = sha(original / "artifacts-sha256.json")
    a.atomic(original / "status.json", status)
    c = {
        "baseline": {
            "root": "original",
            "source_revision": "generated",
            "config": "generated.yaml",
            "identities": {
                f: dict(bytes=(original / f).stat().st_size, sha256=sha(original / f))
                for f in ("status.json", "seal.json", "artifacts-sha256.json")
            },
        }
    }
    monkeypatch.setattr(a, "load_config", lambda p: {"generated": True})
    assert b.verify_baseline(tmp_path, c, rs, arrays=True).shape == (3, 55, 5, 4)
    wrong = copy.deepcopy(c)
    wrong["baseline"]["source_revision"] = "wrong"
    with pytest.raises(ValueError, match="revision"):
        b.verify_baseline(tmp_path, wrong, rs)
    (original / "seed42/clean/logits.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="inventory"):
        b.verify_baseline(tmp_path, c, rs)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError):
        b.run_training_stage(occupied, {}, lambda: pytest.fail("work on collision"))


def test_progress_measured_eta_and_indeterminate_scoring(tmp_path, monkeypatch, capsys):
    clock = [100.0]
    monkeypatch.setattr(bt.time, "monotonic", lambda: clock[0])
    progress = bt.Progress(tmp_path)
    progress("training", seed=43, epoch=2, completed=0, total=10)
    assert read_json(tmp_path / "progress.json")["remaining_stage_seconds"] is None
    clock[0] += 4
    progress("training", seed=43, epoch=2, completed=2, total=10)
    p = read_json(tmp_path / "progress.json")
    assert p["batches_per_second"] == 0.5 and p["remaining_stage_seconds"] == 16
    progress("bootstrap_scoring")
    assert read_json(tmp_path / "progress.json")["total"] is None
    assert "[...]" in capsys.readouterr().out


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_production_training_wires_original_settings_with_mock_model(
    tmp_path, pools, monkeypatch, seed
):
    from rhythmlens import rhythm_training as rt
    from rhythmlens import rhythm_model
    from rhythmlens.configuration import load_experiment

    c = b.load_config(CONFIG)
    original, document = load_experiment(ROOT / c["training_configs"][seed], ROOT)
    s = state(pools)
    s = replace(s, scale_mv=0.18008390945457742)
    model = SimpleModel()
    captured = {}
    monkeypatch.setattr(
        bt,
        "read_json",
        lambda p: s.to_dict() if str(p).endswith("preprocessing.json") else original,
    )
    monkeypatch.setattr(
        rt,
        "setup_runtime",
        lambda config: captured.setdefault("runtime", config) and torch.device("cpu"),
    )
    monkeypatch.setattr(
        rhythm_model,
        "build_model",
        lambda name, config: captured.setdefault("architecture", (name, config))
        and model,
    )

    def optimizer(parameters, **kw):
        captured["optimizer"] = kw
        return "mock-optimizer"

    monkeypatch.setattr(torch.optim, "AdamW", optimizer)
    monkeypatch.setattr(rt, "code_metadata", lambda *a: {})
    monkeypatch.setattr(rt, "environment_metadata", lambda *a: {})
    monkeypatch.setattr(rt, "reproduction_metadata", lambda *a: {})
    monkeypatch.setattr(
        rt, "save_checkpoint", lambda *a, **kw: captured.setdefault("checkpoint", kw)
    )
    train = Records()
    validation = Records()
    validation.partition = "validation"
    trainer = bt.CUDATrainer(
        ROOT,
        c,
        seed,
        train,
        validation,
        pools["augmentation-train"],
        {"source_revision": "fixture"},
    )
    trainer.initialize(tmp_path)
    assert trainer.config == original
    assert trainer.state.to_dict() == s.to_dict()
    assert captured["optimizer"] == {
        "lr": original["learning_rate"],
        "weight_decay": original["weight_decay"],
    }
    assert captured["architecture"] == (original["model_version"], original["model"])
    assert trainer.loader.batch_sampler.seed == seed
    assert trainer.loader.batch_sampler.shuffle is True
    assert trainer.validation_loader.batch_sampler.shuffle is False
    trainer.save(tmp_path / "mock.pt", 2, 4, [{"epoch": 1}, {"epoch": 2}], 0.8, 0, 2)
    assert captured["checkpoint"]["metadata"]["phase_b"] is True
    assert captured["checkpoint"]["rng"]["train_loader"].dtype == torch.uint8
    # A numerical change from the bound original stops before runtime/model setup.
    changed = copy.deepcopy(original)
    changed["learning_rate"] *= 2
    monkeypatch.setattr(bt, "read_json", lambda p: changed)
    with pytest.raises(ValueError, match="resolved numerical"):
        trainer.initialize(tmp_path)


class SimpleModel:
    def to(self, device):
        return self

    def parameters(self):
        return []

    def eval(self):
        return self


def test_nonzero_paired_effect_and_group_bootstrap_not_seed_pooling():
    rs = rows()
    # Deterministic mocked logits: original always N; augmented exactly correct.
    z = np.zeros((2, 3, 55, len(rs), 4), dtype=np.float32)
    z[0, :, :, :, 0] = 2
    for i, r in enumerate(rs):
        z[1, :, :, i, aa.LABELS.index(r["label"])] = 2
    result = ba.comparison(rs, z)
    change = result["augmented_minus_original_equal_seed_summary"]["clean"]
    assert change["automatic_af_sensitivity"]["mean"] == 1
    assert change["automatic_error"]["mean"] == pytest.approx(-0.6)
    draws, stats, meta = ba.paired_bootstrap(rs, z, replicates=7)
    assert np.all(stats[:, 2, :, :, 0, 1] == 1)
    assert np.all(stats[:, 2, :, :, 1, :] == 0)  # same effect at clean and stress
    assert draws.shape == (7, 4)  # duplicate N members kept together
    assert meta["replicates"] == 7
    bad = copy.deepcopy(rs)
    bad[1]["label"] = "A"
    with pytest.raises(ValueError, match="Conflicting"):
        ba.paired_bootstrap(bad, z, replicates=2)
    no_af = rows(("N", "N", "O", "O", "~"))
    _, _, uncertainty = ba.paired_bootstrap(no_af, z, replicates=2)
    af = uncertainty["intervals"]["augmented_minus_original"]["42"]["clean"][
        "absolute_or_arm_difference"
    ]["automatic_af_sensitivity"]
    assert af["interval"] is None and af["undefined_reason"] == "no_AF_labels"


def test_training_stage_failed_work_preserved_and_no_automatic_retry(tmp_path):
    folder = tmp_path / "training"

    def fail():
        write_new(folder / "available.json", {"fixture": True})
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        b.run_training_stage(folder, {"fixture": True}, fail)
    assert read_json(folder / "status.json")["state"] == "interrupted"
    assert (folder / "available.json").exists()
    assert not (folder / "artifacts-sha256.json").exists()
    with pytest.raises(FileExistsError):
        b.run_training_stage(folder, {}, lambda: None)


def test_complete_two_stage_workflow_uses_baseline_once_and_seals_before_scoring(
    generated_run, tmp_path, pools, monkeypatch
):
    rs, original, _, _, _ = generated_run
    before = b.inventory(original)
    c = b.load_config(CONFIG)
    c["execution_authorized"] = True
    c["output_root"] = "phase-b"
    c["baseline"]["root"] = "original"
    monkeypatch.setattr(b, "load_config", lambda path: c)
    monkeypatch.setattr(b, "require_committed_source", lambda *a: None)
    monkeypatch.setattr(b, "metadata_preflight", lambda *a: (Records(), Records(), rs))
    monkeypatch.setattr(a, "protected_identities", lambda *a: {})
    monkeypatch.setattr(
        b.subprocess,
        "check_output",
        lambda args, **kw: "" if args[1] == "ls-files" else "synthetic",
    )
    monkeypatch.setattr(
        b, "verify_baseline", lambda *a, **kw: read_json(original / "plans.json")
    )
    monkeypatch.setattr(b, "reviewed_pools", lambda root, c, role: pools[role])

    def trainer(root, c, seed, *args):
        mock = MockTrainer(pools)
        mock.dataset.seed = seed
        mock.config["seed"] = seed
        return mock

    monkeypatch.setattr(b, "CUDATrainer", trainer)
    monkeypatch.setattr(a, "runtime_versions", lambda: c["environments"]["train"])
    b.execute(tmp_path, CONFIG, "train", True)
    destination = tmp_path / "phase-b"
    assert read_json(destination / "status.json")["state"] == "training_completed"
    assert not (destination / "evaluation").exists()
    assert (
        b.completed_training(destination / "training")["source_revision"] == "synthetic"
    )
    monkeypatch.setattr(a, "runtime_versions", lambda: c["environments"]["evaluate"])
    monkeypatch.setattr(
        b,
        "EvaluationBackend",
        lambda root, c, records, training: GeneratedBackend(
            rs, pools["stress"], destination / "evaluation"
        ),
    )

    def compare(rows, predictions):
        assert (destination / "evaluation/comparison-seal.json").exists()
        assert predictions.shape == (2, 3, 55, 5, 4)
        return ba.comparison(rows, predictions)

    monkeypatch.setattr(b, "comparison", compare)
    monkeypatch.setattr(
        b,
        "paired_bootstrap",
        lambda rows, predictions: ba.paired_bootstrap(rows, predictions, replicates=3),
    )
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        b.execute(tmp_path, CONFIG, "evaluate", True)
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
    status = read_json(destination / "status.json")
    assert status["state"] == "completed"
    evaluation = destination / "evaluation"
    state = read_json(evaluation / "status.json")
    b.verify_index(
        evaluation,
        "artifacts-sha256.json",
        state["artifacts_index_sha256"],
        exclude=("status.json",),
    )
    assert (evaluation / "metrics.json").exists()
    assert b.inventory(original) == before
    with pytest.raises(ValueError, match="exactly one completed training"):
        b.execute(tmp_path, CONFIG, "evaluate", True)
    assert not (tmp_path / b.OUTPUT).exists()


def test_mismatching_maps_or_physical_evidence_block_pairing(generated_run, pools):
    rs, original, augmented, backend, plans = generated_run
    manifest = read_json(original / "plans.json")
    bad = copy.deepcopy(manifest)
    bad["plans"][0]["offset"] += 1
    with pytest.raises(ValueError, match="incompatible"):
        b.compatible_plans(rs, pools["stress"], bad)
    b.evaluate_augmented(
        augmented, rs, backend, plans, lambda *a, **kw: None, lambda: None
    )
    b.verify_pairing(augmented, original, rs)
    path = augmented / "seed43/bw-snr18-repeat1/batch0.json"
    meta = read_json(path)
    meta["cases"][0]["post_transform_sha256"] = "0" * 64
    a.atomic(path, meta)
    with pytest.raises(ValueError, match="physical inputs"):
        b.verify_pairing(augmented, original, rs)


def test_dangling_fixed_root_symlink_blocks_before_metadata(tmp_path, monkeypatch):
    c = b.load_config(CONFIG)
    c["execution_authorized"] = True
    c["output_root"] = "fixed"
    (tmp_path / "fixed").symlink_to(tmp_path / "missing", target_is_directory=True)
    monkeypatch.setattr(b, "load_config", lambda path: c)
    monkeypatch.setattr(b, "require_committed_source", lambda *a: None)
    monkeypatch.setattr(a, "runtime_versions", lambda: c["environments"]["train"])
    monkeypatch.setattr(
        b, "metadata_preflight", lambda *a: pytest.fail("metadata on collision")
    )
    with pytest.raises(ValueError, match="symlink collision"):
        b.execute(tmp_path, CONFIG, "train", True)
    assert not (tmp_path / "missing").exists()
