"""Synthetic ECG only, generated in tmp_path; no real holdout or saved predictions.

Generator seed 2017 (shared fixture), split seed 42, model seed 20261014.
Tests exercise dedicated inference with freshly initialized synthetic checkpoints;
no model fitting, production checkpoint loading or repository output creation.
"""

import copy
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
import yaml

from rhythmlens import final_evaluation as final
from rhythmlens.preprocessing import fit_preprocessing
from rhythmlens.rhythm_data import FrozenRecords
from rhythmlens.rhythm_model import build_model
from rhythmlens.rhythm_training import DEFAULT_CONFIG, save_checkpoint
from rhythmlens.splits import SPLIT_VERSION
from test_rhythm_baseline import frozen_dataset

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/temporal-final-evaluation-v1.yaml"


def rows(labels=("N", "A", "O", "~")):
    return [
        dict(record_id=f"synthetic-{i}", label=label, duplicate_group=f"g{i}")
        for i, label in enumerate(labels)
    ]


def test_fixed_config_authorization_and_policy(tmp_path):
    c = final.load_config(CONFIG)
    assert c["execution_authorized"] is False
    for key, value in [
        ("execution_authorized", "true"),
        ("output_root", "outputs/bypass"),
        ("class_supports", [761, 114, 361, 42]),
    ]:
        changed = copy.deepcopy(c)
        changed[key] = value
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError):
            final.load_config(path)
    c["execution_authorized"] = True
    path = tmp_path / "authorized.yaml"
    path.write_text(yaml.safe_dump(c))
    assert final.load_config(path)["execution_authorized"] is True


def test_unauthorized_never_reads_inputs_or_creates_output(tmp_path, monkeypatch):
    c = final.load_config(CONFIG)
    monkeypatch.setattr(
        final, "metadata_preflight", lambda *a: pytest.fail("input access")
    )
    for authorized in (False, True):
        with pytest.raises(ValueError, match="execution_authorized"):
            final.execute(tmp_path, CONFIG, c, authorized)
    c["execution_authorized"] = True
    with pytest.raises(ValueError, match="authorize-evaluation"):
        final.execute(tmp_path, CONFIG, c, False)
    assert not (tmp_path / "outputs").exists()


def test_real_authority_requires_clean_committed_config(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    path = tmp_path / "config.yaml"
    path.write_text("execution_authorized: true\n")
    with pytest.raises(ValueError, match="clean tree"):
        final.authorize(tmp_path, path, {"execution_authorized": True}, True)
    git("add", "config.yaml")
    git(
        "-c",
        "user.name=Synthetic Test",
        "-c",
        "user.email=synthetic@example.invalid",
        "commit",
        "-m",
        "synthetic authority fixture",
    )
    final.authorize(tmp_path, path, {"execution_authorized": True}, True)
    path.write_text("execution_authorized: false\n")
    with pytest.raises(ValueError, match="clean tree"):
        final.authorize(tmp_path, path, {"execution_authorized": True}, True)


def test_raw_metrics_accounting_ties_noisy_and_undefined_af():
    cases = rows(("A", "A", "N", "O", "~"))
    z = np.array(
        [[0, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 4], [0, 0, 4, 0], [0, 0, 0, 4]],
        dtype=np.float32,
    )
    ledger, m = final.raw_metrics(cases, z)
    assert ledger[0]["predicted_label"] == "N"
    assert m["eligible"] == m["accepted"] == 5
    assert m["af_outcomes"] == [1, 1, 0, 0, 0, 0, 0]
    assert m["automatic_af_sensitivity"]["value"] == 0.5
    assert m["deferral_rate"]["numerator"] == 0
    assert m["rhythm_output_coverage"]["numerator"] == 3
    assert m["probability_metrics"]["binary_af_eligible"] == 4
    assert m["probability_metrics"]["binary_af_excluded_noisy"] == 1
    _, missing = final.raw_metrics(rows(("N",)), np.zeros((1, 4), dtype=np.float32))
    assert missing["automatic_af_sensitivity"]["value"] is None
    assert missing["macro_f1"] == 0.25
    assert missing["zero_support_classes"] == ["A", "O", "~"]


def test_bootstrap_shared_groups_reproducible_and_zero_af():
    cases = rows(("N", "N", "A", "A", "O", "~"))
    cases[1]["duplicate_group"] = cases[0]["duplicate_group"]
    z = np.eye(4)[[0, 0, 1, 0, 2, 3]].astype(np.float32)
    stats, report = final.bootstrap(cases, [z, z, z], replicates=100)
    other, _ = final.bootstrap(cases, [z, z, z], replicates=100)
    np.testing.assert_array_equal(stats, other)
    np.testing.assert_array_equal(stats[:, 0], stats[:, 1])
    np.testing.assert_array_equal(stats[:, 0], stats[:, 3])
    assert report["groups"]["g0"] == ["synthetic-0", "synthetic-1"]
    assert len(report["intervals"]) == 4
    cases = rows(("N", "O", "~"))
    z = np.zeros((3, 4), dtype=np.float32)
    stats, report = final.bootstrap(cases, [z, z, z], replicates=10)
    assert np.isnan(stats[:, :, 1]).all()
    assert report["intervals"]["42"]["automatic_af_sensitivity"]["interval"] is None
    cases[1]["duplicate_group"] = cases[0]["duplicate_group"]
    with pytest.raises(ValueError, match="Conflicting"):
        final.bootstrap(cases, [z, z, z], replicates=10)


@pytest.fixture
def synthetic_execution(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(20261014)
    root = frozen_dataset(tmp_path)
    split = f"splits/{SPLIT_VERSION}"
    train = FrozenRecords(root, split, "train")
    state = fit_preprocessing(train)
    manifest = final.csv_rows(root / split / "holdout.csv")
    c = copy.deepcopy(final.load_config(CONFIG))
    c.update(
        split_dir=split,
        data_identity=train.identity,
        models=[],
        holdout_sha256=final.sha(root / split / "holdout.csv"),
    )
    audit_path = root / c["integrity_audit"]
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("Synthetic exposure audit fixture; no real patient data.\n")
    config_path = root / "final.yaml"
    config_path.write_text("synthetic configuration\n")
    for seed in final.SEEDS:
        cfg = {
            **DEFAULT_CONFIG,
            "seed": seed,
            "model_version": "rhythm-dilated-v1",
            "model": dict(
                stem_channels=8,
                stem_kernel=15,
                channels=[8, 16, 24],
                kernel=7,
                dropout=0.1,
            ),
        }
        model = build_model(cfg["model_version"], cfg["model"])
        base = root / f"synthetic-seed{seed}"
        base.mkdir()
        final.write_new(base / "preprocessing.json", state.to_dict())
        final.write_new(base / "config.json", cfg)
        save_checkpoint(
            base / "best.pt",
            model,
            torch.optim.AdamW(model.parameters()),
            state,
            cfg,
            train.identity,
            epoch=1,
            steps=0,
            kind="baseline",
            metadata={
                "code": dict(revision="synthetic", source_bundle_sha256="synthetic")
            },
        )
        c["models"].append(
            dict(
                seed=seed,
                best_epoch=1,
                checkpoint=f"{base.name}/best.pt",
                preprocessing=f"{base.name}/preprocessing.json",
                resolved_config=f"{base.name}/config.json",
                source_revision="synthetic",
                source_bundle_sha256="synthetic",
            )
        )
    # Replace only external authority/runtime/production metadata, preserving real
    # synthetic decoding, transforms, checkpoint reload, inference and scoring.
    monkeypatch.setattr(final, "authorize", lambda *a: None)
    monkeypatch.setattr(
        final,
        "metadata_preflight",
        lambda *a: (dict(inputs_sha256={}, data_identity=train.identity), manifest),
    )
    monkeypatch.setattr(final, "runtime_preflight", lambda *a: {"synthetic": True})
    original = subprocess.check_output

    def synthetic_git(args, **kwargs):
        if args[:2] == ["git", "ls-files"]:
            return b""
        if args[:2] == ["git", "rev-parse"]:
            return b"synthetic\n"
        return original(args, **kwargs)

    monkeypatch.setattr(final.subprocess, "check_output", synthetic_git)
    return root, config_path, c, manifest


def test_end_to_end_synthetic_three_checkpoints(synthetic_execution, monkeypatch):
    root, path, c, manifest = synthetic_execution
    original = final.score_sealed

    def check_seals(destination, rows, seals):
        assert set(seals) == {"42", "43", "44"}
        assert not list(destination.glob("seed*/metrics.json"))
        return original(destination, rows, seals)

    monkeypatch.setattr(final, "score_sealed", check_seals)
    destination = final.execute(root, path, c)
    summary = final.read_json(destination / "summary.json")
    assert set(summary["per_seed"]) == {"42", "43", "44"}
    assert summary["holdout_consumed_for_future_development"] is True
    assert summary["quality_aware_validation"] is False
    for result in summary["per_seed"].values():
        assert result["accepted"] == len(manifest)
        assert result["deferral_rate"]["numerator"] == 0
        assert result["technical_failures"] == 0
    status = final.read_json(destination / "status.json")
    assert status["state"] == "completed"
    for name, digest in final.read_json(destination / "artifact-sha256.json").items():
        assert final.sha(destination / name) == digest
    with pytest.raises(ValueError, match="collision"):
        final.execute(root, path, c)


@pytest.mark.parametrize("failure_seed", [42, 43, 44])
def test_prediction_failure_blocks_all_scoring(
    synthetic_execution, monkeypatch, failure_seed
):
    root, path, c, manifest = synthetic_execution
    original = final.predict_seed

    def fail(root, c, m, rows, callback):
        if m["seed"] == failure_seed:
            callback(rows[:2], "started")
            raise ValueError("synthetic inference failure")
        return original(root, c, m, rows, callback)

    monkeypatch.setattr(final, "predict_seed", fail)
    monkeypatch.setattr(
        final, "score_sealed", lambda *a: pytest.fail("scoring after failure")
    )
    with pytest.raises(ValueError, match="synthetic inference failure"):
        final.execute(root, path, c)
    destination = root / c["output_root"]
    assert final.read_json(destination / "status.json")["state"] == "failed"
    assert not (destination / "summary.json").exists()
    assert not list(destination.glob("seed*/metrics.json"))
    accounting = final.read_json(destination / "execution-accounting.json")
    for cases in accounting.values():
        assert len(cases) == len(manifest)
        assert len({r["record_id"] for r in cases}) == len(manifest)
        assert all(
            r["confidence_deferred"] == r["quality_deferred"] == 0 for r in cases
        )
    assert accounting[str(failure_seed)][0]["reason"] == "batch_in_progress_or_failed"
    with pytest.raises(ValueError, match="collision"):
        final.execute(root, path, c)


def test_missing_seal_and_changed_seal_block_scoring(tmp_path):
    with pytest.raises(ValueError, match="All three"):
        final.score_sealed(tmp_path, rows(), {"42": "missing"})
    for seed in final.SEEDS:
        p = tmp_path / f"seed{seed}"
        p.mkdir()
        with (p / "logits.npy").open("xb") as stream:
            np.save(stream, np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="changed"):
        final.score_sealed(tmp_path, rows(), dict.fromkeys(["42", "43", "44"], "bad"))


def test_missing_synthetic_signal_blocks_before_prediction(
    synthetic_execution, monkeypatch
):
    root, path, c, manifest = synthetic_execution
    source = {
        r["record_id"]: r
        for r in final.csv_rows(root / "data/manifests/challenge2017-v3/records.csv")
    }
    (root / source[manifest[-1]["record_id"]]["mat_path"]).unlink()
    monkeypatch.setattr(
        final,
        "predict_seed",
        lambda *a: pytest.fail("prediction before all input hashes checked"),
    )
    with pytest.raises(FileNotFoundError):
        final.execute(root, path, c)
    destination = root / c["output_root"]
    assert final.read_json(destination / "status.json")["state"] == "failed"
    assert not list(destination.glob("seed*/logits.npy"))
    for summary in final.read_json(destination / "accounting-summary.json").values():
        assert summary["technical_failure_or_unresolved"] == len(manifest)
        assert summary["accepted"] == 0


def test_unsealed_predictions_are_unresolved_on_write_failure(
    synthetic_execution, monkeypatch
):
    root, path, c, manifest = synthetic_execution

    def fail(*args, **kwargs):
        raise OSError("synthetic persistence failure")

    monkeypatch.setattr(final.np, "save", fail)
    with pytest.raises(OSError, match="persistence failure"):
        final.execute(root, path, c)
    destination = root / c["output_root"]
    accounting = final.read_json(destination / "execution-accounting.json")
    assert all(
        r["disposition"] == "technical_failure"
        and r["reason"] == "prediction_not_sealed"
        for r in accounting["42"]
    )
    assert (
        final.read_json(destination / "accounting-summary.json")["42"]["accepted"] == 0
    )
    assert not (destination / "summary.json").exists()


def test_invalid_logits_never_enter_bootstrap():
    z = np.full((4, 4), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid bootstrap"):
        final.bootstrap(rows(), [z, z, z], replicates=10)
