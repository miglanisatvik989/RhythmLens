"""Synthetic logits/metadata only. No ECG fixtures or real analysis inputs."""

import copy
import csv
import io
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from rhythmlens import calibration as a

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return a.load_config(ROOT / "configs/temporal-calibration-deferral-v1.yaml")


def synthetic():
    manifest, rows = [], []
    for k, label in enumerate(a.LABELS):
        for i in range(12):
            rid = f"synthetic-{k}-{i:02}"
            manifest.append(
                dict(
                    record_id=rid,
                    label=label,
                    subject_id="unknown",
                    duplicate_group=f"group-{k}-{i // 2}",
                    split="validation",
                )
            )
            logits = np.array([-0.5, -0.2, 0.1, -0.3])
            logits[k if i % 5 else (k + 1) % 4] += 2 + i / 20
            p = a.probabilities(logits[None, :])[0].astype(np.float32)
            rows.append(
                dict(
                    record_id=rid,
                    true_label=label,
                    predicted_label=a.LABELS[int(logits.argmax())],
                    **{
                        f"logit_{label_name}": float(logits[j])
                        for j, label_name in enumerate(a.LABELS)
                    },
                    **{
                        f"probability_{label_name}": float(p[j])
                        for j, label_name in enumerate(a.LABELS)
                    },
                )
            )
    return manifest, rows


def save_csv(path, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    a.write_new(path, stream.getvalue().encode())


@pytest.fixture
def repository(tmp_path, config):
    """Build an entirely synthetic provenance chain, not a copy of predictions."""
    c = copy.deepcopy(config)
    manifest, rows = synthetic()
    c["inputs"]["eligible_records"] = len(manifest)
    c["inputs"]["eligible_af_records"] = 12
    split = tmp_path / c["inputs"]["split_dir"]
    save_csv(split / "validation.csv", manifest)
    for part in ("train", "holdout"):
        save_csv(
            split / f"{part}.csv",
            [dict(manifest[0], record_id=part, duplicate_group=part)],
        )
    a.write_new(
        split / "frozen.json",
        {
            "assignment_sha256": c["inputs"]["assignment_sha256"],
            "files_sha256": {
                f"{p}.csv": a.sha(split / f"{p}.csv")
                for p in ("train", "validation", "holdout")
            },
        },
    )
    identity = dict(
        validation_sha256=a.sha(split / "validation.csv"),
        frozen_metadata_sha256=a.sha(split / "frozen.json"),
        assignment_sha256=c["inputs"]["assignment_sha256"],
    )
    for name in ("source_manifest", "source_audit", "source_provenance"):
        a.write_new(tmp_path / c["inputs"][name], b"synthetic source metadata")
        identity[f"{name}_sha256"] = a.sha(tmp_path / c["inputs"][name])
    for path in c["provenance"]["dependency_locks"]:
        a.write_new(tmp_path / path, b"synthetic lock, not a dependency installation")
        c["provenance"]["dependency_locks"][path] = a.sha(tmp_path / path)
    runs = []
    for model in c["inputs"]["models"]:
        seed = model["seed"]
        model["prediction_path"] = None
        model["prediction_sha256"] = None
        train = yaml.safe_load((ROOT / model["config"]).read_text())
        a.write_new(tmp_path / model["config"], yaml.safe_dump(train).encode())
        base = (
            tmp_path
            / train["paths"]["output_root"]
            / train["experiment"]["name"]
            / train["environment"]["profile"]
            / f"seed{seed}"
        )
        a.write_new(base / "train/run-report.json", {"synthetic": True})
        save_csv(base / "evaluate/validation-predictions.csv", rows)
        valid, z, y = a.validate_predictions(rows, manifest)
        cm = np.zeros((4, 4), dtype=int)
        np.add.at(cm, (y, z.argmax(axis=1)), 1)
        ev = dict(
            reproduction={"experiment": {"document": train}},
            partition="validation",
            environment={"device": "cpu"},
            holdout_signals_accessed=False,
            checkpoint_sha256=model["checkpoint_sha256"],
            data_identity=identity,
            training_provenance={
                "preprocessing_sha256": c["inputs"]["preprocessing_sha256"],
                "code": {"revision": "synthetic", "source_bundle_sha256": "synthetic"},
                "configuration_sha256": "synthetic",
            },
            metrics={
                "records": len(rows),
                "confusion_matrix": cm.tolist(),
                "cross_entropy": a.nll(z, y),
            },
        )
        a.write_new(base / "evaluate/evaluation-report.json", ev)
        runs.append(
            dict(
                architecture=c["inputs"]["architecture"],
                seed=seed,
                config_path=model["config"],
                checkpoint_sha256=model["checkpoint_sha256"],
                best_epoch=model["best_epoch"],
                yaml_sha256=a.sha(tmp_path / model["config"]),
                evaluation_report_sha256=a.sha(
                    base / "evaluate/evaluation-report.json"
                ),
                run_report_sha256=a.sha(base / "train/run-report.json"),
                preprocessing_sha256=c["inputs"]["preprocessing_sha256"],
                source_revision="synthetic",
                source_bundle_sha256="synthetic",
                resolved_config_sha256="synthetic",
            )
        )
    a.write_new(
        tmp_path / c["inputs"]["report"],
        dict(
            runs=runs,
            data_identity=identity,
            dependency_locks_sha256=c["provenance"]["dependency_locks"],
        ),
    )
    a.write_new(tmp_path / c["protocol_document"], b"synthetic protocol only")
    path = tmp_path / "analysis.yaml"
    a.write_new(path, yaml.safe_dump(c).encode())
    return tmp_path, c, path


def test_roles_deterministic_and_indivisible():
    manifest, _ = synthetic()
    first = a.assign_roles(manifest, 17)
    assert first == a.assign_roles(list(reversed(manifest)), 17)
    assert {r["record_id"] for r in first} == {r["record_id"] for r in manifest}
    assert len(first) == len(manifest)
    for group in {r["duplicate_group"] for r in first}:
        assert len({r["role"] for r in first if r["duplicate_group"] == group}) == 1
    for role in a.ROLES:
        assert {r["label"] for r in first if r["role"] == role} == set(a.LABELS)
    assert sum(r["label"] == "A" for r in first) == 12


@pytest.mark.parametrize(
    "mutation", ["conflict", "subject", "missing_group", "few_groups", "duplicate_id"]
)
def test_grouping_rejects(mutation):
    manifest, _ = synthetic()
    if mutation == "conflict":
        manifest[0]["label"] = "A"
    elif mutation == "subject":
        manifest[0]["subject_id"] = "verified"
    elif mutation == "missing_group":
        manifest[0]["duplicate_group"] = ""
    elif mutation == "few_groups":
        for r in manifest:
            r["duplicate_group"] = r["label"]
    else:
        manifest[0]["record_id"] = manifest[1]["record_id"]
    with pytest.raises(ValueError):
        a.assign_roles(manifest, 17)


def test_deterministic_fit_unchanged_argmax_and_convex_objective(config):
    manifest, rows = synthetic()
    _, z, y = a.validate_predictions(rows, manifest)
    fit = a.fit_temperature(z, y, config["calibration"])
    assert fit == a.fit_temperature(z, y, config["calibration"])
    assert fit["objective"] <= a.nll(z, y)
    assert (
        fit["objective"]
        <= min(a.nll(z, y, b) for b in np.linspace(0.05, 20, 200)) + 1e-10
    )
    assert np.array_equal(
        z.argmax(axis=1), a.probabilities(z, fit["beta"]).argmax(axis=1)
    )
    assert math.isclose(fit["temperature"] * fit["beta"], 1)
    with pytest.raises(ValueError, match="converge"):
        a.fit_temperature(z, y, dict(config["calibration"], max_iterations=0))
    flat = a.fit_temperature(np.zeros((4, 4)), np.arange(4), config["calibration"])
    assert flat["beta"] == 1


def test_threshold_selection_against_independent_enumeration(config):
    manifest, rows = synthetic()
    _, z, y = a.validate_predictions(rows, manifest)
    result = a.select_threshold(z, y, 1.0, config["selection"])
    p = a.probabilities(z)
    possibilities = []
    for j in range(101):
        accepted = [i for i in range(len(y)) if max(p[i]) >= j / 100]
        af = sum(y[i] == 1 for i in accepted)
        if len(accepted) * 10 >= 9 * len(y) and af * 10 >= 9 * sum(y == 1):
            error = sum(int(p[i].argmax()) != y[i] for i in accepted)
            possibilities.append((error / len(accepted), -len(accepted), j / 100))
    assert result["threshold"] == min(possibilities)[2]
    assert len(result["table"]) == 101
    with pytest.raises(ValueError, match="AF support"):
        a.select_threshold(z[y != 1], y[y != 1], 1, config["selection"])
    tied = a.select_threshold(np.zeros((4, 4)), np.arange(4), 1.0, config["selection"])
    assert tied["threshold"] == 0


def test_noisy_binary_scope_undefined_and_complete_ledger():
    # True noisy p_A=.9 must not count as an AF-negative Brier error.
    cases = [
        dict(
            record_id="n",
            true_label="N",
            predicted_label="N",
            disposition="accepted",
            probabilities=[0.7, 0.2, 0.05, 0.05],
            nll=-math.log(0.7),
        ),
        dict(
            record_id="a",
            true_label="A",
            predicted_label="~",
            disposition="accepted",
            probabilities=[0.1, 0.3, 0.1, 0.5],
            nll=-math.log(0.3),
        ),
        dict(
            record_id="d",
            true_label="A",
            predicted_label="A",
            disposition="confidence_deferred",
            probabilities=[0.2, 0.4, 0.2, 0.2],
            nll=-math.log(0.4),
        ),
        dict(
            record_id="q",
            true_label="~",
            predicted_label="A",
            disposition="accepted",
            probabilities=[0.03, 0.9, 0.03, 0.04],
            nll=-math.log(0.04),
        ),
    ]
    m = a.metrics(cases)
    assert m["eligible"] == 4 and m["accepted"] == 3
    assert sum(map(sum, m["outcome_matrix"])) == 4
    assert m["af_outcomes"] == [0, 0, 0, 1, 1, 0, 0]
    assert m["automatic_af_sensitivity"]["denominator"] == 2
    assert m["automatic_af_sensitivity"]["value"] == 0
    assert m["accepted_af_recall"]["denominator"] == 1
    assert m["rhythm_output_coverage"]["numerator"] == 2
    pm = m["probability_metrics"]
    assert pm["binary_af_eligible"] == 3 and pm["binary_af_excluded_noisy"] == 1
    assert pm["binary_af_brier"]["value"] == pytest.approx(
        (0.2**2 + 0.7**2 + 0.6**2) / 3
    )
    assert sum(b["count"] for b in pm["binary_af_reliability"]["bins"]) == 3
    assert pm["nll"]["numerator"] == pytest.approx(sum(r["nll"] for r in cases))
    assert pm["multiclass_brier"]["value"] == pytest.approx(
        sum(
            sum(
                (r["probabilities"][k] - int(r["true_label"] == label_name)) ** 2
                for k, label_name in enumerate(a.LABELS)
            )
            for r in cases
        )
        / 4
    )
    empty = a.metrics([])
    assert empty["automatic_af_sensitivity"]["value"] is None
    assert empty["macro_f1"] is None and empty["accepted_accuracy"]["value"] is None
    noisy = a.metrics(cases[-1:])
    assert noisy["probability_metrics"]["binary_af_brier"]["value"] is None
    assert noisy["probability_metrics"]["binary_af_excluded_noisy"] == 1
    deferred = a.metrics([dict(cases[1], disposition="confidence_deferred")])
    assert deferred["accepted_af_recall"]["value"] is None
    assert deferred["automatic_af_sensitivity"]["value"] == 0
    failed = a.metrics([dict(cases[1], disposition="technical_failure")])
    assert not failed["complete"] and failed["probability_metrics"] is None
    assert failed["af_outcomes"][-1] == 1


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "label", "class", "probability", "sum", "nan", "missing_logit"],
)
def test_prediction_failures(mutation):
    manifest, rows = synthetic()
    if mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation == "label":
        rows[0]["true_label"] = "N" if rows[0]["true_label"] != "N" else "A"
    elif mutation == "class":
        rows[0]["predicted_label"] = "~"
    elif mutation == "probability":
        rows[0]["probability_N"] += 0.01
        rows[0]["probability_A"] -= 0.01
    elif mutation == "sum":
        rows[0]["probability_N"] += 0.001
    elif mutation == "nan":
        rows[0]["logit_N"] = float("nan")
    else:
        del rows[0]["logit_N"]
    with pytest.raises((ValueError, KeyError)):
        a.validate_predictions(rows, manifest)


def test_float32_tolerance_and_exact_argmax():
    manifest, rows = synthetic()
    a.validate_predictions(rows, manifest)
    rows[0]["probability_N"] += 1e-8
    a.validate_predictions(rows, manifest)
    rows[0]["predicted_label"] = "wrong"
    with pytest.raises(ValueError, match="classes"):
        a.validate_predictions(rows, manifest)


def test_read_only_preflight(repository):
    root, c, path = repository
    before = {str(p): a.sha(p) for p in root.rglob("*") if p.is_file()}
    registry = a.run(root, path, "analysis-preflight")
    assert set(registry["inputs"]) == {"42", "43", "44"}
    assert all(
        r["records"] == 48 and r["required_logits_verified"]
        for r in registry["inputs"].values()
    )
    assert (
        not registry["holdout_signals_accessed"]
        and not registry["role_manifest_generated"]
    )
    assert not (root / c["partition"]["manifest_path"]).exists()
    assert before == {str(p): a.sha(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "which", ["report", "prediction", "checkpoint", "lock", "group_overlap"]
)
def test_provenance_failures(repository, which):
    root, c, path = repository
    report = a.read_json(root / c["inputs"]["report"])
    base = root / "outputs/rhythm/temporal-v1/cuda128/seed43"
    if which == "report":
        (base / "evaluate/evaluation-report.json").write_text("{}")
    elif which == "prediction":
        rows = a.csv_rows(base / "evaluate/validation-predictions.csv")
        rows[0]["record_id"] = "wrong"
        (base / "evaluate/validation-predictions.csv").unlink()
        save_csv(base / "evaluate/validation-predictions.csv", rows)
    elif which == "checkpoint":
        c["inputs"]["models"][1]["checkpoint_sha256"] = "wrong"
    elif which == "lock":
        (root / "uv.lock").write_text("wrong")
    else:
        split = root / c["inputs"]["split_dir"]
        rows = a.csv_rows(split / "train.csv")
        rows[0]["duplicate_group"] = "group-0-0"
        (split / "train.csv").unlink()
        save_csv(split / "train.csv", rows)
        frozen = a.read_json(split / "frozen.json")
        frozen["files_sha256"]["train.csv"] = a.sha(split / "train.csv")
        (split / "frozen.json").write_bytes(a.encoded(frozen))
        report["data_identity"]["frozen_metadata_sha256"] = a.sha(split / "frozen.json")
        (root / c["inputs"]["report"]).write_bytes(a.encoded(report))
    with pytest.raises(ValueError):
        a.preflight(root, c)


@pytest.fixture
def authorized_synthetic(repository, monkeypatch):
    # Authorize only synthetic YAML; bypass Git/environment, not authorization.
    root, c, path = repository
    c["execution_authorized"] = True
    path.write_text(yaml.safe_dump(c))
    monkeypatch.setattr(a, "require_committed_source", lambda *args: None)
    monkeypatch.setattr(a, "environment", lambda root: {"revision": "synthetic"})


def test_synthetic_workflow_gate_ledgers_and_overwrite(
    repository, authorized_synthetic
):
    root, c, path = repository
    fitted = a.run(root, path, "analysis-fit", "synthetic", authorized=True)
    assert fitted["dev_assess_inspected"] is False
    destination = (
        root / "outputs/calibration/temporal-calibration-deferral-v1/synthetic"
    )
    assert not (destination / "assessment").exists()
    frozen = a.read_json(destination / "frozen-policy.json")
    assert set(frozen["policies"]) == {"42", "43", "44"}
    result = a.run(root, path, "analysis-assess", "synthetic", authorized=True)
    assert result["status"] == "exploratory_assessment_complete"
    summary = a.read_json(root / result["summary_path"])
    for seed in ("42", "43", "44"):
        for arm in a.ARMS:
            cases = a.read_json(
                destination / f"assessment/seed{seed}/{arm}-ledger.json"
            )
            actual = summary["seeds"][seed][arm]
            assert actual == a.metrics(cases)
            # Independently count all outcomes and automatic AF numerator.
            outcomes = Counter(
                (
                    r["true_label"],
                    r["predicted_label"]
                    if r["disposition"] == "accepted"
                    else r["disposition"],
                )
                for r in cases
            )
            assert sum(outcomes.values()) == actual["eligible"]
            af = [r for r in cases if r["true_label"] == "A"]
            assert actual["automatic_af_sensitivity"]["denominator"] == len(af)
            assert actual["automatic_af_sensitivity"]["numerator"] == sum(
                r["disposition"] == "accepted" and r["predicted_label"] == "A"
                for r in af
            )
            correct = sum(
                r["disposition"] == "accepted"
                and r["true_label"] == r["predicted_label"]
                for r in cases
            )
            assert actual["accepted_accuracy"]["numerator"] == correct
    for action in ("analysis-fit", "analysis-assess"):
        with pytest.raises(FileExistsError):
            a.run(root, path, action, "synthetic", authorized=True)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_seed",
        "changed_inputs",
        "changed_config",
        "nonconverged",
        "changed_artifact",
        "changed_policy",
    ],
)
def test_assessment_gate_blocks_before_outcomes(
    repository, authorized_synthetic, monkeypatch, damage
):
    root, c, path = repository
    a.run(root, path, "analysis-fit", "synthetic", authorized=True)
    destination = (
        root / "outputs/calibration/temporal-calibration-deferral-v1/synthetic"
    )
    frozen_path = destination / "frozen-policy.json"
    frozen = a.read_json(frozen_path)
    if damage == "missing_seed":
        del frozen["policies"]["43"]
    elif damage == "changed_inputs":
        frozen["input_registry"]["report_sha256"] = "wrong"
    elif damage == "changed_config":
        frozen["config_sha256"] = "wrong"
    elif damage == "nonconverged":
        frozen["policies"]["43"]["fit"]["converged"] = False
    elif damage == "changed_policy":
        frozen["policies"]["43"]["threshold"] = 0.99
    else:
        (destination / "seed43/calibration.json").write_text("{}")
    frozen_path.write_bytes(a.encoded(frozen))
    monkeypatch.setattr(
        a,
        "ledger",
        lambda *args: pytest.fail("Assessment outcome inspected before gate"),
    )
    with pytest.raises(ValueError):
        a.run(root, path, "analysis-assess", "synthetic", authorized=True)
    assert not (destination / "assessment").exists()


def test_failed_fit_has_no_assessment_gate(
    repository, authorized_synthetic, monkeypatch
):
    root, c, path = repository
    monkeypatch.setattr(
        a,
        "fit_temperature",
        lambda *args: (_ for _ in ()).throw(ValueError("synthetic failure")),
    )
    with pytest.raises(ValueError, match="synthetic failure"):
        a.run(root, path, "analysis-fit", "failed", authorized=True)
    assert not (
        root
        / "outputs/calibration/temporal-calibration-deferral-v1/failed/frozen-policy.json"
    ).exists()
    with pytest.raises(FileNotFoundError):
        a.run(root, path, "analysis-assess", "failed", authorized=True)


def test_authorization_and_path_protection(repository):
    root, c, path = repository
    with pytest.raises(ValueError, match="authorize"):
        a.run(root, path, "analysis-fit", "forbidden")
    with pytest.raises(ValueError, match="escapes"):
        a.local(root, "../outside")
    assert not (root / "outputs/calibration").exists()


def test_exclusive_write(tmp_path):
    path = tmp_path / "artifact.json"
    a.write_new(path, {"value": 1})
    with pytest.raises(FileExistsError):
        a.write_new(path, {"value": 2})
    assert a.read_json(path) == {"value": 1}


def test_reliability_edges_and_empty_bins():
    result = a.reliability([(0.0, 0), (1.0, 1), (1 / 15, 1)])
    assert result["bins"][0]["count"] == 1
    assert result["bins"][1]["count"] == 1
    assert result["bins"][-1]["count"] == 1
    assert result["bins"][2]["mean_probability"] is None
    assert result["ece"] == pytest.approx((1 - 1 / 15) / 3)
    assert a.reliability([])["ece"] is None


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("assessment", "binary_af_scope", "all_labels"),
        ("decision", "noisy_prediction_is_deferral", True),
        ("selection", "min_af_coverage", 0.5),
        ("numerical_verification", "probability_atol", 0.1),
    ],
)
def test_schema_rejects_unreviewed_semantics(tmp_path, config, section, key, value):
    config[section][key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        a.load_config(path)


def test_launcher_analysis_routes_without_training_loader(monkeypatch, capsys):
    from test_configuration import manager_module
    import sys

    manager = manager_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manage.py",
            "analysis-fit",
            "--config",
            str(ROOT / "configs/temporal-calibration-deferral-v1.yaml"),
            "--execution-id",
            "future",
            "--authorize-analysis",
            "--dry-run",
        ],
    )
    monkeypatch.setattr(manager, "ensure_yaml", lambda: None)
    manager.main()
    text = capsys.readouterr().out
    assert "-m rhythmlens.calibration analysis-fit" in text
    assert "--execution-id future --authorize-analysis" in text
    assert "train-rhythm" not in text


@pytest.mark.parametrize("action", ["analysis-fit", "analysis-assess"])
@pytest.mark.parametrize(
    "value", [False, None, 0, 1, 1.0, "true", "false", [], {}, "missing"]
)
@pytest.mark.parametrize("cli_authorized", [False, True])
def test_yaml_authorization_blocks_before_analysis(
    repository, monkeypatch, action, value, cli_authorized
):
    root, c, path = repository
    if value == "missing":
        c.pop("execution_authorized")
    else:
        c["execution_authorized"] = value
    path.write_text(yaml.safe_dump(c))
    before = {str(p): a.sha(p) for p in root.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Unauthorized call reached inputs, analysis, Git check or output writing"
        )

    for name in (
        "require_committed_source",
        "preflight",
        "assign_roles",
        "fit_temperature",
        "select_threshold",
        "ledger",
        "write_new",
    ):
        monkeypatch.setattr(a, name, forbidden)
    with pytest.raises(ValueError, match="literal boolean true"):
        a.run(root, path, action, "blocked", authorized=cli_authorized)
    assert before == {str(p): a.sha(p) for p in root.rglob("*") if p.is_file()}
    assert not (root / c["partition"]["manifest_path"]).exists()
    assert not (root / "outputs/calibration").exists()


@pytest.mark.parametrize("action", ["analysis-fit", "analysis-assess"])
def test_yaml_true_still_requires_cli(repository, monkeypatch, action):
    root, c, path = repository
    c["execution_authorized"] = True
    path.write_text(yaml.safe_dump(c))
    monkeypatch.setattr(
        a,
        "preflight",
        lambda *args: pytest.fail("Inputs loaded without CLI authorization"),
    )
    monkeypatch.setattr(
        a,
        "require_committed_source",
        lambda *args: pytest.fail("Git checked before CLI authorization"),
    )
    with pytest.raises(ValueError, match="explicit --authorize-analysis"):
        a.run(root, path, action, "blocked", authorized=False)
    assert not (root / "outputs/calibration").exists()


def test_false_yaml_preflight_bypasses_execution_gate(repository, monkeypatch):
    root, c, path = repository
    assert c["execution_authorized"] is False
    monkeypatch.setattr(
        a,
        "require_authorization",
        lambda *args: pytest.fail("Read-only preflight reached execution gate"),
    )
    result = a.run(root, path, "analysis-preflight")
    assert set(result["inputs"]) == {"42", "43", "44"}
    assert result["role_manifest_generated"] is False
    assert not (root / c["partition"]["manifest_path"]).exists()
    assert not (root / "outputs/calibration").exists()


def test_authorized_analysis_still_requires_clean_committed_source(
    repository, monkeypatch
):
    root, c, path = repository
    c["execution_authorized"] = True
    path.write_text(yaml.safe_dump(c))
    monkeypatch.setattr(
        a.subprocess, "check_output", lambda *args, **kwargs: b" M synthetic-source.py"
    )
    monkeypatch.setattr(
        a, "preflight", lambda *args: pytest.fail("Inputs loaded with dirty source")
    )
    with pytest.raises(ValueError, match="clean tree"):
        a.run(root, path, "analysis-fit", "blocked", authorized=True)
