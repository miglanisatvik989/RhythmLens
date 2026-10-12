"""Workflow routing and repeatable integrity checks; generated data only."""

import copy
import csv
import importlib.util
import sys

import pytest
import torch
import yaml

from rhythmlens.data import MANIFESTS
from rhythmlens.rhythm_data import FrozenRecords
from rhythmlens.splits import SPLIT_VERSION
from test_configuration import ROOT, manager_module, save_config
from test_rhythm_baseline import frozen_dataset


def doctor_module():
    spec = importlib.util.spec_from_file_location(
        "workflow_doctor", ROOT / "scripts/doctor_environment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("training", ["cpu", "cuda128"])
@pytest.mark.parametrize("evaluation", ["cpu", "cuda128"])
def test_evaluation_setup_uses_own_lock_and_preserves_document(training, evaluation):
    manager = manager_module()
    document = yaml.safe_load((ROOT / "config.yaml").read_text())
    document["environment"]["profile"] = training
    document["evaluation"]["profile"] = evaluation
    original = copy.deepcopy(document)
    commands = manager.commands_for(
        "setup-evaluation", ROOT, ROOT / "config.yaml", document
    )
    expected = copy.deepcopy(document)
    expected["environment"]["profile"] = evaluation
    assert commands == manager.commands_for(
        "setup", ROOT, ROOT / "config.yaml", expected
    )
    assert document == original
    assert "--locked" in commands[0]
    if evaluation == "cuda128":
        assert "--no-deps" in commands[1] and "--no-sources" in commands[1]
        assert "--no-build-isolation" in commands[1]
    else:
        assert "--all-extras" in commands[0]


def configure_manager(manager, root, monkeypatch):
    config_module = manager.configuration_module()
    monkeypatch.setattr(manager, "ROOT", root)
    monkeypatch.setattr(manager, "configuration_module", lambda: config_module)
    monkeypatch.setattr(manager, "ensure_yaml", lambda: None)
    (root / ".python-version").write_text("3.11.16\n")


def test_evaluation_setup_dry_run_needs_no_gpu_or_environment(
    tmp_path, monkeypatch, capsys
):
    manager = manager_module()
    document = yaml.safe_load((ROOT / "config.yaml").read_text())
    save_config(tmp_path, document)
    configure_manager(manager, tmp_path, monkeypatch)
    monkeypatch.setattr(manager.platform, "system", lambda: "Windows")
    monkeypatch.setattr(manager.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(sys, "argv", ["manage.py", "setup-evaluation", "--dry-run"])
    monkeypatch.setattr(
        manager.subprocess, "run", lambda *a, **k: pytest.fail("executed dry run")
    )
    manager.main()
    output = capsys.readouterr().out
    assert "--project ." in output and "--all-extras" in output
    assert "--project environments/cuda128" not in output
    assert not (tmp_path / ".venv").exists()


def test_missing_evaluator_points_to_its_setup(tmp_path, monkeypatch, capsys):
    manager = manager_module()
    save_config(tmp_path, yaml.safe_load((ROOT / "config.yaml").read_text()))
    configure_manager(manager, tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["manage.py", "evaluate"])
    with pytest.raises(SystemExit) as error:
        manager.main()
    assert error.value.code == 1
    message = capsys.readouterr().err
    assert "setup-evaluation" in message and str(tmp_path / "config.yaml") in message


def test_verify_route_does_not_request_smoke_preflight():
    manager = manager_module()
    document = yaml.safe_load((ROOT / "config.yaml").read_text())
    command = manager.commands_for("verify-data", ROOT, ROOT / "config.yaml", document)[
        0
    ]
    assert "--integrity-only" in command and "--smoke" not in command


def test_integrity_repeatable_with_outputs_and_without_holdout_signals(
    tmp_path, monkeypatch
):
    root = frozen_dataset(tmp_path)
    split_dir = f"splits/{SPLIT_VERSION}"
    with (root / MANIFESTS / "records.csv").open() as stream:
        manifest = {r["record_id"]: r for r in csv.DictReader(stream)}
    with (root / split_dir / "holdout.csv").open() as stream:
        holdout = list(csv.DictReader(stream))
    for row in holdout:
        for field in ("mat_path", "header_path"):
            (root / manifest[row["record_id"]][field]).unlink()
    for action in ("smoke", "train", "evaluate"):
        path = root / "outputs/rhythm/baseline-v1/cuda128/seed42" / action
        path.mkdir(parents=True)
        (path / "preserved.txt").write_text("completed output\n")
    doctor = doctor_module()
    monkeypatch.setattr(
        doctor, "preflight", lambda *a, **k: pytest.fail("called preflight")
    )
    monkeypatch.setattr(
        FrozenRecords, "load_physical", lambda *a: pytest.fail("decoded ECG")
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("queried GPU"))
    monkeypatch.setattr(
        torch, "randn", lambda *a, **k: pytest.fail("constructed tensor")
    )
    before = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    rng = torch.get_rng_state().clone()
    config = {"split_dir": split_dir}
    first = doctor.verify_data(root, config)
    assert first == doctor.verify_data(root, config)
    assert first["raw_files_checked"] == 2 * sum(first["partition_records"].values())
    assert first["ECG_signals_decoded"] == 0
    assert not first["holdout_signals_accessed"]
    assert not first["synthetic_execution"] and not first["output_created"]
    assert torch.equal(rng, torch.get_rng_state())
    assert before == {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("partition", ["train", "validation"])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_integrity_rejects_damaged_development_inputs(tmp_path, partition, damage):
    root = frozen_dataset(tmp_path)
    split_dir = f"splits/{SPLIT_VERSION}"
    records = FrozenRecords(root, split_dir, partition)
    path = root / records.rows[0]["mat_path"]
    if damage == "missing":
        path.unlink()
        expected = FileNotFoundError
    else:
        path.write_bytes(path.read_bytes() + b"damaged")
        expected = ValueError
    with pytest.raises(expected, match="Missing local ECG|Audited ECG input changed"):
        doctor_module().verify_data(root, {"split_dir": split_dir})


def test_doctor_still_uses_full_preflight(tmp_path, monkeypatch, capsys):
    doctor = doctor_module()
    document = yaml.safe_load((ROOT / "config.yaml").read_text())
    path = save_config(tmp_path, document)
    monkeypatch.setattr(
        doctor, "__file__", str(tmp_path / "scripts/doctor_environment.py")
    )
    called = {}

    def preflight(root, config, **kwargs):
        called.update(kwargs)
        assert root == tmp_path
        assert config["device"] == "cuda" and config["model"]["dropout"] == 0.1
        return {"normal_doctor": True}

    monkeypatch.setattr(doctor, "preflight", preflight)
    monkeypatch.setattr(
        doctor, "verify_data", lambda *a: pytest.fail("integrity-only route")
    )
    monkeypatch.setattr(sys, "argv", ["doctor_environment.py", "--config", str(path)])
    doctor.main()
    assert called["synthetic_execution"] is True and called["smoke"] is False
    assert called["output"] == "outputs/rhythm/baseline-v1/cuda128/seed42/train"
    assert '"normal_doctor": true' in capsys.readouterr().out


@pytest.mark.parametrize("fail", [False, True])
def test_integrity_cli_uses_only_integrity_route(tmp_path, monkeypatch, capsys, fail):
    doctor = doctor_module()
    path = save_config(tmp_path, yaml.safe_load((ROOT / "config.yaml").read_text()))
    monkeypatch.setattr(
        doctor, "__file__", str(tmp_path / "scripts/doctor_environment.py")
    )
    monkeypatch.setattr(
        doctor, "preflight", lambda *a, **k: pytest.fail("called GPU preflight")
    )

    def verify(root, config):
        assert root == tmp_path and config["model"]["dropout"] == 0.1
        if fail:
            raise ValueError("Audited ECG input changed: synthetic.mat")
        return {"synthetic_execution": False}

    monkeypatch.setattr(doctor, "verify_data", verify)
    monkeypatch.setattr(
        sys,
        "argv",
        ["doctor_environment.py", "--config", str(path), "--integrity-only"],
    )
    if fail:
        with pytest.raises(SystemExit) as error:
            doctor.main()
        assert error.value.code == 1
        assert "Audited ECG input changed" in capsys.readouterr().err
    else:
        doctor.main()
        assert '"synthetic_execution": false' in capsys.readouterr().out
