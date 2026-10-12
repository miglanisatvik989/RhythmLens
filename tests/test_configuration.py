"""Configuration and environment routing checks; generated signals only."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
import yaml

from rhythmlens.audit import audit
from rhythmlens.configuration import (
    output_path,
    read_document,
    relative_path,
    validate_experiment,
)
from rhythmlens.rhythm_training import DEFAULT_CONFIG, load_config, setup_runtime
from rhythmlens.splits import SPLIT_VERSION, freeze
from test_challenge2017 import dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def document():
    return read_document(ROOT / "config.yaml")


def save_config(root, document):
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


def manager_module():
    spec = importlib.util.spec_from_file_location("manager", ROOT / "scripts/manage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_yaml_drives_runtime_model_optimizer_and_smoke(tmp_path, document, monkeypatch):
    document["training"]["batch_size"] = 7
    document["optimizer"]["learning_rate"] = 0.0007
    document["runtime"]["gpu_index"] = 2
    document["model"]["dropout"] = 0.3
    document["smoke"]["dropout"] = 0.0
    path = save_config(tmp_path, document)
    monkeypatch.chdir(tmp_path)
    training = load_config(path)
    smoke = load_config(path, smoke=True)
    assert training["batch_size"] == 7 and training["learning_rate"] == 0.0007
    assert training["gpu_index"] == 2 and training["model"]["dropout"] == 0.3
    assert smoke["model"]["dropout"] == 0.0
    assert (
        training["_experiment"]["source_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert read_document(path) == document


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("runtime", "gpu_index", -1),
        ("runtime", "cpu_threads", True),
        ("runtime", "device", "cdua"),
        ("environment", "profile", "cuda999"),
        ("optimizer", "name", "SGD"),
        ("optimizer", "learning_rate", float("nan")),
        ("model", "kernel", 4),
        ("model", "channels", [32, 64]),
        ("model", "dropout", 1),
        ("training", "batch_size", 0),
        ("training", "cache_signals", "true"),
        ("training", "selection_metric", "holdout_accuracy"),
        ("smoke", "max_steps", 201),
        ("smoke", "max_seconds", 1201),
        ("preprocessing", "method", "per-record-standardization"),
        ("preprocessing", "reuse", "/tmp/preprocessing.json"),
        ("data", "split_dir", "../holdout"),
        ("paths", "output_root", "data/manifests"),
        ("experiment", "name", "../../overwrite"),
    ],
)
def test_bad_choices_fail_before_install_or_execution(
    tmp_path, document, section, key, value
):
    document[section][key] = value
    with pytest.raises(ValueError):
        validate_experiment(document, tmp_path)


def test_unknown_missing_and_duplicate_keys_rejected(tmp_path, document):
    invalid = copy.deepcopy(document)
    invalid["runtime"]["gpu_indx"] = 1
    with pytest.raises(ValueError, match="runtime"):
        validate_experiment(invalid, tmp_path)
    invalid = copy.deepcopy(document)
    del invalid["runtime"]["gpu_index"]
    with pytest.raises(ValueError, match="runtime"):
        validate_experiment(invalid, tmp_path)
    path = tmp_path / "duplicate.yaml"
    path.write_text("runtime:\n  device: cpu\n  device: cuda\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_document(path)


def test_cpu_profile_rejects_explicit_cuda(tmp_path, document):
    document["environment"]["profile"] = "cpu"
    with pytest.raises(ValueError, match="requires"):
        validate_experiment(document, tmp_path)
    document["runtime"]["device"] = "cpu"
    validate_experiment(document, tmp_path)


def test_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "outputs").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="escapes"):
        relative_path(root, "outputs/run")


def test_setup_and_run_use_selected_lock_and_interpreter(tmp_path, document):
    manager = manager_module()
    path = save_config(tmp_path, document)
    gpu = manager.commands_for("setup", tmp_path, path, document)
    assert gpu[0][gpu[0].index("--project") + 1] == "environments/cuda128"
    assert "--locked" in gpu[0] and "--no-build-isolation" in gpu[1]
    assert "--no-deps" in gpu[1] and "--no-sources" in gpu[1]
    train = manager.commands_for("train", tmp_path, path, document)[0]
    assert "environments/cuda128/.venv" in train[0].replace("\\", "/")
    assert train[-1] == output_path(document, "train")
    document["environment"]["profile"] = "cpu"
    document["runtime"]["device"] = "cpu"
    cpu = manager.commands_for("setup", tmp_path, path, document)
    assert len(cpu) == 2 and "--all-extras" in cpu[0]
    assert cpu[0][cpu[0].index("--project") + 1] == "."
    assert output_path(document, "smoke") != output_path(document, "train")


def test_dry_run_does_not_install_train_or_make_outputs(
    tmp_path, document, monkeypatch, capsys
):
    manager = manager_module()
    config_module = manager.configuration_module()
    save_config(tmp_path, document)
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    monkeypatch.setattr(manager, "configuration_module", lambda: config_module)
    monkeypatch.setattr(manager, "ensure_yaml", lambda: None)
    monkeypatch.setattr(sys, "argv", ["manage.py", "train", "--dry-run"])
    monkeypatch.setattr(
        manager.subprocess, "run", lambda *a, **kw: pytest.fail("executed a dry run")
    )
    manager.main()
    assert "train-rhythm" in capsys.readouterr().out
    assert not (tmp_path / "outputs").exists()


def test_gpu_index_selects_requested_device(monkeypatch):
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    assert (
        str(setup_runtime({**DEFAULT_CONFIG, "device": "cuda", "gpu_index": 1}))
        == "cuda:1"
    )
    assert selected == [1]
    with pytest.raises(ValueError, match="unavailable"):
        setup_runtime({**DEFAULT_CONFIG, "device": "cuda", "gpu_index": 2})


def test_wrong_torch_build_cannot_claim_selected_profile():
    wrong = "cpu" if "+cu128" in str(torch.__version__) else "cuda128"
    with pytest.raises(ValueError, match="Wrong environment"):
        setup_runtime({**DEFAULT_CONFIG, "_experiment": {"profile": wrong}})


def test_legacy_json_and_cli_help_still_work(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"batch_size": 8}))
    assert load_config(path)["batch_size"] == 8
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/manage.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "doctor" in result.stdout


@pytest.mark.parametrize("reuse", [False, True])
def test_yaml_cli_smoke_retains_holdout_and_records_provenance(
    tmp_path, document, reuse, monkeypatch
):
    root = dataset(tmp_path, count=8)
    shutil.copy2(ROOT / "uv.lock", root / "uv.lock")
    shutil.copy2(ROOT / "pyproject.toml", root / "pyproject.toml")
    profile = "cuda128" if "+cu128" in str(torch.__version__) else "cpu"
    if profile == "cuda128":
        (root / "environments/cuda128").mkdir(parents=True)
        shutil.copy2(
            ROOT / "environments/cuda128/uv.lock", root / "environments/cuda128/uv.lock"
        )
    assert audit(root, previews=False)["validation_passed"]
    freeze(root)
    holdout = root / "splits" / SPLIT_VERSION / "holdout.csv"
    before = holdout.read_bytes()
    document["environment"]["profile"] = profile
    document["runtime"]["device"] = "cpu"
    document["data"]["split_dir"] = f"splits/{SPLIT_VERSION}"
    document["model"].update(stem_channels=8, channels=[8, 16, 24])
    document["smoke"]["max_steps"] = 2
    path = save_config(root, document)
    if reuse:
        from rhythmlens import rhythm_training

        monkeypatch.chdir(root)
        fitted = rhythm_training.fit_command(
            root, load_config(path), "outputs/saved-scale"
        )
        document["preprocessing"]["reuse"] = "outputs/saved-scale/preprocessing.json"
        path = save_config(root, document)
        monkeypatch.setattr(
            rhythm_training,
            "fit_preprocessing",
            lambda *a, **kw: pytest.fail("refitted saved preprocessing"),
        )
        report = rhythm_training.run_training(
            root, load_config(path, smoke=True), "outputs/yaml-smoke", smoke=True
        )
        assert report["preprocessing"] == fitted["preprocessing"]
    else:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rhythmlens",
                "smoke-rhythm",
                "--config",
                str(path),
                "--output",
                "outputs/yaml-smoke",
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    report = json.loads((root / "outputs/yaml-smoke/run-report.json").read_text())
    assert report["training_records"] == 4 and report["checkpoint_reload_verified"]
    assert (
        not report["validation_signals_accessed"]
        and not report["holdout_signals_accessed"]
    )
    assert report["config"]["model"]["dropout"] == 0.0
    assert (
        report["metadata"]["reproduction"]["dependency_locks_sha256"]["uv.lock"]
        == hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    )
    assert (
        root / "outputs/yaml-smoke/experiment.yaml"
    ).read_bytes() == path.read_bytes()
    assert holdout.read_bytes() == before
