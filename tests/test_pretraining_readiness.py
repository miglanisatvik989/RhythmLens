"""Reproducibility regressions using generated signals and metadata only."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rhythmlens import rhythm_training as training
from rhythmlens.compatibility import validate_decoder_contract
from rhythmlens.lifecycle import run_lifecycle
from rhythmlens.rhythm_data import FrozenRecords, LengthBucketSampler
from rhythmlens.rhythm_model import build_model, temporal_capability
from test_rhythm_baseline import frozen_dataset, small_model, state
from rhythmlens.splits import SPLIT_VERSION


@pytest.mark.parametrize("count", [65, 5969])
def test_remainder_uniform_before_sort_and_complete(count):
    sampler = LengthBucketSampler(list(range(1, count + 1)), 16, seed=42)
    remainders = []
    for epoch in range(200):
        sampler.set_epoch(epoch)
        batches = list(sampler)
        assert batches == list(sampler)
        flat = [i for batch in batches for i in batch]
        assert len(flat) == len(set(flat)) == count
        assert sorted(flat) == list(range(count))
        singleton = [batch for batch in batches if len(batch) == 1]
        assert len(singleton) == 1
        remainders.extend(singleton[0])
    assert len(set(remainders)) > 40
    assert 0.35 * count < np.mean(remainders) < 0.65 * count
    assert min(remainders) < count / 4 and max(remainders) > count * 3 / 4


class SyntheticDataset:
    lengths = [21, 39, 72, 101]

    def __len__(self):
        return 4

    def __getitem__(self, i):
        return {
            "record_id": str(i),
            "signal": torch.arange(self.lengths[i], dtype=torch.float32),
            "label": i,
        }


def test_validation_loader_owns_rng():
    config = training.DEFAULT_CONFIG
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    loader = training.make_loader(SyntheticDataset(), config, shuffle=False)
    list(loader)
    assert torch.equal(before, torch.get_rng_state())
    expected = torch.rand(10)
    torch.set_rng_state(before)
    list(loader)
    torch.testing.assert_close(torch.rand(10), expected, rtol=0, atol=0)
    train = training.make_loader(SyntheticDataset(), config, shuffle=True)
    assert train.generator is not loader.generator


def test_lock_edit_supported_without_rewriting_frozen(tmp_path):
    root = frozen_dataset(tmp_path)
    frozen = root / "splits" / SPLIT_VERSION / "frozen.json"
    original = frozen.read_bytes()
    first = FrozenRecords(root, f"splits/{SPLIT_VERSION}", "train")
    (root / "uv.lock").write_text("# benign software lock edit\n")
    second = FrozenRecords(root, f"splits/{SPLIT_VERSION}", "train")
    assert first.identity == second.identity and frozen.read_bytes() == original
    assert (
        second.compatibility["contract"]["contract_version"]
        == "challenge2017-physical-v1"
    )


def test_unsupported_decoder_contract(monkeypatch):
    original = Path.read_text

    def altered(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.name == "data_compatibility.json":
            value = json.loads(text)
            value["contract_version"] = "unknown"
            return json.dumps(value)
        return text

    monkeypatch.setattr(Path, "read_text", altered)
    with pytest.raises(ValueError, match="Unsupported decoder"):
        validate_decoder_contract()


@pytest.mark.parametrize("version", ["rhythm-cnn-v1", "rhythm-dilated-v1"])
@pytest.mark.parametrize("length", [1, 17, 157])
def test_models_roundtrip_padding_backward(tmp_path, version, length):
    torch.set_num_threads(2)
    config = small_model().config.to_dict()
    model = build_model(version, config)
    x = torch.randn(2, 1, length)
    mask = torch.ones(2, length, dtype=torch.bool)
    out = model(x, mask)
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()
    padded = torch.nn.functional.pad(x, (0, 53), value=float("nan"))
    pmask = torch.nn.functional.pad(mask, (0, 53))
    torch.testing.assert_close(
        model(x, mask), model(padded, pmask), atol=2e-6, rtol=2e-6
    )
    path = tmp_path / "model.pt"
    training.save_checkpoint(
        path,
        model,
        torch.optim.AdamW(model.parameters()),
        state(),
        {**training.DEFAULT_CONFIG, "model": config, "model_version": version},
        {"train_sha256": "training-hash"},
        epoch=1,
        steps=1,
        kind="baseline",
        metadata={},
    )
    loaded, _, _ = training.load_checkpoint(path)
    torch.testing.assert_close(model(x, mask), loaded(x, mask), rtol=0, atol=0)
    assert temporal_capability(model)["receptive_field_samples"] == (
        81 if version == "rhythm-cnn-v1" else 3075
    )


def test_timing_sensitivity_is_engineering_evidence():
    torch.manual_seed(61)
    # Equal length/count/amplitude/morphology, separated from boundaries; pulse
    # centers aligned with reference stride. Only inter-pulse spacing changes.
    x = torch.zeros(2, 1, 4096)
    for row, centers in enumerate(([1024, 1536, 2048, 2560], [1024, 1152, 2048, 2560])):
        for center in centers:
            x[row, 0, center - 1 : center + 2] = torch.tensor([0.5, 1.0, 0.5])
    assert torch.equal(x[0].sort().values, x[1].sort().values)
    config = small_model().config.to_dict()
    mask = torch.ones(2, 4096, dtype=torch.bool)
    ref = build_model("rhythm-cnn-v1", config).eval()(x, mask)
    candidate = build_model("rhythm-dilated-v1", config).eval()(x, mask)
    torch.testing.assert_close(ref[0], ref[1], atol=1e-6, rtol=1e-6)
    assert (candidate[0] - candidate[1]).abs().max() > 1e-6


@pytest.fixture
def payload_path(tmp_path):
    model = small_model()
    path = tmp_path / "valid.pt"
    training.save_checkpoint(
        path,
        model,
        torch.optim.AdamW(model.parameters()),
        state(),
        {**training.DEFAULT_CONFIG, "model": model.config.to_dict()},
        {"train_sha256": "training-hash"},
        epoch=1,
        steps=2,
        kind="baseline",
        metadata={},
    )
    return path


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 99),
        ("schema_version", True),
        ("model_version", "unknown"),
        ("task", "af"),
        ("label_order", ["A", "N", "O", "~"]),
        ("epoch", "one"),
        ("config", []),
        ("metadata", None),
        ("preprocessing", {}),
        ("data_identity", {}),
    ],
)
def test_bad_checkpoint_metadata(payload_path, field, value):
    payload = torch.load(payload_path, weights_only=True)
    payload[field] = value
    torch.save(payload, payload_path)
    with pytest.raises(ValueError):
        training.load_checkpoint(payload_path)


@pytest.mark.parametrize("field", ["model_config", "preprocessing", "data_identity"])
def test_checkpoint_inconsistency(payload_path, field):
    payload = torch.load(payload_path, weights_only=True)
    if field == "model_config":
        payload[field]["dropout"] = 0.4
    elif field == "preprocessing":
        payload[field]["scale_mv"] = 9.0
    else:
        payload[field]["train_sha256"] = "corrupt"
    torch.save(payload, payload_path)
    with pytest.raises(ValueError):
        training.load_checkpoint(payload_path)


@pytest.mark.parametrize(
    "exception,status",
    [(RuntimeError("failure"), "failed"), (KeyboardInterrupt(), "interrupted")],
)
def test_lifecycle_exception(tmp_path, exception, status):
    with pytest.raises(type(exception)):
        with run_lifecycle(tmp_path, training.DEFAULT_CONFIG):
            raise exception
    value = json.loads((tmp_path / "status.json").read_text())
    assert (
        value["state"] == status
        and value["details"]["type"] == type(exception).__name__
    )


@pytest.mark.parametrize(
    "exception", [RuntimeError("simulated failure"), KeyboardInterrupt()]
)
def test_resume_matches_uninterrupted_and_repeats_partial_epoch(
    tmp_path, monkeypatch, exception
):
    root = frozen_dataset(tmp_path)
    config = {
        **training.DEFAULT_CONFIG,
        "split_dir": f"splits/{SPLIT_VERSION}",
        "model": small_model().config.to_dict(),
        "epochs": 3,
        "batch_size": 4,
        "noncanonical_development": True,
    }
    # Include dropout in the equality check.
    config["model"]["dropout"] = 0.2
    training.run_training(root, config, "outputs/continuous", smoke=False)
    train_batch = training.train_batch
    count = 0

    def interrupted(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 8:  # Six batches in first epoch, then one partial-epoch step.
            raise exception
        return train_batch(*args, **kwargs)

    monkeypatch.setattr(training, "train_batch", interrupted)
    with pytest.raises(type(exception)):
        training.run_training(root, config, "outputs/resumed", smoke=False)
    destination = root / "outputs/resumed"
    status = json.loads((destination / "status.json").read_text())
    assert status["last_completed_epoch"] == 1
    checkpoint = torch.load(destination / "last.pt", weights_only=True)
    assert checkpoint["epoch"] == 1
    original = (destination / "last.pt").read_bytes()
    with pytest.raises(ValueError, match="Incompatible resume"):
        training.run_training(
            root, {**config, "seed": 99}, "outputs/resumed", smoke=False, resume=True
        )
    assert (destination / "last.pt").read_bytes() == original
    monkeypatch.setattr(training, "train_batch", train_batch)
    training.run_training(root, config, "outputs/resumed", smoke=False, resume=True)
    for name in ("last.pt", "best.pt"):
        a = torch.load(root / "outputs/continuous" / name, weights_only=True)
        b = torch.load(destination / name, weights_only=True)
        assert a["epoch"] == b["epoch"] and a["steps"] == b["steps"]
        assert a["continuation"] == b["continuation"]
        for key in a["model_state"]:
            torch.testing.assert_close(
                a["model_state"][key], b["model_state"][key], rtol=0, atol=0
            )
    assert json.loads(
        (root / "outputs/continuous/history.json").read_text()
    ) == json.loads((destination / "history.json").read_text())
    assert json.loads((destination / "status.json").read_text())["state"] == "completed"
    with pytest.raises(ValueError, match="Completed"):
        training.run_training(root, config, "outputs/resumed", smoke=False, resume=True)


def test_cuda_labelled_checkpoint_evaluates_on_cpu(tmp_path):
    root = frozen_dataset(tmp_path)
    config = {
        **training.DEFAULT_CONFIG,
        "split_dir": f"splits/{SPLIT_VERSION}",
        "model": small_model().config.to_dict(),
        "epochs": 1,
        "noncanonical_development": True,
    }
    training.run_training(root, config, "outputs/source", smoke=False)
    path = root / "outputs/source/best.pt"
    payload = torch.load(path, weights_only=True)
    payload["config"]["device"] = "cuda"
    payload["config"]["gpu_index"] = 7
    payload["metadata"]["environment"]["device"] = "cuda:7"
    torch.save(payload, path)
    report = training.evaluate_command(root, path, "outputs/eval")
    assert report["environment"]["device"] == "cpu"
    assert report["training_config"]["device"] == "cuda"
    assert report["training_provenance"]["environment"]["device"] == "cuda:7"


def test_canonical_gate_before_output_or_signal_access(tmp_path, monkeypatch):
    from rhythmlens.preflight import preflight

    monkeypatch.setattr(
        FrozenRecords, "load_physical", lambda *a: pytest.fail("decoded a signal")
    )
    with pytest.raises(ValueError, match="clean committed source"):
        preflight(tmp_path, training.DEFAULT_CONFIG, output="outputs/canonical")
    assert not (tmp_path / "outputs").exists()


def test_preflight_missing_raw_and_bad_reuse(tmp_path):
    from rhythmlens.preflight import preflight

    root = frozen_dataset(tmp_path)
    config = {
        **training.DEFAULT_CONFIG,
        "split_dir": f"splits/{SPLIT_VERSION}",
        "noncanonical_development": True,
    }
    records = FrozenRecords(root, config["split_dir"], "train")
    path = root / records.rows[0]["mat_path"]
    path.unlink()
    with pytest.raises(FileNotFoundError, match="Missing local ECG"):
        preflight(root, config, output="outputs/check", smoke=True)
    assert not (root / "outputs/check").exists()


def test_atomic_checkpoint_failure_preserves_last(payload_path, monkeypatch):
    original = payload_path.read_bytes()

    def broken(*args, **kwargs):
        raise OSError("simulated serialization failure")

    monkeypatch.setattr(torch, "save", broken)
    model = small_model()
    with pytest.raises(OSError):
        training.save_checkpoint(
            payload_path,
            model,
            torch.optim.AdamW(model.parameters()),
            state(),
            {**training.DEFAULT_CONFIG, "model": model.config.to_dict()},
            {"train_sha256": "training-hash"},
            epoch=2,
            steps=3,
            kind="baseline",
            metadata={},
        )
    assert payload_path.read_bytes() == original
    training.load_checkpoint(payload_path)


def test_evaluation_launcher_routes_current_profile_and_config():
    from test_configuration import manager_module
    from rhythmlens.configuration import read_document

    root = Path(__file__).resolve().parents[1]
    document = read_document(root / "config.yaml")
    document["environment"]["profile"] = "cuda128"
    document["evaluation"]["profile"] = "cpu"
    command = manager_module().commands_for(
        "evaluate", root, root / "config.yaml", document
    )[0]
    assert command[0] == str(root / ".venv/bin/python")
    assert "--config" in command
    assert "/cuda128/" in command[command.index("--checkpoint") + 1]


def test_resume_flag_is_an_action_not_a_learning_change():
    from rhythmlens.lifecycle import resume_signature
    from rhythmlens.configuration import load_experiment

    root = Path(__file__).resolve().parents[1]
    config, _ = load_experiment(root / "configs/reference.yaml", root)
    import copy

    resumed = copy.deepcopy(config)
    resumed["_experiment"]["document"]["execution"]["resume"] = True
    resumed["_experiment"]["source_yaml"] = "changed resume flag"
    resumed["_experiment"]["source_sha256"] = "new YAML hash"
    assert resume_signature(resumed) == resume_signature(config)
    resumed["learning_rate"] *= 2
    assert resume_signature(resumed) != resume_signature(config)


def test_legacy_reference_checkpoint_without_new_optional_fields(payload_path):
    payload = torch.load(payload_path, weights_only=True)
    for key in (
        "data_identity_sha256",
        "preprocessing_identity_sha256",
        "temporal_capability",
        "continuation",
    ):
        payload.pop(key, None)
    payload["config"].pop("model_version")
    payload["config"].pop("noncanonical_development")
    payload["rng"].pop("numpy")
    torch.save(payload, payload_path)
    model, _, _ = training.load_checkpoint(payload_path)
    assert model.model_version == "rhythm-cnn-v1"


@pytest.mark.parametrize("mode", ["label", "membership", "duplicate", "accounting"])
def test_split_semantics_fail_even_if_file_hashes_refreshed(tmp_path, mode):
    import csv
    from rhythmlens.data import sha256

    root = frozen_dataset(tmp_path)
    split = root / "splits" / SPLIT_VERSION
    target = split / "train.csv"
    with target.open() as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if mode == "label":
        rows[0]["label"] = "A" if rows[0]["label"] != "A" else "N"
    elif mode == "membership":
        rows[0]["split"] = "validation"
    elif mode == "duplicate":
        rows[0]["duplicate_group"] = rows[1]["duplicate_group"]
    else:
        rows.pop(0)
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    frozen = json.loads((split / "frozen.json").read_text())
    frozen["files_sha256"]["train.csv"] = sha256(target)
    (split / "frozen.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError):
        FrozenRecords(root, f"splits/{SPLIT_VERSION}", "train")


def test_active_yaml_is_separate_from_source_bundle(tmp_path):
    from rhythmlens.configuration import load_experiment, read_document
    import yaml

    root = Path(__file__).resolve().parents[1]
    document = read_document(root / "configs/reference.yaml")
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs/run.yaml"
    path.write_text(yaml.safe_dump(document))
    first, _ = load_experiment(path, tmp_path)
    before = training.code_metadata(tmp_path, first)
    document["execution"]["resume"] = True
    path.write_text(yaml.safe_dump(document))
    second, _ = load_experiment(path, tmp_path)
    assert (
        before["source_bundle_sha256"]
        == training.code_metadata(tmp_path, second)["source_bundle_sha256"]
    )
    assert (
        first["_experiment"]["source_sha256"] != second["_experiment"]["source_sha256"]
    )
