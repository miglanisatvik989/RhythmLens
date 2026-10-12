"""Synthetic-only checks for leakage, padding invariance and checkpoint parity.

Signals are generated in pytest tmp_path. No patient signals are embedded.
"""

from dataclasses import replace
import json
import sys

import numpy as np
import pytest
import torch

from rhythmlens.audit import audit
from rhythmlens.data import LABELS, MANIFESTS, RAW
from rhythmlens.preprocessing import (
    PREPROCESSING_VERSION,
    PreprocessingState,
    fit_preprocessing,
    transform_physical,
)
from rhythmlens.rhythm_data import (
    FrozenRecords,
    LengthBucketSampler,
    RhythmDataset,
    collate_records,
)
from rhythmlens.rhythm_inference import predict_physical
from rhythmlens.rhythm_model import ModelConfig, RhythmCNN, masked_average
from rhythmlens.rhythm_training import (
    DEFAULT_CONFIG,
    classification_metrics,
    create_output,
    evaluate_command,
    load_checkpoint,
    load_config,
    run_training,
    save_checkpoint,
)
from rhythmlens.splits import SPLIT_VERSION, freeze
from test_challenge2017 import dataset


@pytest.fixture(autouse=True)
def cpu_test_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(2017)
    yield
    torch.set_num_threads(previous)


def small_model():
    return RhythmCNN(ModelConfig(stem_channels=8, channels=(8, 16, 24), dropout=0.0))


def state(scale=2.0, digest="training-hash"):
    return PreprocessingState(
        version=PREPROCESSING_VERSION,
        scale_mv=scale,
        training_manifest_sha256=digest,
        fitted_record_ids_sha256="ids-hash",
        fitted_records=2,
        flat_training_records=0,
    )


class FakeRecords:
    def __init__(self, signals, partition="train"):
        self.partition = partition
        self.partition_sha256 = "training-hash"
        self.rows = [{"record_id": f"synthetic-{i}"} for i in range(len(signals))]
        self.signals, self.loaded = signals, []

    def load_physical(self, index):
        self.loaded.append(index)
        return self.signals[index]


def frozen_dataset(tmp_path):
    root = dataset(tmp_path, count=8)
    assert audit(root, previews=False)["validation_passed"]
    freeze(root)
    return root


def test_fit_uses_only_training_and_known_statistic():
    records = FakeRecords([np.array([-1.0, 0, 1]), np.array([-3.0, 0, 3])])
    fitted = fit_preprocessing(records)
    assert fitted.scale_mv == pytest.approx(2 * np.sqrt(2 / 3))
    assert fitted.training_manifest_sha256 == "training-hash"
    assert records.loaded == [0, 1]
    other = FakeRecords([np.array([-1000.0, 0, 1000])], partition="validation")
    with pytest.raises(ValueError, match="only on training"):
        fit_preprocessing(other)
    assert other.loaded == []


def test_flat_signals_are_retained_without_per_record_division():
    records = FakeRecords(
        [np.array([9.0, 9, 9]), np.array([-2.0, 0, 2]), np.array([-3.0, 0, 3])]
    )
    fitted = fit_preprocessing(records)
    assert fitted.fitted_records == 3 and fitted.flat_training_records == 1
    np.testing.assert_array_equal(
        transform_physical(records.signals[0], fitted), np.zeros(3, dtype=np.float32)
    )
    with pytest.raises(ValueError, match="too small"):
        fit_preprocessing(FakeRecords([np.zeros(10)]))


def test_shared_transform_preserves_original_and_relative_amplitudes():
    original = np.array([10.0, 12, 14])
    before = original.copy()
    first = transform_physical(original, state())
    second = transform_physical(2 * original, state())
    np.testing.assert_array_equal(original, before)
    np.testing.assert_allclose(first, [-1, 0, 1])
    np.testing.assert_allclose(second, 2 * first)
    assert first.dtype == np.float32
    restored = PreprocessingState.from_dict(state().to_dict())
    assert restored == state()


def test_expired_fit_budget_stops_before_loading_any_signal():
    records = FakeRecords([np.arange(3.0)])
    with pytest.raises(TimeoutError, match="time budget"):
        fit_preprocessing(records, deadline=0.0)
    assert records.loaded == []


@pytest.mark.parametrize(
    "signal",
    [
        np.array([]),
        np.ones((1, 4)),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([1 + 2j]),
        np.array(["x"]),
    ],
)
def test_invalid_physical_inputs(signal):
    with pytest.raises(ValueError):
        transform_physical(signal, state())


@pytest.mark.parametrize("arguments", [{"sampling_rate_hz": 250}, {"units": "uV"}])
def test_sampling_rate_and_units_must_match_saved_transform(arguments):
    with pytest.raises(ValueError, match="Sampling rate/units"):
        transform_physical(np.array([1.0, 2]), state(), **arguments)


def test_float32_overflow_is_not_silently_clipped():
    with pytest.raises(ValueError, match="float32"):
        transform_physical(np.array([-1e100, 0.0, 1e100]), state())


def test_repeated_fitting_ids_and_unknown_versions_rejected():
    records = FakeRecords([np.arange(3.0), np.arange(3.0)])
    records.rows[1]["record_id"] = records.rows[0]["record_id"]
    with pytest.raises(ValueError, match="Repeated"):
        fit_preprocessing(records)
    with pytest.raises(ValueError, match="version"):
        replace(state(), version="unsupported")


@pytest.mark.parametrize("length", [1, 2, 3, 4, 7, 16, 17, 101])
def test_logits_are_invariant_to_padding_and_batch_companions(length):
    model = small_model().eval()
    # Make padded activations adversarial: affine and convolution biases are nonzero.
    with torch.no_grad():
        for module in model.modules():
            if (
                isinstance(module, (torch.nn.Conv1d, torch.nn.LayerNorm))
                and module.bias is not None
            ):
                module.bias.fill_(0.6)
    original = torch.randn(1, 1, length)
    mask = torch.ones(1, length, dtype=torch.bool)
    padded = torch.full((1, 1, length + 39), float("nan"))
    padded[:, :, :length] = original
    padded_mask = torch.arange(length + 39)[None, :] < length
    companion = torch.randn_like(padded)
    together = torch.cat([padded, companion])
    together_mask = torch.cat([padded_mask, torch.ones_like(padded_mask)])
    with torch.inference_mode():
        reference = model(original, mask)
        torch.testing.assert_close(
            reference, model(padded, padded_mask), atol=2e-6, rtol=2e-6
        )
        torch.testing.assert_close(
            reference, model(together, together_mask)[:1], atol=2e-6, rtol=2e-6
        )
        features, down = model.features(padded, padded_mask)
    assert int(down.sum()) == (length + 3) // 4
    assert (
        torch.count_nonzero(
            features.masked_select(~down[:, None, :].expand_as(features))
        )
        == 0
    )


def test_padding_gradients_are_zero_and_valid_gradients_finite():
    model = small_model()
    x = torch.randn(2, 1, 71, requires_grad=True)
    lengths = torch.tensor([29, 71])
    mask = torch.arange(71)[None, :] < lengths[:, None]
    torch.nn.functional.cross_entropy(model(x, mask), torch.tensor([0, 3])).backward()
    assert torch.count_nonzero(x.grad[0, :, 29:]) == 0
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad[0, :, :29]) > 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_masked_pooling_excludes_arbitrary_invalid_features():
    x = torch.tensor([[[1.0, 3, float("nan"), 1e20]]])
    mask = torch.tensor([[True, True, False, False]])
    torch.testing.assert_close(masked_average(x, mask), torch.tensor([[2.0]]))
    with pytest.raises(ValueError, match="no valid"):
        masked_average(x, torch.zeros_like(mask))


@pytest.mark.parametrize(
    "mask",
    [
        torch.zeros(1, 4, dtype=torch.bool),
        torch.tensor([[True, False, True, False]]),
        torch.ones(1, 4),
        torch.ones(1, 3, dtype=torch.bool),
    ],
)
def test_invalid_masks_are_rejected(mask):
    with pytest.raises(ValueError):
        small_model()(torch.randn(1, 1, 4), mask)


def test_nonfinite_valid_samples_are_rejected():
    with pytest.raises(ValueError, match="Nonfinite valid"):
        small_model()(
            torch.full((1, 1, 4), float("nan")), torch.ones(1, 4, dtype=torch.bool)
        )


def test_collation_and_label_order():
    items = [
        {
            "record_id": f"synthetic-{i}",
            "signal": torch.arange(i + 2, dtype=torch.float32),
            "label": i,
        }
        for i in range(4)
    ]
    batch = collate_records(items)
    assert LABELS == ("N", "A", "O", "~")
    assert batch["labels"].tolist() == [0, 1, 2, 3]
    assert batch["lengths"].tolist() == [2, 3, 4, 5]
    assert batch["mask"].sum(1).tolist() == [2, 3, 4, 5]
    assert batch["signals"].shape == (4, 1, 5)
    assert (
        torch.count_nonzero(batch["signals"].masked_select(~batch["mask"][:, None, :]))
        == 0
    )
    with pytest.raises(ValueError, match="Repeated"):
        collate_records([items[0], items[0]])


def test_bucket_order_is_reproducible_and_covers_every_record():
    sampler = LengthBucketSampler(list(range(1, 38)), 4, seed=42)
    first = list(sampler)
    assert first == list(sampler)
    assert sorted(i for batch in first for i in batch) == list(range(37))
    assert len(first) == len(sampler) == 10
    orderings = []
    for epoch in range(4):
        sampler.set_epoch(epoch)
        orderings.append(list(sampler))
    assert any(order != orderings[0] for order in orderings[1:])


def test_frozen_loader_fit_and_inference_parity(tmp_path):
    root = frozen_dataset(tmp_path)
    records = FrozenRecords(root, f"splits/{SPLIT_VERSION}", "train")
    fitted = fit_preprocessing(records)
    data = RhythmDataset(records, fitted)
    item = data[0]
    physical = records.load_physical(0)
    model = small_model().eval()
    batch = collate_records([item])
    with torch.inference_mode():
        from_batch = model(batch["signals"], batch["mask"])[0].numpy()
    from_inference = predict_physical(
        model, fitted, physical, sampling_rate_hz=300, units="mV"
    )
    np.testing.assert_allclose(from_batch, from_inference, rtol=1e-6, atol=1e-6)
    assert item["label"] == LABELS.index(records.rows[0]["label"])
    with pytest.raises(ValueError, match="another training"):
        RhythmDataset(records, replace(fitted, training_manifest_sha256="wrong"))


def test_holdout_locked_and_snapshot_tampering_detected(tmp_path):
    root = frozen_dataset(tmp_path)
    with pytest.raises(ValueError, match="holdout is locked"):
        FrozenRecords(root, f"splits/{SPLIT_VERSION}", "holdout")
    partition = root / "splits" / SPLIT_VERSION / "train.csv"
    partition.write_text(partition.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen split file changed"):
        FrozenRecords(root, f"splits/{SPLIT_VERSION}", "train")


def test_changed_raw_record_is_blocked_before_decoding(tmp_path):
    root = frozen_dataset(tmp_path)
    records = FrozenRecords(root, f"splits/{SPLIT_VERSION}", "train")
    path = root / records.rows[0]["header_path"]
    path.write_text(path.read_text() + "# changed\n")
    with pytest.raises(ValueError, match="Audited ECG input changed"):
        records.load_physical(0)


def test_source_snapshot_tampering_is_detected(tmp_path):
    root = frozen_dataset(tmp_path)
    path = root / MANIFESTS / "records.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="source snapshot"):
        FrozenRecords(root, f"splits/{SPLIT_VERSION}", "train")


def test_checkpoint_reload_and_identity_checks(tmp_path):
    model = small_model().eval()
    optimizer = torch.optim.AdamW(model.parameters())
    identity = {"train_sha256": "training-hash"}
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        state(),
        {**DEFAULT_CONFIG, "model": model.config.to_dict()},
        identity,
        epoch=1,
        steps=1,
        kind="baseline",
        metadata={},
    )
    loaded, preprocessing, payload = load_checkpoint(
        path, expected_data_identity=identity
    )
    signal = torch.randn(2, 1, 61)
    mask = torch.ones(2, 61, dtype=torch.bool)
    with torch.inference_mode():
        torch.testing.assert_close(
            model(signal, mask), loaded(signal, mask), rtol=0, atol=0
        )
    assert preprocessing == state() and payload["steps"] == 1
    with pytest.raises(ValueError, match="identity mismatch"):
        load_checkpoint(path, expected_data_identity={"train_sha256": "different"})
    payload["label_order"] = ["A", "N", "O", "~"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="label order"):
        load_checkpoint(path)


def test_metrics_use_all_af_labels_and_all_four_classes():
    result = classification_metrics([1, 1, 1, 0, 2, 3], [1, 0, 3, 1, 1, 1])
    assert result["af_denominator"] == 3 and result["af_correct"] == 1
    assert result["af_recall_all_labeled_af"] == pytest.approx(1 / 3)
    assert result["per_class"]["~"]["support"] == 1
    assert result["confusion_matrix"][3][1] == 1
    assert classification_metrics([0], [0])["af_recall_all_labeled_af"] is None


@pytest.mark.parametrize("output", ["reports/new-run", "../outside", "outputs"])
def test_output_directory_guards(tmp_path, output):
    with pytest.raises(ValueError):
        create_output(tmp_path, output)


def test_no_output_overwrite(tmp_path):
    path = create_output(tmp_path, "outputs/smoke")
    (path / "marker").write_text("keep")
    with pytest.raises(ValueError, match="refusing overwrite"):
        create_output(tmp_path, "outputs/smoke")
    assert (path / "marker").read_text() == "keep"


@pytest.mark.parametrize(
    "key,value",
    [
        ("smoke_max_steps", 201),
        ("smoke_max_seconds", 1201),
        ("batch_size", 0),
        ("learning_rate", float("nan")),
        ("seed", -1),
        ("unknown", 1),
    ],
)
def test_invalid_configuration(tmp_path, key, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({key: value}))
    with pytest.raises(ValueError):
        load_config(path)


def test_smoke_never_decodes_validation_or_holdout(tmp_path, monkeypatch):
    root = frozen_dataset(tmp_path)
    decoder = FrozenRecords.load_physical
    accessed = []

    def guarded(records, index):
        assert records.partition == "train", "smoke decoded a nontraining signal"
        accessed.append(records.rows[index]["record_id"])
        return decoder(records, index)

    monkeypatch.setattr(FrozenRecords, "load_physical", guarded)
    config = {
        **DEFAULT_CONFIG,
        "noncanonical_development": True,
        "split_dir": f"splits/{SPLIT_VERSION}",
        "model": small_model().config.to_dict(),
        "smoke_max_steps": 4,
    }
    before = (root / "splits" / SPLIT_VERSION / "frozen.json").read_bytes()
    report = run_training(root, config, "outputs/smoke", smoke=True)
    assert report["steps"] == 4
    assert report["training_records"] == 16
    assert (
        not report["holdout_signals_accessed"]
        and not report["validation_signals_accessed"]
    )
    assert report["preprocessing"]["fitted_records"] > 16
    assert report["checkpoint_reload_verified"] and accessed
    assert (root / "splits" / SPLIT_VERSION / "frozen.json").read_bytes() == before
    with pytest.raises(ValueError, match="Smoke checkpoints"):
        evaluate_command(root, root / "outputs/smoke/best.pt", "outputs/not-validation")
    assert not (root / "outputs/not-validation").exists()


def test_baseline_validation_command_and_checkpoint_selection(tmp_path, monkeypatch):
    root = frozen_dataset(tmp_path)
    decoder = FrozenRecords.load_physical

    def guarded(records, index):
        assert records.partition in ("train", "validation")
        return decoder(records, index)

    monkeypatch.setattr(FrozenRecords, "load_physical", guarded)
    config = {
        **DEFAULT_CONFIG,
        "noncanonical_development": True,
        "split_dir": f"splits/{SPLIT_VERSION}",
        "model": small_model().config.to_dict(),
        "batch_size": 4,
        "epochs": 2,
    }
    report = run_training(root, config, "outputs/baseline", smoke=False)
    repeated = evaluate_command(
        root, root / "outputs/baseline/best.pt", "outputs/validation"
    )
    assert report["best_selection_metrics"] == repeated["metrics"]
    assert (
        report["validation_signals_accessed"] and not report["holdout_signals_accessed"]
    )
    assert (root / "outputs/validation/validation-confusion.png").is_file()
    assert (root / "outputs/validation/validation-predictions.csv").is_file()


def test_cli_rejects_holdout_option():
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rhythmlens",
            "evaluate-rhythm",
            "--checkpoint",
            "x.pt",
            "--output",
            "outputs/x",
            "--partition",
            "holdout",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "unrecognized arguments" in result.stderr


def test_missing_local_ecg_is_explained(tmp_path):
    root = frozen_dataset(tmp_path)
    records = FrozenRecords(root, f"splits/{SPLIT_VERSION}", "train")
    (root / records.rows[0]["mat_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="Missing local ECG input"):
        records.load_physical(0)
    assert (root / RAW).exists()
