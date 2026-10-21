"""Pure synthetic adapter checks, including global RNG isolation and physical SNR."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
from scipy.signal import resample_poly

from rhythmlens.noise import (
    ROLES,
    RESAMPLING,
    NOISE_RECORDS,
    STRESS_SNR,
    TRAIN_SNR,
    prepare_pool,
    stress_plan,
    training_plan,
    apply_plan,
    corrupt_physical,
)
from rhythmlens.preprocessing import (
    PreprocessingState,
    PREPROCESSING_VERSION,
    transform_physical,
)


@pytest.fixture(scope="module")
def digital():
    t = np.arange(650000, dtype=np.int32)
    return np.column_stack((t % 201 - 100, (t * 7) % 301 - 150)).astype(np.int16)


@pytest.fixture(scope="module")
def pools(digital):
    return {
        role: {
            record: prepare_pool(
                digital, record=record, role=role, source_sha256="a" * 64
            )
            for record in NOISE_RECORDS
        }
        for role in ROLES
    }


@pytest.mark.parametrize("role", list(ROLES))
def test_slice_before_filter_and_trim_isolation(digital, pools, role):
    start, stop = (x * 360 for x in ROLES[role])
    alternate = digital.copy()
    alternate[:start] = 1777
    alternate[stop:] = -1777
    modified = prepare_pool(alternate, record="bw", role=role, source_sha256="b" * 64)
    original = pools[role]["bw"]
    np.testing.assert_array_equal(modified.values, original.values)
    expected = resample_poly(
        digital[start:stop].astype(np.float64), axis=0, **RESAMPLING
    )[300:-300]
    np.testing.assert_array_equal(original.values, expected)
    assert len(original.values) == (stop - start) * 5 // 6 - 600
    assert original.metadata()["raw_bounds_samples_half_open"] == [start, stop]
    assert not original.values.flags.writeable


def independent_rng(key):
    seed = int.from_bytes(
        hashlib.sha256(
            json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode()
        ).digest()[:8],
        "little",
    )
    return np.random.Generator(np.random.PCG64(seed)), seed


def test_stress_exact_key_draws_and_global_rng_isolation(pools):
    import torch

    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    stress = pools["stress"]
    key = [20261015, "stress", "synthetic-é", "ma", 2]
    expected_rng, expected_seed = independent_rng(key)
    channel = int(expected_rng.integers(0, 2))
    offset = int(expected_rng.integers(0, len(stress["ma"].values) - 1301 + 1))
    plan = stress_plan("synthetic-é", "ma", 2, 1301, stress, partition="validation")
    assert plan.seed == expected_seed
    assert (plan.channel, plan.offset) == (channel, offset)
    assert plan.key == tuple(key)
    assert plan == stress_plan(
        "synthetic-é", "ma", 2, 1301, stress, partition="validation"
    )
    x = np.sin(np.arange(1301) / 17) * 0.2
    shapes = []
    for snr in STRESS_SNR:
        result, meta = apply_plan(x, plan, stress, partition="validation", snr_db=snr)
        shapes.append((result - x) / meta["added_rms_mv"])
        assert meta["plan"]["seed"] == expected_seed
    for shape in shapes[1:]:
        np.testing.assert_allclose(shape, shapes[0], atol=1e-12)
    after = np.random.get_state()
    assert numpy_state[0] == after[0] and numpy_state[2:] == after[2:]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    assert torch.equal(torch_state, torch.random.get_rng_state())


def test_training_exact_draws_both_branches_and_rng_isolation(pools):
    import torch

    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    training = pools["augmentation-train"]
    seen = set()
    for index in range(30):
        record_id = f"synthetic-{index}"
        key = [20261014, "train", 43, 2, record_id]
        rng, seed = independent_rng(key)
        clean = rng.random() >= 0.5
        plan = training_plan(record_id, 43, 2, 777, training, partition="train")
        assert plan.seed == seed and plan.key == tuple(key)
        seen.add(clean)
        if clean:
            assert plan.noise_record is None
        else:
            record = NOISE_RECORDS[int(rng.integers(0, 3))]
            channel = int(rng.integers(0, 2))
            snr = TRAIN_SNR[int(rng.integers(0, 4))]
            offset = int(rng.integers(0, len(training[record].values) - 777 + 1))
            assert (plan.noise_record, plan.channel, plan.snr_db, plan.offset) == (
                record,
                channel,
                snr,
                offset,
            )
        x = np.sin(np.arange(777) / 13)
        output, metadata = apply_plan(x, plan, training, partition="train")
        assert output.shape == x.shape
        assert metadata["corruption_applied"] == (not clean)
        if clean:
            np.testing.assert_array_equal(output, x)
    assert seen == {False, True}
    np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
    assert numpy_state[2:] == np.random.get_state()[2:]
    assert torch.equal(torch_state, torch.random.get_rng_state())


@pytest.mark.parametrize("snr", STRESS_SNR)
def test_requested_achieved_snr_unitless_semantics_and_preprocessing(snr):
    x = 0.7 + 0.15 * np.sin(np.arange(3001) / 31)
    v = np.cos(np.arange(3001) / 7) + 50
    before_x, before_v = x.copy(), v.copy()
    result, metadata = corrupt_physical(x, v, snr)
    altered, _ = corrupt_physical(x, v * 17 + 123, snr)
    np.testing.assert_allclose(result, altered, atol=1e-13)
    reference_rms = np.sqrt(np.mean((x - np.median(x)) ** 2))
    added = result - x
    measured = 20 * np.log10(reference_rms / np.sqrt(np.mean(added**2)))
    assert abs(measured - snr) <= 1e-6
    assert abs(metadata["achieved_snr_db"] - snr) <= 1e-6
    assert abs(np.mean(added)) <= 1e-14
    assert metadata["added_peak_mv"] == pytest.approx(np.max(np.abs(added)))
    state = PreprocessingState(
        PREPROCESSING_VERSION,
        0.18008390945457742,
        "synthetic-manifest",
        "synthetic-ids",
        3,
        0,
    )
    processed = transform_physical(result, state)
    assert processed.shape == x.shape and processed.dtype == np.float32
    np.testing.assert_array_equal(
        processed, ((result - np.median(result)) / state.scale_mv).astype(np.float32)
    )
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(v, before_v)


@pytest.mark.parametrize("amplitude", [0, 1e-10])
def test_flat_retains_case_zero_added_and_null_snr(amplitude):
    x = 5 + amplitude * np.sin(np.arange(501))
    output, metadata = corrupt_physical(x, np.sin(np.arange(501) / 7), 6)
    np.testing.assert_array_equal(output, x)
    assert metadata["reason"] == "flat_or_near_flat_reference"
    assert metadata["corruption_applied"] is False
    assert metadata["achieved_snr_db"] is None
    assert metadata["requested_snr_db"] == 6 and metadata["added_rms_mv"] == 0


@pytest.mark.parametrize(
    "case",
    [
        "nonfinite_ecg",
        "nonfinite_noise",
        "zero_noise",
        "flat_zero_noise",
        "overflow",
        "empty",
        "length",
        "bad_snr",
    ],
)
def test_technical_failures_never_redraw(case):
    x, v, snr = np.arange(101, dtype=float), np.sin(np.arange(101)), 6
    if case == "nonfinite_ecg":
        x[10] = np.nan
    elif case == "nonfinite_noise":
        v[10] = np.inf
    elif case == "zero_noise":
        v[:] = 4
    elif case == "flat_zero_noise":
        x[:], v[:] = 0, 0
    elif case == "overflow":
        x *= 1e305
    elif case == "empty":
        x, v = x[:0], v[:0]
    elif case == "length":
        v = v[:-1]
    else:
        snr = np.nan
    with pytest.raises(ValueError):
        corrupt_physical(x, v, snr)


def test_plan_roles_bounds_tamper_and_holdout_exclusion(pools):
    stress, training = pools["stress"], pools["augmentation-train"]
    length = len(stress["bw"].values)
    plan = stress_plan("synthetic", "bw", 0, length, stress, partition="validation")
    assert plan.offset == 0
    for invalid in (0, length + 1, True):
        with pytest.raises(ValueError):
            stress_plan("synthetic", "bw", 0, invalid, stress, partition="validation")
    for partition in ("holdout", "train"):
        with pytest.raises(ValueError):
            stress_plan("synthetic", "bw", 0, 100, stress, partition=partition)
    with pytest.raises(ValueError):
        training_plan("synthetic", 42, 1, 100, training, partition="holdout")
    with pytest.raises(ValueError):
        stress_plan("synthetic", "bw", 0, 100, training, partition="validation")
    with pytest.raises(ValueError):
        training_plan("synthetic", 42, 1, 100, stress, partition="train")
    plan = stress_plan("synthetic", "bw", 0, 100, stress, partition="validation")
    with pytest.raises(ValueError, match="Altered"):
        apply_plan(
            np.arange(100),
            replace(plan, channel=1 - plan.channel),
            stress,
            partition="validation",
            snr_db=6,
        )
    with pytest.raises(ValueError, match="holdout"):
        apply_plan(np.arange(100), plan, stress, partition="holdout", snr_db=6)
    with pytest.raises(ValueError, match="length"):
        apply_plan(np.arange(101), plan, stress, partition="validation", snr_db=6)
    with pytest.raises(ValueError):
        prepare_pool(
            np.ones((10, 2), dtype=int),
            record="bw",
            role="stress",
            source_sha256="a" * 64,
        )


def test_pool_tamper_and_wrong_resampler_version_fail(pools, digital, monkeypatch):
    import scipy

    stress = dict(pools["stress"])
    plan = stress_plan("synthetic", "bw", 0, 100, stress, partition="validation")
    changed = stress["bw"].values.copy()
    changed[plan.offset, plan.channel] += 1
    stress["bw"] = replace(stress["bw"], values=changed)
    with pytest.raises(ValueError, match="Pool changed"):
        apply_plan(np.arange(100), plan, stress, partition="validation", snr_db=6)
    monkeypatch.setattr(scipy, "__version__", "unsupported")
    with pytest.raises(ValueError, match="SciPy 1.15.3"):
        prepare_pool(digital, record="bw", role="stress", source_sha256="a" * 64)
