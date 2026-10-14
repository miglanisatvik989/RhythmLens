"""Unitless NSTDB pools and deterministic plans; no experiment runner or ECG I/O."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import re

import numpy as np
import scipy
from scipy.signal import resample_poly

from .preprocessing import physical_array

ROLES = {
    "augmentation-train": (0, 600),
    "development-reserve": (660, 1140),
    "stress": (1200, 1800),
}
NOISE_RECORDS = ("bw", "ma", "em")
STRESS_SNR = (24, 18, 12, 6, 0, -6)
TRAIN_SNR = (24, 18, 12, 6)
RESAMPLING = {
    "up": 5,
    "down": 6,
    "window": ("kaiser", 5.0),
    "padtype": "constant",
    "cval": 0,
}


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class NoisePool:
    record: str
    role: str
    source_sha256: str
    values: np.ndarray
    pool_sha256: str

    def metadata(self) -> dict:
        start, stop = ROLES[self.role]
        return {
            "record": self.record,
            "role": self.role,
            "source_sha256": self.source_sha256,
            "pool_sha256": self.pool_sha256,
            "hash_encoding": "little_endian_float64_C_order_samples_by_channels",
            "raw_bounds_samples_half_open": [start * 360, stop * 360],
            "source_sampling_rate_hz": 360,
            "target_sampling_rate_hz": 300,
            "resampling": RESAMPLING,
            "scipy_version": scipy.__version__,
            "trim_output_samples_each_end": 300,
            "samples_per_channel": len(self.values),
            "channels": [0, 1],
            "semantics": "unitless_digital_shape_no_voltage_calibration",
        }


def prepare_pool(
    digital: np.ndarray, *, record: str, role: str, source_sha256: str
) -> NoisePool:
    if scipy.__version__ != "1.15.3":
        raise ValueError("Frozen noise resampling requires SciPy 1.15.3")
    if record not in NOISE_RECORDS or role not in ROLES:
        raise ValueError("Unsupported noise record/role")
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("Required original digital-file SHA256")
    value = np.asarray(digital)
    if (
        value.shape != (650000, 2)
        or not np.issubdtype(value.dtype, np.integer)
        or np.any(value <= -2048)
        or np.any(value > 2047)
    ):
        raise ValueError("Invalid digital inventory or missing-value sentinel")
    start, stop = ROLES[role]
    # ADC zero is strictly validated as zero by the separate NSTDB decoder.
    # Slice first: even polyphase filter support cannot read outside this role.
    source = value[start * 360 : stop * 360].astype(np.float64)
    result = resample_poly(source, axis=0, **RESAMPLING)[300:-300].copy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite resampled noise: technical failure")
    result.flags.writeable = False
    return NoisePool(record, role, source_sha256, result, array_hash(result))


def _rng(key: list) -> tuple[np.random.Generator, int]:
    encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")
    return np.random.Generator(np.random.PCG64(seed)), seed


def _length(length: int, pool: NoisePool) -> None:
    if type(length) is not int or not 0 < length <= len(pool.values):
        raise ValueError("Invalid length; never wrap/concatenate/redraw")


def _record_id(record_id: str) -> None:
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("Nonempty recording identity required")


@dataclass(frozen=True)
class NoisePlan:
    key: tuple
    seed: int
    role: str
    length: int
    noise_record: str | None
    channel: int | None
    offset: int | None
    pool_sha256: str | None
    source_sha256: str | None
    snr_db: int | None = None

    def metadata(self) -> dict:
        return asdict(self)


def stress_plan(
    record_id: str,
    noise_record: str,
    repeat: int,
    length: int,
    pools: dict[str, NoisePool],
    *,
    partition: str,
) -> NoisePlan:
    if partition != "validation":
        raise ValueError("Stress plans are validation-only; holdout is excluded")
    _record_id(record_id)
    if (
        noise_record not in NOISE_RECORDS
        or type(repeat) is not int
        or repeat not in (0, 1, 2)
    ):
        raise ValueError("Invalid stress noise/repetition")
    pool = pools[noise_record]
    if pool.role != "stress" or pool.record != noise_record:
        raise ValueError("Stress role mismatch")
    _length(length, pool)
    key = [20261015, "stress", record_id, noise_record, repeat]
    rng, seed = _rng(key)
    channel = int(rng.integers(0, 2))
    offset = int(rng.integers(0, len(pool.values) - length + 1))
    return NoisePlan(
        tuple(key),
        seed,
        pool.role,
        length,
        noise_record,
        channel,
        offset,
        pool.pool_sha256,
        pool.source_sha256,
    )


def training_plan(
    record_id: str,
    model_seed: int,
    epoch: int,
    length: int,
    pools: dict[str, NoisePool],
    *,
    partition: str,
) -> NoisePlan:
    if partition != "train":
        raise ValueError("Training plans are training-only; holdout is excluded")
    _record_id(record_id)
    if (
        type(model_seed) is not int
        or model_seed not in (42, 43, 44)
        or type(epoch) is not int
        or epoch < 1
    ):
        raise ValueError("Invalid training seed/epoch")
    if set(pools) != set(NOISE_RECORDS):
        raise ValueError("All three noise records required; no substitution")
    for record, pool in pools.items():
        if pool.role != "augmentation-train" or pool.record != record:
            raise ValueError("Training role mismatch")
        _length(length, pool)
    key = [20261014, "train", model_seed, epoch, record_id]
    rng, seed = _rng(key)
    if rng.random() >= 0.5:
        return NoisePlan(
            tuple(key), seed, "augmentation-train", length, None, None, None, None, None
        )
    record = NOISE_RECORDS[int(rng.integers(0, 3))]
    channel = int(rng.integers(0, 2))
    snr = TRAIN_SNR[int(rng.integers(0, len(TRAIN_SNR)))]
    pool = pools[record]
    offset = int(rng.integers(0, len(pool.values) - length + 1))
    return NoisePlan(
        tuple(key),
        seed,
        pool.role,
        length,
        record,
        channel,
        offset,
        pool.pool_sha256,
        pool.source_sha256,
        snr,
    )


def corrupt_physical(
    signal_mv: np.ndarray, noise_shape: np.ndarray, snr_db: float
) -> tuple[np.ndarray, dict]:
    """Pure array operation before the unchanged saved ECG transform, never a loader."""
    x = physical_array(signal_mv)
    v = physical_array(noise_shape)
    if (
        x.shape != v.shape
        or isinstance(snr_db, (bool, np.bool_))
        or snr_db not in STRESS_SNR
    ):
        raise ValueError("Mismatched lengths or unsupported requested SNR")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        v -= np.mean(v)
        noise_rms = float(np.sqrt(np.mean(v * v)))
        centered = x - np.median(x)
        reference_rms = float(np.sqrt(np.mean(centered * centered)))
    if not np.isfinite(noise_rms) or noise_rms <= 0:
        raise ValueError("Zero/nonfinite noise RMS; technical failure, never redraw")
    if not np.isfinite(reference_rms):
        raise ValueError("Nonfinite reference RMS; technical failure")
    metadata = {
        "requested_snr_db": float(snr_db),
        "reference_rms_mv": reference_rms,
        "selected_centered_noise_rms_unitless": noise_rms,
        "achieved_snr_db": None,
        "added_rms_mv": 0.0,
        "added_peak_mv": 0.0,
        "corruption_applied": False,
        "reason": "flat_or_near_flat_reference",
    }
    if reference_rms < 1e-8:
        return x, metadata
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        added = (v / noise_rms) * reference_rms * 10 ** (-float(snr_db) / 20)
        output = x + added
        added_rms = float(np.sqrt(np.mean(added * added)))
        achieved = float(20 * np.log10(reference_rms / added_rms))
    if (
        not np.isfinite(output).all()
        or not np.isfinite(added_rms)
        or not np.isfinite(achieved)
        or abs(achieved - snr_db) > 1e-6
    ):
        raise ValueError("Invalid corruption/achieved SNR; technical failure")
    metadata.update(
        achieved_snr_db=achieved,
        added_rms_mv=added_rms,
        added_peak_mv=float(np.max(np.abs(added))),
        corruption_applied=True,
        reason=None,
    )
    return output, metadata


def apply_plan(
    signal_mv: np.ndarray,
    plan: NoisePlan,
    pools: dict[str, NoisePool],
    *,
    partition: str,
    snr_db: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Verify the plan against its exact deterministic reconstruction; no redraw."""
    x = physical_array(signal_mv)
    if len(x) != plan.length:
        raise ValueError("ECG length differs from frozen plan")
    if plan.role == "stress" and len(plan.key) == 5:
        expected = stress_plan(
            plan.key[2],
            plan.key[3],
            plan.key[4],
            plan.length,
            pools,
            partition=partition,
        )
        if snr_db not in STRESS_SNR or isinstance(snr_db, bool):
            raise ValueError("Stress application requires explicit SNR")
    elif plan.role == "augmentation-train" and len(plan.key) == 5:
        expected = training_plan(
            plan.key[4],
            plan.key[2],
            plan.key[3],
            plan.length,
            pools,
            partition=partition,
        )
        if snr_db is not None:
            raise ValueError("Training SNR comes from the frozen plan")
        snr_db = plan.snr_db
    else:
        raise ValueError("Unsupported role/plan key")
    if plan != expected:
        raise ValueError("Altered deterministic plan")
    if plan.noise_record is None:
        return x, {
            "corruption_applied": False,
            "reason": "bernoulli_clean",
            "requested_snr_db": None,
            "achieved_snr_db": None,
            "added_rms_mv": 0.0,
            "added_peak_mv": 0.0,
            "plan": plan.metadata(),
        }
    pool = pools[plan.noise_record]
    if array_hash(pool.values) != plan.pool_sha256:
        raise ValueError("Pool changed after plan creation")
    noise = pool.values[plan.offset : plan.offset + plan.length, plan.channel]
    output, metadata = corrupt_physical(x, noise, snr_db)
    metadata["plan"] = plan.metadata()
    metadata["role"] = pool.metadata()
    metadata["original_physical_sha256"] = array_hash(x)
    metadata["corrupted_physical_sha256"] = array_hash(output)
    return output, metadata
