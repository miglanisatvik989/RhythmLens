"""Versioned physical-signal transform shared by training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import time
from typing import Protocol

import numpy as np

PREPROCESSING_VERSION = "median-fixed-training-rms-v1"


class TrainingRecords(Protocol):
    partition: str
    partition_sha256: str
    rows: list[dict]

    def load_physical(self, index: int) -> np.ndarray: ...


@dataclass(frozen=True)
class PreprocessingState:
    version: str
    scale_mv: float
    training_manifest_sha256: str
    fitted_record_ids_sha256: str
    fitted_records: int
    flat_training_records: int
    sampling_rate_hz: float = 300.0
    units: str = "mV"
    epsilon_mv: float = 1e-8
    scale_definition: str = (
        "median of per-record RMS after per-record median subtraction"
    )
    flat_policy: str = (
        "retain flat signals as zeros after centering; never divide by per-record scale"
    )

    def __post_init__(self) -> None:
        if any(
            type(v) is not int
            for v in (self.fitted_records, self.flat_training_records)
        ):
            raise ValueError("Invalid preprocessing count types")
        if any(
            not isinstance(v, str) or not v
            for v in (self.training_manifest_sha256, self.fitted_record_ids_sha256)
        ):
            raise ValueError("Invalid preprocessing identity")
        if (
            self.scale_definition
            != "median of per-record RMS after per-record median subtraction"
            or self.flat_policy
            != "retain flat signals as zeros after centering; never divide by per-record scale"
        ):
            raise ValueError("Unsupported preprocessing semantics")
        if isinstance(self.scale_mv, bool) or isinstance(self.epsilon_mv, bool):
            raise ValueError("Invalid preprocessing numeric types")
        if self.version != PREPROCESSING_VERSION:
            raise ValueError("Unsupported preprocessing version")
        if not np.isfinite(self.epsilon_mv) or self.epsilon_mv <= 0:
            raise ValueError("Invalid preprocessing epsilon")
        if not np.isfinite(self.scale_mv) or self.scale_mv < self.epsilon_mv:
            raise ValueError("Training amplitude scale is too small or nonfinite")
        if self.sampling_rate_hz != 300 or self.units != "mV":
            raise ValueError("This baseline requires physical mV signals at 300 Hz")
        if (
            self.fitted_records <= 0
            or not 0 <= self.flat_training_records <= self.fitted_records
        ):
            raise ValueError("Invalid preprocessing fitting counts")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> PreprocessingState:
        return cls(**value)


def physical_array(signal: np.ndarray) -> np.ndarray:
    """Validate without mutating the original physical signal."""
    value = np.asarray(signal)
    if value.ndim != 1 or not value.size:
        raise ValueError("Expected one nonempty ECG channel")
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise ValueError("Expected a real numeric ECG signal")
    value = value.astype(np.float64, copy=True)
    if not np.isfinite(value).all():
        raise ValueError("ECG contains nonfinite samples")
    return value


def fit_preprocessing(
    records: TrainingRecords, *, deadline: float | None = None
) -> PreprocessingState:
    """Fit only the declared training partition, retaining every training record."""
    if records.partition != "train":
        raise ValueError("Preprocessing fitting is permitted only on training records")
    if not records.rows:
        raise ValueError("Cannot fit preprocessing on an empty training partition")
    amplitudes, ids, flat = [], [], 0
    for index, row in enumerate(records.rows):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                "Smoke time budget expired during training-only scale fitting"
            )
        signal = physical_array(records.load_physical(index))
        centered = signal - np.median(signal)
        with np.errstate(over="ignore", invalid="ignore"):
            rms = float(np.sqrt(np.mean(centered * centered)))
        if not np.isfinite(rms):
            raise ValueError(f"Nonfinite training RMS for {row['record_id']}")
        amplitudes.append(rms)
        ids.append(row["record_id"])
        flat += int(np.ptp(signal) == 0)
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated fitting record IDs")
    return PreprocessingState(
        version=PREPROCESSING_VERSION,
        scale_mv=float(np.median(amplitudes)),
        training_manifest_sha256=records.partition_sha256,
        fitted_record_ids_sha256=hashlib.sha256(
            json.dumps(sorted(ids), separators=(",", ":")).encode()
        ).hexdigest(),
        fitted_records=len(ids),
        flat_training_records=flat,
    )


def transform_physical(
    signal: np.ndarray,
    state: PreprocessingState,
    *,
    sampling_rate_hz: float = 300.0,
    units: str = "mV",
) -> np.ndarray:
    """Return float32 model input; no denoising, clipping or per-record scaling."""
    if sampling_rate_hz != state.sampling_rate_hz or units != state.units:
        raise ValueError("Sampling rate/units do not match saved preprocessing")
    physical = physical_array(signal)
    with np.errstate(over="ignore", invalid="ignore"):
        processed = ((physical - np.median(physical)) / state.scale_mv).astype(
            np.float32
        )
    if not np.isfinite(processed).all():
        raise ValueError("ECG cannot be represented as finite float32 model input")
    return processed
