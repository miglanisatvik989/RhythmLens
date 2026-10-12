"""Read frozen partitions without refreezing them or touching holdout signals."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from rhythmlens.data import LABELS, MANIFESTS, sha256
from rhythmlens.preprocessing import PreprocessingState, transform_physical


def project_path(root: Path, relative: str | Path) -> Path:
    value = Path(relative)
    path = (root / value).resolve()
    if value.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Expected a path inside the project: {relative}")
    return path


class FrozenRecords:
    """Integrity-checked metadata; signals are decoded only on explicit access.

    Freeze-time implementation hashes remain historical provenance. The verified
    decoder modules must match the active reviewed compatibility contract;
    software locks are validated and recorded separately by runtime preflight.
    """

    def __init__(self, root: Path, split_dir: str | Path, partition: str):
        if partition not in ("train", "validation"):
            raise ValueError(
                "Baseline development permits train/validation only; holdout is locked"
            )
        self.root = root.resolve()
        self.partition = partition
        self.split_dir = project_path(self.root, split_dir)
        frozen_path = self.split_dir / "frozen.json"
        self.frozen = json.loads(frozen_path.read_text())
        for name, digest in self.frozen["files_sha256"].items():
            path = project_path(self.split_dir, name)
            if sha256(path) != digest:
                raise ValueError(f"Frozen split file changed: {name}")
        for name, key in (
            ("records.csv", "source_manifest_sha256"),
            ("audit.json", "source_audit_sha256"),
            ("provenance.json", "source_provenance_sha256"),
        ):
            if sha256(self.root / MANIFESTS / name) != self.frozen[key]:
                raise ValueError(f"Frozen source snapshot changed: {name}")
        from rhythmlens.compatibility import validate_decoder_contract

        self.compatibility = validate_decoder_contract()
        config = json.loads((self.split_dir / "config.json").read_text())
        if config["label_order"] != list(LABELS):
            raise ValueError("Frozen label order differs from N,A,O,~")
        self.audit = json.loads((self.root / MANIFESTS / "audit.json").read_text())
        if not self.audit["validation_passed"] or self.audit["errors"]:
            raise ValueError("Cannot train from a failed data audit")
        with (self.root / MANIFESTS / "records.csv").open(newline="") as stream:
            manifest_rows = list(csv.DictReader(stream))
        manifest = {row["record_id"]: row for row in manifest_rows}
        if len(manifest) != len(manifest_rows):
            raise ValueError("Repeated source-manifest IDs")
        assignments, groups, selected, subjects = {}, {}, [], {}
        for split in ("train", "validation", "holdout"):
            with (self.split_dir / f"{split}.csv").open(newline="") as stream:
                partition_rows = list(csv.DictReader(stream))
            if {row["label"] for row in partition_rows} != set(LABELS):
                raise ValueError(f"Class coverage failed for {split}")
            for row in partition_rows:
                record_id = row["record_id"]
                if record_id in assignments or row["split"] != split:
                    raise ValueError("Repeated record or wrong partition tag")
                source = manifest.get(record_id)
                if source is None or source["valid"] != "True":
                    raise ValueError(f"Invalid source record: {record_id}")
                if any(
                    source[key] != row[key]
                    for key in ("label", "subject_id", "duplicate_group")
                ):
                    raise ValueError(
                        f"Frozen/source label or grouping mismatch: {record_id}"
                    )
                subject = row["subject_id"]
                if (
                    subject not in ("", "unknown")
                    and subject in subjects
                    and subjects[subject] != split
                ):
                    raise ValueError("Verified subject-group leakage")
                if subject not in ("", "unknown"):
                    subjects[subject] = split
                group = row["duplicate_group"]
                if group in groups and groups[group] != split:
                    raise ValueError("Duplicate-group leakage")
                assignments[record_id], groups[group] = split, split
                if split == partition:
                    selected.append({**source, "split": split})
        if set(assignments) != set(manifest):
            raise ValueError("Frozen record accounting failed")
        assignment_digest = hashlib.sha256(
            json.dumps(sorted(assignments.items()), separators=(",", ":")).encode()
        ).hexdigest()
        if assignment_digest != self.frozen["assignment_sha256"]:
            raise ValueError("Frozen assignment hash mismatch")
        self.rows = sorted(selected, key=lambda row: row["record_id"])
        self.partition_sha256 = sha256(self.split_dir / f"{partition}.csv")
        self.identity = {
            "split_version": self.frozen["split_version"],
            "assignment_sha256": assignment_digest,
            "frozen_metadata_sha256": sha256(frozen_path),
            "source_manifest_sha256": self.frozen["source_manifest_sha256"],
            "source_audit_sha256": self.frozen["source_audit_sha256"],
            "source_provenance_sha256": self.frozen["source_provenance_sha256"],
            "train_sha256": self.frozen["files_sha256"]["train.csv"],
            "validation_sha256": self.frozen["files_sha256"]["validation.csv"],
            "dependency_lock_sha256": self.frozen.get("dependency_lock_sha256"),
        }

    def load_physical(self, index: int) -> np.ndarray:
        from rhythmlens.audit import decode_record

        row = self.rows[index]
        paths = []
        for key in ("mat_path", "header_path"):
            name = row[key]
            path = project_path(self.root, name)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing local ECG input: {name}; source archives contain no raw ECG"
                )
            if sha256(path) != self.audit["inputs_sha256"].get(name):
                raise ValueError(f"Audited ECG input changed: {name}")
            paths.append(path)
        signal, metadata = decode_record(*paths)
        if (
            metadata["samples"] != int(row["samples"])
            or metadata["duplicate_group"] != row["duplicate_group"]
            or metadata["sampling_rate_hz"] != 300
            or metadata["physical_units"] != "mV"
        ):
            raise ValueError(
                f"Decoded ECG differs from audited metadata: {row['record_id']}"
            )
        return signal


class RhythmDataset(Dataset):
    def __init__(
        self,
        records: FrozenRecords,
        state: PreprocessingState,
        indices: list[int] | None = None,
        *,
        cache: bool = True,
    ):
        if state.training_manifest_sha256 != records.identity["train_sha256"]:
            raise ValueError("Preprocessing was fitted on another training manifest")
        self.records, self.state = records, state
        self.indices = (
            list(range(len(records.rows))) if indices is None else list(indices)
        )
        if not self.indices or len(self.indices) != len(set(self.indices)):
            raise ValueError("Dataset needs unique, nonempty record indices")
        if any(index < 0 or index >= len(records.rows) for index in self.indices):
            raise ValueError("Record index outside the selected partition")
        self.lengths = [int(records.rows[index]["samples"]) for index in self.indices]
        self.cache = cache
        self._signals: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        record_index = self.indices[index]
        row = self.records.rows[record_index]
        processed = self._signals.get(record_index)
        if processed is None:
            physical = self.records.load_physical(record_index)
            processed = transform_physical(
                physical,
                self.state,
                sampling_rate_hz=float(row["sampling_rate_hz"]),
                units=row["physical_units"],
            )
            if self.cache:
                self._signals[record_index] = processed
        return {
            "record_id": row["record_id"],
            "signal": torch.from_numpy(processed.copy()),
            "label": LABELS.index(row["label"]),
        }


def collate_records(items: list[dict]) -> dict:
    if not items:
        raise ValueError("Cannot batch an empty list")
    if len({item["record_id"] for item in items}) != len(items):
        raise ValueError("Repeated recording within a batch")
    for item in items:
        signal = item["signal"]
        if signal.ndim != 1 or signal.numel() == 0 or not signal.is_floating_point():
            raise ValueError("Expected a nonempty floating point channel")
        if not torch.isfinite(signal).all() or item["label"] not in range(len(LABELS)):
            raise ValueError("Invalid signal values or label")
    lengths = torch.tensor([item["signal"].numel() for item in items], dtype=torch.long)
    signals = torch.zeros(len(items), 1, int(lengths.max()), dtype=torch.float32)
    mask = torch.arange(signals.shape[-1])[None, :] < lengths[:, None]
    for index, item in enumerate(items):
        signals[index, 0, : lengths[index]] = item["signal"]
    return {
        "signals": signals,
        "mask": mask,
        "lengths": lengths,
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "record_ids": [item["record_id"] for item in items],
    }


class LengthBucketSampler(Sampler):
    """Seeded local length sorting reduces padding; no recording is dropped."""

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        *,
        seed: int = 42,
        shuffle: bool = True,
    ):
        if not lengths or any(length <= 0 for length in lengths) or batch_size <= 0:
            raise ValueError("Invalid batch size or recording lengths")
        self.lengths, self.batch_size = list(lengths), batch_size
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        # Select the remainder before sorting, so long recordings are not singled out.
        remainder = len(indices) % self.batch_size
        batches = [indices[:remainder]] if remainder else []
        indices = indices[remainder:]
        bucket_size = self.batch_size * 20 if self.shuffle else max(1, len(indices))
        for start in range(0, len(indices), bucket_size):
            bucket = sorted(
                indices[start : start + bucket_size],
                key=lambda index: self.lengths[index],
            )
            batches.extend(
                bucket[offset : offset + self.batch_size]
                for offset in range(0, len(bucket), self.batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches
