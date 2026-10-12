"""Record-level, non-destructive integrity audit; no quality-based exclusions."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import platform
import zipfile

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import matfile_version
import wfdb

from rhythmlens.data import (
    LABELS,
    MANIFESTS,
    RAW,
    read_labels,
    relative,
    sha256,
    utc_now,
    write_json,
)

POLICY = {
    "version": "challenge2017-audit-v1",
    "label_order": list(LABELS),
    "sampling_rate_hz": 300,
    "physical_conversion": "(ADC - baseline) / gain, then convert declared units to mV",
    "baseline_default": "WFDB ADC zero when baseline absent",
    "flat_definition": "all physical samples exactly equal; flag and retain",
    "noise_policy": "retain every ~ label; no quality threshold or filtering",
    "header_advisories": "zero checksum is unverified; initial value is unused by format 16; flag discrepancies without altering samples",
    "integrity_policy": "nonzero checksum mismatch blocks; all originals must match archive bytes and MAT/WFDB payloads must agree",
    "duplicate_definition": "exact float64 physical mV samples, sample count, channels and sampling rate",
    "subject_policy": "unknown; no verified patient identifiers in these inputs",
    "blocking": "any structural/data/label/provenance error or conflicting duplicate labels",
    "preprocessing": "unit decoding only; no normalization, filtering, resampling, or fitting",
}


class RecordError(ValueError):
    """An identified record integrity failure."""


def decode_record(mat: Path, hea: Path) -> tuple[np.ndarray, dict]:
    """Return physical mV without changing either source file.

    Use scipy for MAT V4 payloads and WFDB for header semantics. Require
    explicit calibration and units instead of accepting uncalibrated defaults.
    """
    lines = [
        line.strip()
        for line in hea.read_text(encoding="ascii").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != 2 or len(lines[0].split()) < 4:
        raise RecordError("malformed_header")
    fields = lines[1].split()
    if len(fields) < 6:
        raise RecordError("missing_calibration")
    if fields[0] != mat.name:
        raise RecordError("signal_filename_mismatch")
    calibration = re.fullmatch(
        r"([+-]?[\d.eE]+)(?:\(([+-]?\d+)\))?/([^\s]+)", fields[2]
    )
    if not calibration:
        raise RecordError("missing_or_invalid_gain_units")
    declared_gain = float(calibration[1])
    if not np.isfinite(declared_gain) or declared_gain <= 0:
        raise RecordError("invalid_gain")
    header = wfdb.rdheader(str(hea.with_suffix("")))
    if header.record_name != mat.stem or hea.stem != mat.stem:
        raise RecordError("record_name_mismatch")
    if header.n_sig != 1:
        raise RecordError("not_single_channel")
    if header.fs != 300:
        raise RecordError("unexpected_sampling_rate")
    if header.file_name != [mat.name]:
        raise RecordError("signal_filename_mismatch")
    if (
        header.fmt != ["16"]
        or header.samps_per_frame != [1]
        or header.skew[0] not in (None, 0)
    ):
        raise RecordError("unsupported_signal_encoding")
    if not header.sig_len or header.sig_len <= 0:
        raise RecordError("invalid_declared_sample_count")
    gain, baseline, units = header.adc_gain[0], header.baseline[0], header.units[0]
    if gain != declared_gain or not np.isfinite(baseline):
        raise RecordError("invalid_calibration")
    scales = {"mV": 1.0, "V": 1000.0, "uV": 0.001}
    if units not in scales:
        raise RecordError("unsupported_physical_units")
    if matfile_version(mat)[0] != 0:
        raise RecordError("not_matlab_v4")
    values = loadmat(mat).get("val")
    if values is None or values.shape != (1, header.sig_len):
        raise RecordError("sample_count_or_channel_shape_mismatch")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise RecordError("invalid_signal_type")
    digital = values[0].astype(np.float64)
    if not np.isfinite(digital).all():
        raise RecordError("nonfinite_signal")
    if (
        not np.equal(digital, np.rint(digital)).all()
        or (digital < -32768).any()
        or (digital > 32767).any()
    ):
        raise RecordError("invalid_format16_values")
    if (digital == -32768).any():
        raise RecordError("missing_signal_samples")
    advisories = []
    if header.init_value[0] is not None and header.init_value[0] != digital[0]:
        advisories.append("initial_value_mismatch_format16")
    computed = int(digital.astype(np.int64).sum()) % 65536
    if header.checksum[0] is not None:
        if header.checksum[0] == 0:
            advisories.append("zero_header_checksum_unverified")
        elif computed != header.checksum[0] % 65536:
            raise RecordError("signal_checksum_mismatch")
    wfdb_digital = wfdb.rdrecord(str(hea.with_suffix("")), physical=False).d_signal
    if wfdb_digital.shape != (header.sig_len, 1) or not np.array_equal(
        wfdb_digital[:, 0], digital
    ):
        raise RecordError("mat_wfdb_payload_mismatch")
    physical = (digital - baseline) / gain * scales[units]
    if not np.isfinite(physical).all():
        raise RecordError("nonfinite_physical_signal")
    physical[physical == 0] = 0.0  # canonicalize IEEE signed zero for hashing only
    metadata = {
        "channels": 1,
        "sampling_rate_hz": float(header.fs),
        "samples": int(header.sig_len),
        "duration_seconds": header.sig_len / header.fs,
        "gain": float(gain),
        "baseline": int(baseline),
        "original_units": units,
        "physical_units": "mV",
        "format": header.fmt[0],
        "byte_offset": header.byte_offset[0],
        "adc_zero": header.adc_zero[0],
        "declared_initial_value": header.init_value[0],
        "actual_initial_value": int(digital[0]),
        "declared_checksum": header.checksum[0],
        "computed_checksum_unsigned": computed,
        "advisories": advisories,
        "flat": bool(np.ptp(physical) == 0),
    }
    # MAT container timestamps/record names and file bytes are deliberately irrelevant.
    canonical = json.dumps(
        {
            k: metadata[k]
            for k in ("channels", "sampling_rate_hz", "samples", "physical_units")
        },
        sort_keys=True,
    ).encode()
    metadata["duplicate_group"] = hashlib.sha256(
        canonical + b"\n" + physical.astype("<f8").tobytes()
    ).hexdigest()
    calibration_key = json.dumps(
        {
            k: metadata[k]
            for k in (
                "gain",
                "baseline",
                "original_units",
                "sampling_rate_hz",
                "samples",
            )
        },
        sort_keys=True,
    ).encode()
    metadata["digital_decoding_sha256"] = hashlib.sha256(
        calibration_key + b"\n" + digital.astype("<f8").tobytes()
    ).hexdigest()
    return physical, metadata


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(root: Path, previews: bool = True) -> dict:
    root = root.resolve()
    if list((root / "splits").glob("*/frozen.json")):
        raise ValueError(
            "Frozen splits exist; refusing to replace their audit snapshot"
        )
    raw, output = root / RAW, root / MANIFESTS
    extracted = raw / "extracted"
    errors = []
    inputs = {}
    provenance_path = output / "provenance.json"
    if provenance_path.exists():
        inputs[relative(provenance_path, root)] = sha256(provenance_path)
        provenance = json.loads(provenance_path.read_text())
        for entry in provenance["files"].values():
            path = root / entry["path"]
            if not path.resolve().is_relative_to(raw.resolve()):
                errors.append("provenance_path_outside_raw")
            elif not path.is_file() or sha256(path) != entry["sha256"]:
                errors.append(f"provenance_checksum_mismatch:{entry['path']}")
    else:
        errors.append("missing_provenance")
    # Verify extracted originals against the downloaded archive, including reference.
    archive = raw / "training2017.zip"
    if archive.is_file():
        try:
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    path = extracted / info.filename
                    if not path.resolve().is_relative_to(extracted.resolve()):
                        errors.append("unsafe_archive_member")
                        continue
                    expected = hashlib.sha256(zf.read(info)).hexdigest()
                    if not path.is_file() or sha256(path) != expected:
                        errors.append(f"extraction_mismatch:{info.filename}")
        except (OSError, zipfile.BadZipFile) as exc:
            errors.append(f"invalid_archive:{type(exc).__name__}")
    else:
        errors.append("missing_archive")
    references = sorted(extracted.rglob("REFERENCE.csv"))
    labels, original = {}, {}
    for kind, path in (
        ("v3", raw / "REFERENCE-v3.csv"),
        ("archive", references[0] if len(references) == 1 else None),
    ):
        if path is None or not path.is_file():
            errors.append(f"missing_or_ambiguous_{kind}_reference")
            continue
        inputs[relative(path, root)] = sha256(path)
        try:
            loaded = read_labels(path)
            if kind == "v3":
                labels = loaded
            else:
                original = loaded
        except ValueError as exc:
            errors.append(str(exc))
    if set(labels) != set(original):
        errors.append("v3_archive_record_id_sets_differ")
    files = {suffix: defaultdict(list) for suffix in (".mat", ".hea")}
    for suffix in files:
        for path in sorted(extracted.rglob("*" + suffix)):
            files[suffix][path.stem].append(path)
    record_ids = sorted(
        set(labels) | set(original) | set(files[".mat"]) | set(files[".hea"])
    )
    rows = []
    groups = defaultdict(list)
    for record_id in record_ids:
        row = {
            "record_id": record_id,
            "subject_id": "unknown",
            "label": labels.get(record_id, ""),
            "archive_label": original.get(record_id, ""),
            "flags": [],
            "valid": False,
        }
        for suffix, key in ((".mat", "mat_path"), (".hea", "header_path")):
            paths = files[suffix].get(record_id, [])
            row[key] = relative(paths[0], root) if len(paths) == 1 else ""
            if len(paths) != 1:
                row["flags"].append(
                    "missing_" + key if not paths else "ambiguous_" + key
                )
            for path in paths:
                inputs[relative(path, root)] = sha256(path)
        if record_id not in labels:
            row["flags"].append("orphan_record_without_v3_label")
        if not row["flags"]:
            try:
                _, metadata = decode_record(
                    root / row["mat_path"], root / row["header_path"]
                )
                row.update(metadata)
                row["valid"] = True
                row["flags"].extend(metadata["advisories"])
                if metadata["flat"]:
                    row["flags"].append("flat")
                groups[row["duplicate_group"]].append(row)
            except Exception as exc:
                # A malformed record must remain represented, without stopping the inventory.
                row["flags"].append(
                    str(exc)
                    if isinstance(exc, RecordError)
                    else f"malformed_record:{type(exc).__name__}"
                )
        if not row["valid"]:
            errors.append(f"{record_id}:{'|'.join(row['flags'])}")
        rows.append(row)
    duplicates = []
    conflicts = []
    for group, members in sorted(groups.items()):
        if len(members) > 1:
            item = {
                "duplicate_group": group,
                "record_ids": [r["record_id"] for r in members],
                "labels": sorted({r["label"] for r in members}, key=LABELS.index),
            }
            duplicates.append(item)
            if len(item["labels"]) > 1:
                conflicts.append(item)
    if conflicts:
        errors.append("conflicting_labels_within_duplicate_groups")
    if not rows:
        errors.append("no_records")
    changes = [
        {"record_id": rid, "archive": original[rid], "v3": labels[rid]}
        for rid in sorted(set(labels) & set(original))
        if labels[rid] != original[rid]
    ]
    summaries = {}
    for label in LABELS:
        durations = [
            r["duration_seconds"] for r in rows if r["valid"] and r["label"] == label
        ]
        summaries[label] = {
            "labeled_records": list(labels.values()).count(label),
            "decoded_records": len(durations),
            "duration_seconds": {
                name: float(fn(durations)) if durations else None
                for name, fn in (
                    ("min", np.min),
                    ("median", np.median),
                    ("mean", np.mean),
                    ("max", np.max),
                )
            },
        }
    preview_paths = []
    if previews:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        preview_dir = root / "outputs/challenge2017/previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for label in LABELS:
            examples = [r for r in rows if r["valid"] and r["label"] == label][:5]
            for row in examples:
                physical, _ = decode_record(
                    root / row["mat_path"], root / row["header_path"]
                )
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(
                    np.arange(physical.size) / row["sampling_rate_hz"],
                    physical,
                    linewidth=0.5,
                )
                ax.set(
                    xlabel="Time (s)",
                    ylabel="ECG (mV)",
                    title=f"{row['record_id']} — v3 {label}; data audit before split freeze",
                )
                fig.tight_layout()
                path = preview_dir / (row["record_id"] + ".png")
                fig.savefig(path, dpi=120)
                plt.close(fig)
                preview_paths.append(relative(path, root))
    fields = [
        "record_id",
        "subject_id",
        "label",
        "archive_label",
        "mat_path",
        "header_path",
        "valid",
        "flags",
        "channels",
        "sampling_rate_hz",
        "samples",
        "duration_seconds",
        "gain",
        "baseline",
        "original_units",
        "physical_units",
        "format",
        "byte_offset",
        "adc_zero",
        "declared_initial_value",
        "actual_initial_value",
        "declared_checksum",
        "computed_checksum_unsigned",
        "flat",
        "duplicate_group",
        "digital_decoding_sha256",
    ]
    manifest = output / "records.csv"
    write_csv(
        manifest,
        [
            {k: "|".join(r[k]) if k == "flags" else r.get(k, "") for k in fields}
            for r in rows
        ],
        fields,
    )
    policy_path = root / "configs/challenge2017-audit-v1.json"
    write_json(policy_path, POLICY)
    inputs[relative(policy_path, root)] = sha256(policy_path)
    report = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "data_kind": "actual acquisition or supplied inputs; see provenance",
        "policy": POLICY,
        "validation_passed": not errors,
        "errors": errors,
        "total_inventory_records": len(rows),
        "v3_class_counts": dict(Counter(labels.values())),
        "archive_class_counts": dict(Counter(original.values())),
        "label_changes": changes,
        "class_summaries": summaries,
        "invalid_records": sum(not r["valid"] for r in rows),
        "flag_counts": dict(Counter(flag for r in rows for flag in r["flags"])),
        "duplicate_groups": duplicates,
        "conflicting_duplicate_groups": conflicts,
        "preview_paths": preview_paths,
        "inputs_sha256": inputs,
        "manifest_path": relative(manifest, root),
        "manifest_sha256": sha256(manifest),
        "python_version": platform.python_version(),
        "implementation_sha256": {
            p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "dependency_lock_sha256": sha256(root / "uv.lock")
        if (root / "uv.lock").exists()
        else None,
        "versions": {
            name: version(name)
            for name in ("rhythmlens", "numpy", "scipy", "wfdb", "matplotlib")
        },
    }
    write_json(output / "audit.json", report)
    lines = [
        "# Challenge 2017 v3 data audit",
        "",
        f"Validation passed: **{not errors}**.",
        "",
        f"Inventory: {len(rows)} records; invalid: {report['invalid_records']}; label changes: {len(changes)}.",
        f"Exact duplicate groups: {len(duplicates)}; conflicting groups: {len(conflicts)}.",
        "",
        "| Label | Archive count | v3 count | Decoded | Duration min / median / mean / max (s) |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for label, summary in summaries.items():
        durations = " / ".join(
            f"{v:.3f}" if v is not None else "undefined"
            for v in summary["duration_seconds"].values()
        )
        lines.append(
            f"| {label} | {report['archive_class_counts'].get(label, 0)} | "
            f"{summary['labeled_records']} | {summary['decoded_records']} | {durations} |"
        )
    lines += [
        "",
        "Flags: " + json.dumps(report["flag_counts"]),
        "",
        "No quality exclusions. Flat signals are flagged and retained; noisy labels remain eligible.",
        "Subject IDs are unknown: recording and exact-duplicate grouping cannot establish patient independence.",
        "No transformations beyond calibrated unit conversion; original MAT/header files are preserved.",
        "No AF predictions or model metrics were computed; deferral/evaluation work remains pending.",
        "",
        "Waveform previews (local ignored outputs; generated before holdout assignment):",
        "",
    ]
    lines += [f"- `{p}`" for p in preview_paths]
    lines += ["", "Validation blockers:", ""] + (
        [f"- {e}" for e in errors] or ["None."]
    )
    lines += [
        "",
        "Machine-readable details, changed label IDs, duplicate groups, versions and hashes:",
        "`data/manifests/challenge2017-v3/audit.json`.",
        "",
        "Attribution and license: see `data/manifests/challenge2017-v3/provenance.json` and README.",
        "",
    ]
    report_path = root / "reports/challenge2017-audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    return report
