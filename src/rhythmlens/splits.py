"""Deterministic recording splits with indivisible exact-duplicate groups."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import tempfile

from rhythmlens.audit import write_csv
from rhythmlens.data import (
    LABELS,
    MANIFESTS,
    RAW,
    relative,
    sha256,
    utc_now,
    write_json,
)

SPLIT_NAMES = ("train", "validation", "holdout")
SPLIT_VERSION = "challenge2017-v3-seed42-v1"
CONFIG = {
    "schema_version": 1,
    "split_version": SPLIT_VERSION,
    "seed": 42,
    "fractions": {"train": 0.70, "validation": 0.15, "holdout": 0.15},
    "label_order": list(LABELS),
    "label_version": "REFERENCE-v3",
    "algorithm": "per-class largest-group-first, seeded ties, largest remaining target deficit; v1",
    "grouping": "exact physical signal duplicate_group; otherwise recording",
    "conflict_policy": "block freeze; never relabel or silently drop conflicting duplicates",
    "eligibility": "all valid labeled recordings, including noisy and flat recordings",
    "subject_ids": "unknown; patient independence cannot be verified",
    "holdout_policy": "no development or tuning; freeze models, thresholds, quality gates and analyses before evaluation",
}


def assign_splits(rows: list[dict]) -> list[dict]:
    """Assign all input records exactly once; fail instead of silently dropping data."""
    ids = [r["record_id"] for r in rows]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Empty input or repeated record IDs")
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["record_id"]):
        if row["label"] not in LABELS or not row.get("duplicate_group"):
            raise ValueError("Invalid label or absent duplicate group")
        if row.get("subject_id") != "unknown":
            raise ValueError(
                "Verified subject information requires a reviewed subject-grouped split policy"
            )
        groups[row["duplicate_group"]].append(row)
    for group, members in groups.items():
        if len({r["label"] for r in members}) != 1:
            raise ValueError(
                f"Conflicting labels in duplicate group {group}: {[r['record_id'] for r in members]}"
            )
    rng = random.Random(CONFIG["seed"])
    assigned = []
    for label in LABELS:
        candidates = [
            members
            for _, members in sorted(groups.items())
            if members[0]["label"] == label
        ]
        if len(candidates) < 3:
            raise ValueError(
                f"Class {label} needs at least three independent groups for class coverage"
            )
        rng.shuffle(candidates)
        candidates.sort(
            key=len, reverse=True
        )  # stable: random seed breaks equal-size ties
        total = sum(map(len, candidates))
        targets = {s: CONFIG["fractions"][s] * total for s in SPLIT_NAMES}
        counts = Counter()
        for index, members in enumerate(candidates):
            empty = [s for s in SPLIT_NAMES if counts[s] == 0]
            allowed = empty if len(candidates) - index == len(empty) else SPLIT_NAMES
            chosen = max(allowed, key=lambda s: targets[s] - counts[s])
            counts[chosen] += len(members)
            assigned.extend({**row, "split": chosen} for row in members)
    assert_partitions(assigned, set(ids))
    return sorted(assigned, key=lambda r: r["record_id"])


def assert_partitions(rows: list[dict], eligible: set[str]) -> None:
    ids, groups = {}, {}
    coverage = defaultdict(set)
    for row in rows:
        record_id, group, split = row["record_id"], row["duplicate_group"], row["split"]
        if split not in SPLIT_NAMES or record_id in ids:
            raise ValueError("Invalid partition or repeated record ID")
        if group in groups and groups[group] != split:
            raise ValueError("Duplicate-group leakage")
        ids[record_id], groups[group] = split, split
        coverage[split].add(row["label"])
    if set(ids) != eligible:
        raise ValueError("Eligible record accounting failed")
    if any(coverage[s] != set(LABELS) for s in SPLIT_NAMES):
        raise ValueError("Class coverage failed")


def freeze(root: Path) -> dict:
    root = root.resolve()
    destination = root / "splits" / SPLIT_VERSION
    config_path = root / "configs" / f"{SPLIT_VERSION}.json"
    if destination.exists():
        raise ValueError(
            f"Frozen split already exists: {destination.relative_to(root)}; refusing overwrite"
        )
    if config_path.exists():
        raise ValueError("Split configuration already exists; refusing overwrite")
    audit_path = root / MANIFESTS / "audit.json"
    report = json.loads(audit_path.read_text())
    if (
        not report["validation_passed"]
        or report["errors"]
        or report["conflicting_duplicate_groups"]
    ):
        raise ValueError(
            f"Audit validation failed: {report['errors']}; splits remain pending"
        )
    if report["implementation_sha256"] != {
        p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
    }:
        raise ValueError(
            "Implementation changed after audit; rerun audit before freezing"
        )
    lock_digest = sha256(root / "uv.lock") if (root / "uv.lock").exists() else None
    if report["dependency_lock_sha256"] != lock_digest:
        raise ValueError("Dependency lock changed after audit")
    audited_records = {
        name
        for name in report["inputs_sha256"]
        if Path(name).suffix in (".mat", ".hea")
    }
    current_records = {
        relative(path, root)
        for suffix in (".mat", ".hea")
        for path in (root / RAW / "extracted").rglob("*" + suffix)
    }
    if current_records != audited_records:
        raise ValueError("Record inventory changed after audit")
    manifest = root / report["manifest_path"]
    if (
        not manifest.resolve().is_relative_to(root)
        or sha256(manifest) != report["manifest_sha256"]
    ):
        raise ValueError("Manifest changed after audit")
    for name, expected in report["inputs_sha256"].items():
        path = root / name
        if (
            not path.resolve().is_relative_to(root)
            or not path.is_file()
            or sha256(path) != expected
        ):
            raise ValueError(f"Audited input changed or missing: {name}")
    # Include the large downloaded archive too; provenance is bound to its content.
    provenance = json.loads((root / MANIFESTS / "provenance.json").read_text())
    for entry in provenance["files"].values():
        path = root / entry["path"]
        if not path.resolve().is_relative_to(root) or sha256(path) != entry["sha256"]:
            raise ValueError(f"Acquired input changed: {entry['path']}")
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(r["valid"] != "True" for r in rows):
        raise ValueError("Invalid records cannot be silently excluded")
    assigned = assign_splits(rows)
    counts = {
        s: dict(Counter(r["label"] for r in assigned if r["split"] == s))
        for s in SPLIT_NAMES
    }
    totals = {s: sum(counts[s].values()) for s in SPLIT_NAMES}
    summary = {
        "records": len(rows),
        "class_counts": counts,
        "split_counts": totals,
        "actual_proportions": {s: totals[s] / len(rows) for s in SPLIT_NAMES},
        "class_proportions": {
            s: {
                label: counts[s][label] / sum(counts[t][label] for t in SPLIT_NAMES)
                for label in LABELS
            }
            for s in SPLIT_NAMES
        },
        "record_overlap": 0,
        "duplicate_group_overlap": 0,
        "class_coverage": True,
        "patient_independence": "unverified: patient IDs unavailable; no identities inferred",
        "source_time_independence": "unverified: single source and no verified longitudinal subject/time linkage",
        "evaluation": "pending; holdout frozen for future final evaluation only",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".freeze-", dir=destination.parent) as temp:
        staged = Path(temp) / SPLIT_VERSION
        staged.mkdir()
        fields = ["record_id", "subject_id", "label", "duplicate_group", "split"]
        for split in SPLIT_NAMES:
            write_csv(
                staged / f"{split}.csv",
                [{k: r[k] for k in fields} for r in assigned if r["split"] == split],
                fields,
            )
        write_json(staged / "config.json", CONFIG)
        write_json(staged / "summary.json", summary)
        lines = [
            "# Frozen Challenge 2017 recording splits",
            "",
            "Seed 42; label order N, A, O, ~. Exact duplicate groups remain together.",
            "",
            "| Split | Records | Proportion | N | A | O | ~ |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for split in SPLIT_NAMES:
            lines.append(
                f"| {split} | {totals[split]} | {totals[split] / len(rows):.6f} | "
                + " | ".join(str(counts[split][label]) for label in LABELS)
                + " |"
            )
        lines += [
            "",
            "All eligible records accounted for; no record or duplicate-group overlap; all classes covered.",
            "Patient independence cannot be established: subject IDs are unknown. Source/time independence is unverified.",
            "Noisy and flat records retained. No model results or quality gating decisions exist.",
            "Holdout is reserved for evaluation after model, threshold, quality-gating and analysis choices are frozen.",
            "",
        ]
        (staged / "README.md").write_text("\n".join(lines))
        hashes = {p.name: sha256(p) for p in sorted(staged.iterdir())}
        frozen = {
            "created_at_utc": utc_now(),
            "split_version": SPLIT_VERSION,
            "files_sha256": hashes,
            "source_manifest_sha256": sha256(manifest),
            "source_audit_sha256": sha256(audit_path),
            "source_provenance_sha256": sha256(root / MANIFESTS / "provenance.json"),
            "dependency_lock_sha256": sha256(root / "uv.lock")
            if (root / "uv.lock").exists()
            else None,
            "implementation_sha256": {
                p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "assignment_sha256": hashlib.sha256(
                json.dumps(
                    [(r["record_id"], r["split"]) for r in assigned],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
        write_json(staged / "frozen.json", frozen)
        # Exclusive config creation and directory publication: no overwrite escape hatch.
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("x") as stream:
            stream.write((staged / "config.json").read_text())
        try:
            staged.rename(destination)
        except Exception:
            config_path.unlink()
            raise
    return summary
