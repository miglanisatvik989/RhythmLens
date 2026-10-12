"""Verify the pre-readiness immutable snapshot without opening ECG files."""

import csv
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    snapshot = json.loads((root / "reports/frozen-provenance-sha256.json").read_text())
    changed = [
        name
        for name, digest in snapshot.items()
        if not (root / name).is_file()
        or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
    ]
    if changed:
        raise SystemExit(f"Frozen/historical files changed: {changed}")
    counts = {}
    for partition in ("train", "validation", "holdout"):
        with (
            root / "splits/challenge2017-v3-seed42-v1" / f"{partition}.csv"
        ).open() as stream:
            counts[partition] = sum(1 for _ in csv.DictReader(stream))
    if counts != {"train": 5969, "validation": 1280, "holdout": 1279}:
        raise SystemExit(f"Unexpected frozen counts: {counts}")
    print(
        json.dumps(
            {
                "unchanged_files": len(snapshot),
                "partition_counts": counts,
                "ECG_files_opened": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
