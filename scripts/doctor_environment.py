"""Configured integrity preflight plus synthetic CPU/CUDA execution."""

import argparse
import json
from pathlib import Path

from rhythmlens.configuration import load_experiment, output_path
from rhythmlens.preflight import preflight


def verify_data(root, config):
    """Verify frozen metadata and development raw checksums without decoding."""
    from rhythmlens.data import sha256
    from rhythmlens.rhythm_data import FrozenRecords, project_path

    counts = {}
    checked = 0
    training = None
    for partition in ("train", "validation"):
        records = FrozenRecords(root, config["split_dir"], partition)
        if partition == "train":
            training = records
        counts[partition] = len(records.rows)
        for row in records.rows:
            for key in ("mat_path", "header_path"):
                name = row[key]
                path = project_path(root, name)
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing local ECG input: {name}; restore recorded raw files without refreezing"
                    )
                if sha256(path) != records.audit["inputs_sha256"].get(name):
                    raise ValueError(f"Audited ECG input changed: {name}")
                checked += 1
    return {
        "kind": "data integrity verification; not hardware or model verification",
        "partition_records": counts,
        "raw_files_checked": checked,
        "holdout_signals_accessed": False,
        "ECG_signals_decoded": 0,
        "synthetic_execution": False,
        "output_created": False,
        "data_identity": training.identity,
        "data_compatibility": training.compatibility,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Permit development source; training inputs only",
    )
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="Repeatable metadata/raw checksums only; no run output or device execution",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config, document = load_experiment(
        args.config, root, smoke=args.smoke and not args.integrity_only
    )
    if args.integrity_only:
        try:
            result = verify_data(root, config)
        except (ValueError, OSError) as exc:
            parser.exit(1, f"rhythmlens verify-data: {exc}\n")
        print(json.dumps(result, indent=2))
        return
    print(
        json.dumps(
            preflight(
                root,
                config,
                output=output_path(document, "smoke" if args.smoke else "train"),
                smoke=args.smoke,
                resume=document["execution"]["resume"],
                synthetic_execution=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
