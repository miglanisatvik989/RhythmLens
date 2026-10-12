"""Command-line entry points; heavy dependencies are imported only when needed."""

import argparse
import json
from pathlib import Path

from rhythmlens import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rhythmlens", description="RhythmLens ECG research tools"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command")
    acquire_parser = sub.add_parser(
        "acquire", help="Acquire Challenge 2017 archive and v3 labels"
    )
    acquire_parser.add_argument(
        "--local-archive", type=Path, help="Reuse a local training2017.zip"
    )
    acquire_parser.add_argument(
        "--local-reference", type=Path, help="Reuse a local REFERENCE-v3.csv"
    )
    acquire_parser.add_argument(
        "--mirror",
        choices=("physionet", "s3"),
        default="physionet",
        help="Official download host (s3 is the PhysioNet public bucket)",
    )
    audit_parser = sub.add_parser(
        "audit", help="Audit labels, calibration, integrity and duplicates"
    )
    audit_parser.add_argument(
        "--no-previews", action="store_true", help="Skip waveform images"
    )
    sub.add_parser(
        "split", help="Freeze validated 70/15/15 recording splits with seed 42"
    )
    args = parser.parse_args()
    root = Path.cwd()
    if args.command is None:
        parser.print_help()
        return
    try:
        if args.command == "acquire":
            from rhythmlens.data import acquire

            result = acquire(
                root, args.local_archive, args.local_reference, args.mirror
            )
            print(json.dumps({"acquired": list(result["files"])}))
        elif args.command == "audit":
            from rhythmlens.audit import audit

            result = audit(root, previews=not args.no_previews)
            print(
                json.dumps(
                    {
                        "validation_passed": result["validation_passed"],
                        "records": result["total_inventory_records"],
                        "error_count": len(result["errors"]),
                        "first_errors": result["errors"][:10],
                        "report": "data/manifests/challenge2017-v3/audit.json",
                    }
                )
            )
            if not result["validation_passed"]:
                raise SystemExit(1)
        else:
            from rhythmlens.splits import freeze

            print(json.dumps(freeze(root), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"rhythmlens: {exc}\n")
