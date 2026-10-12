"""Configured integrity preflight plus synthetic CPU/CUDA execution."""

import argparse
import json
from pathlib import Path

from rhythmlens.configuration import load_experiment, output_path
from rhythmlens.preflight import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Permit development source; training inputs only",
    )
    parser.add_argument("--integrity-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config, document = load_experiment(args.config, root, smoke=args.smoke)
    print(
        json.dumps(
            preflight(
                root,
                config,
                output=output_path(document, "smoke" if args.smoke else "train"),
                smoke=args.smoke,
                resume=document["execution"]["resume"],
                synthetic_execution=not args.integrity_only,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
