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
    for command, help_text in (
        (
            "fit-preprocessing",
            "Fit the fixed amplitude scale on frozen training records",
        ),
        (
            "smoke-rhythm",
            "Bounded training-only wiring check; no validation or holdout signals",
        ),
        (
            "train-rhythm",
            "Train the baseline with development validation; no final holdout",
        ),
    ):
        command_parser = sub.add_parser(command, help=help_text)
        command_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
        command_parser.add_argument(
            "--output",
            type=Path,
            required=True,
            help="New ignored run directory; refuses overwrite",
        )
        if command == "train-rhythm":
            command_parser.add_argument(
                "--resume",
                action="store_true",
                help="Resume from the last completed epoch",
            )
        if command != "fit-preprocessing":
            command_parser.add_argument(
                "--preprocessing",
                type=Path,
                help="Reuse a full-training preprocessing.json",
            )
    evaluation_parser = sub.add_parser(
        "evaluate-rhythm", help="Evaluate a baseline checkpoint on validation only"
    )
    evaluation_parser.add_argument("--config", type=Path)
    evaluation_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluation_parser.add_argument("--output", type=Path, required=True)
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
        elif args.command == "split":
            from rhythmlens.splits import freeze

            print(json.dumps(freeze(root), indent=2))
        elif args.command == "evaluate-rhythm":
            from rhythmlens.rhythm_training import evaluate_command

            runtime = None
            if args.config:
                from rhythmlens.configuration import evaluation_config, load_experiment

                flat, document = load_experiment(args.config, root)
                runtime = evaluation_config(flat, document)
            result = evaluate_command(
                root, args.checkpoint, args.output, runtime_config=runtime
            )
            print(
                json.dumps(
                    {"partition": result["partition"], "metrics": result["metrics"]},
                    indent=2,
                )
            )
        else:
            from rhythmlens.rhythm_training import (
                fit_command,
                load_config,
                run_training,
            )

            config = load_config(args.config, smoke=args.command == "smoke-rhythm")
            if args.command == "fit-preprocessing":
                result = fit_command(root, config, args.output)
                print(json.dumps(result, indent=2))
            else:
                result = run_training(
                    root,
                    config,
                    args.output,
                    smoke=args.command == "smoke-rhythm",
                    supplied_preprocessing=args.preprocessing,
                    resume=getattr(args, "resume", False)
                    or bool(
                        config.get("_experiment")
                        and config["_experiment"]["document"]["execution"]["resume"]
                    ),
                )
                print(
                    json.dumps(
                        {
                            "kind": result["kind"],
                            "steps": result["steps"],
                            "stop_reason": result["stop_reason"],
                            "report": str(args.output / "run-report.json"),
                        },
                        indent=2,
                    )
                )
    except ImportError as exc:
        parser.exit(
            1,
            f"rhythmlens: missing optional dependency ({exc}); install audit and train extras\n",
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"rhythmlens: {exc}\n")
