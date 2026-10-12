#!/usr/bin/env python3
"""Set up and run the environment selected by config.yaml, without activation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ACTIONS = (
    "show",
    "setup",
    "doctor",
    "verify-data",
    "test",
    "fit-preprocessing",
    "smoke",
    "train",
    "evaluate",
)


def configuration_module():
    # Load this dependency-light module even before RhythmLens is installed.
    spec = importlib.util.spec_from_file_location(
        "rhythmlens_configuration", ROOT / "src/rhythmlens/configuration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_yaml():
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None and yaml.__version__ == "6.0.3":
        return
    if os.environ.get("RHYTHMLENS_YAML_BOOTSTRAP") == "1":
        raise ValueError("Pinned PyYAML bootstrap failed")
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("Install uv first; see README.md")
    os.environ["RHYTHMLENS_YAML_BOOTSTRAP"] = "1"
    os.execv(
        uv,
        [
            uv,
            "run",
            "--no-project",
            "--python",
            (ROOT / ".python-version").read_text().strip(),
            "--with",
            "pyyaml==6.0.3",
            "--default-index",
            "https://pypi.org/simple",
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
    )


def python_path(root, profile):
    module = configuration_module()
    venv = module.relative_path(root, module.PROFILES[profile]["venv"])
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def commands_for(action, root, path, document):
    module = configuration_module()
    profile = (
        document["evaluation"]["profile"]
        if action == "evaluate"
        else document["environment"]["profile"]
    )
    descriptor = module.PROFILES[profile]
    python = str(python_path(root, profile))
    config = str(path.resolve())
    output = module.output_path(document, action)
    if action == "setup":
        commands = [
            [
                "uv",
                "sync",
                "--project",
                descriptor["project"],
                "--locked",
                "--python",
                (ROOT / ".python-version").read_text().strip(),
                "--default-index",
                "https://pypi.org/simple",
            ]
        ]
        if profile == "cpu":
            commands[0].append("--all-extras")
        else:
            commands.append(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    python,
                    "--no-deps",
                    "--no-sources",
                    "--no-build-isolation",
                    "-e",
                    ".",
                ]
            )
        commands.append(["uv", "pip", "check", "--python", python])
        return commands
    if action in ("doctor", "verify-data"):
        command = [
            python,
            str(root / "scripts/doctor_environment.py"),
            "--config",
            config,
        ]
        if action == "verify-data":
            command.extend(["--smoke", "--integrity-only"])
        return [command]
    if action == "test":
        return [[python, "-m", "pytest", "-q", "tests"]]
    if action == "evaluate":
        checkpoint = str(
            module.relative_path(root, module.output_path(document, "train"))
            / "best.pt"
        )
        return [
            [
                python,
                "-m",
                "rhythmlens",
                "evaluate-rhythm",
                "--checkpoint",
                checkpoint,
                "--config",
                config,
                "--output",
                output,
            ]
        ]
    command = {
        "train": "train-rhythm",
        "smoke": "smoke-rhythm",
        "fit-preprocessing": "fit-preprocessing",
    }[action]
    return [
        [python, "-m", "rhythmlens", command, "--config", config, "--output", output]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show commands without installing or executing them",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Development doctor: training inputs and dirty source allowed",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Explicit train resume without changing YAML",
    )
    args = parser.parse_args()
    try:
        ensure_yaml()
        module = configuration_module()
        path = args.config if args.config.is_absolute() else ROOT / args.config
        flat, document = module.load_experiment(
            path, ROOT, smoke=args.action == "smoke"
        )
        profile = document["environment"]["profile"]
        if profile == "cuda128" and (
            platform.system() != "Linux" or platform.machine() != "x86_64"
        ):
            raise ValueError(
                "cuda128 is currently locked for Linux x86_64 (including WSL)"
            )
        if args.action == "show":
            print(
                json.dumps(
                    {
                        "profile": profile,
                        "environment_python": str(python_path(ROOT, profile)),
                        "resolved_training_config": {
                            k: v for k, v in flat.items() if k != "_experiment"
                        },
                        "configuration_sha256": flat["_experiment"]["source_sha256"],
                        "output_directories": {
                            a: module.output_path(document, a)
                            for a in ("train", "smoke", "fit-preprocessing", "evaluate")
                        },
                        "setup_commands": commands_for("setup", ROOT, path, document),
                    },
                    indent=2,
                )
            )
            return
        commands = commands_for(args.action, ROOT, path, document)
        if args.resume:
            if args.action != "train":
                raise ValueError("--resume requires the train action")
            commands[0].append("--resume")
        if args.smoke:
            if args.action != "doctor":
                raise ValueError(
                    "--smoke is a doctor option; use the smoke action for training"
                )
            commands[0].append("--smoke")
        for command in commands:
            print("+ " + shlex.join(command), flush=True)
        if args.dry_run:
            return
        if args.action == "setup" and shutil.which("uv") is None:
            raise ValueError("Install uv first; see README.md")
        execution_profile = (
            document["evaluation"]["profile"] if args.action == "evaluate" else profile
        )
        if (
            args.action != "setup"
            and not python_path(ROOT, execution_profile).is_file()
        ):
            raise ValueError(
                "Selected environment is missing; run python3 scripts/manage.py setup"
            )
        env = dict(os.environ)
        env["UV_PROJECT_ENVIRONMENT"] = (
            str(Path(commands[0][0]).parent.parent)
            if args.action == "evaluate"
            else str(python_path(ROOT, profile).parent.parent)
        )
        for command in commands:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    except KeyboardInterrupt:
        print(
            "Interrupted. Use execution.resume: true to repeat any partial epoch from last.pt.",
            file=sys.stderr,
        )
        raise SystemExit(130) from None
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None
    except (ValueError, OSError) as exc:
        parser.exit(1, f"rhythmlens: {exc}\n")


if __name__ == "__main__":
    main()
