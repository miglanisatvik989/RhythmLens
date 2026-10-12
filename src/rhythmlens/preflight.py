"""Training gates; metadata/raw checksums only, never holdout signal decoding."""

import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import tomllib

import torch

from rhythmlens.configuration import PROFILES, relative_path
from rhythmlens.data import sha256
from rhythmlens.rhythm_data import FrozenRecords, project_path


def lock_environment(root, profile):
    lock = relative_path(root, PROFILES[profile]["lock"])
    value = tomllib.loads(lock.read_text())
    installed = {
        d.metadata["Name"].lower().replace("_", "-"): d.version
        for d in importlib.metadata.distributions()
        if d.metadata.get("Name")
    }
    versions = {}
    for package in value["package"]:
        versions.setdefault(package["name"], set()).add(package["version"])
    errors = []
    for name, version in installed.items():
        if name in versions and version not in versions[name]:
            errors.append(f"{name}=={version}; lock permits {sorted(versions[name])}")
    # Required decoding, training, plotting and build tools must be installed.
    for name in (
        "numpy",
        "scipy",
        "wfdb",
        "torch",
        "matplotlib",
        "pyyaml",
        "editables",
        "hatchling",
    ):
        if name not in installed:
            errors.append(f"Missing required dependency: {name}")
    if errors:
        raise ValueError("Active lock incompatibility: " + "; ".join(errors))
    result = subprocess.run(
        ["uv", "pip", "check", "--python", sys.executable],
        capture_output=True,
        text=True,
        env={**os.environ, "UV_CACHE_DIR": "/tmp/rhythmlens-uv-cache"},
    )
    if result.returncode:
        raise ValueError(
            "Dependency consistency failed: " + result.stderr + result.stdout
        )
    result = subprocess.run(
        [
            "uv",
            "lock",
            "--check",
            "--offline",
            "--default-index",
            "https://pypi.org/simple",
            "--project",
            str(lock.parent),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "UV_CACHE_DIR": "/tmp/rhythmlens-uv-cache"},
    )
    if result.returncode:
        raise ValueError("Lock/descriptor consistency failed: " + result.stderr)
    return {
        "profile": profile,
        "lock_sha256": sha256(lock),
        "interpreter": sys.executable,
    }


def preflight(
    root, config, *, output=None, smoke=False, resume=False, synthetic_execution=True
):
    from rhythmlens.rhythm_training import (
        code_metadata,
        setup_runtime,
        preprocessing_for_run,
        environment_metadata,
    )
    from rhythmlens.rhythm_model import build_model, temporal_capability

    root = root.resolve()
    source = code_metadata(root, config)
    if not smoke and not config.get("noncanonical_development", False):
        if not source["revision"] or source["working_tree_dirty"] is not False:
            raise ValueError(
                "Canonical training requires clean committed source; explicitly set execution.noncanonical_development: true for development"
            )
    device = setup_runtime(config)
    context = config.get("_experiment")
    if (
        not smoke
        and not config.get("noncanonical_development", False)
        and context is None
    ):
        raise ValueError(
            "Canonical training requires resolved YAML with an explicit environment profile"
        )
    environment = lock_environment(root, context["profile"]) if context else None
    train = FrozenRecords(root, config["split_dir"], "train")
    partitions = (
        [train]
        if smoke
        else [train, FrozenRecords(root, config["split_dir"], "validation")]
    )
    checked = 0
    for records in partitions:
        for row in records.rows:
            for key in ("mat_path", "header_path"):
                path = project_path(root, row[key])
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing local ECG input: {row[key]}; restore recorded raw files without refreezing"
                    )
                if sha256(path) != records.audit["inputs_sha256"].get(row[key]):
                    raise ValueError(f"Audited ECG input changed: {row[key]}")
                checked += 1
    reuse = context["document"]["preprocessing"]["reuse"] if context else None
    if reuse:
        preprocessing_for_run(train, relative_path(root, reuse))
    if output is not None:
        path = project_path(root, output)
        if len(path.relative_to(root).parts) < 2 or path.relative_to(root).parts[
            0
        ] not in ("outputs", "runs", "checkpoints"):
            raise ValueError("Invalid run output location")
        if resume:
            if smoke:
                raise ValueError(
                    "Epoch-boundary resume is supported for full training only"
                )
            if not (path / "last.pt").is_file() or not (path / "status.json").is_file():
                raise ValueError(
                    "Resume requires an existing lifecycle record and usable last.pt"
                )
            if json.loads((path / "status.json").read_text())["state"] == "completed":
                raise ValueError("Completed runs cannot be resumed or overwritten")
        elif path.exists():
            raise ValueError(
                "Output already exists; refusing overwrite; use explicit resume"
            )
        parent = path.parent
        while not parent.exists():
            parent = parent.parent
        with tempfile.TemporaryFile(dir=parent):
            pass
    diagnostics = {}
    # CUDA always executes a real synthetic forward/backward before training.
    if synthetic_execution or device.type == "cuda":
        model = build_model(config["model_version"], config["model"]).to(device).train()
        x = torch.randn(3, 1, 1024, device=device)
        mask = (
            torch.arange(1024, device=device)[None, :]
            < torch.tensor([300, 700, 1024], device=device)[:, None]
        )
        loss = torch.nn.functional.cross_entropy(
            model(x, mask), torch.tensor([0, 1, 2], device=device)
        )
        loss.backward()
        if not torch.isfinite(loss) or any(
            p.grad is None or not torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise ValueError("Synthetic forward/backward failed")
        diagnostics["forward_backward_verified"] = True
        diagnostics["temporal_capability"] = temporal_capability(model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            diagnostics.update(
                compute_capability=list(torch.cuda.get_device_capability(device)),
                compiled_architectures=torch.cuda.get_arch_list(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                memory_bytes=list(torch.cuda.mem_get_info(device)),
            )
    return {
        "source": source,
        "noncanonical_development": config.get("noncanonical_development", False),
        "environment_contract": environment,
        "runtime": environment_metadata(device),
        "data_identity": train.identity,
        "data_compatibility": train.compatibility,
        "raw_files_checked": checked,
        "holdout_signals_accessed": False,
        "preprocessing": "reuse verified" if reuse else "pending training-only fit",
        "synthetic": diagnostics,
    }
