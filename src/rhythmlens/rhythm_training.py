"""Bounded smoke checks and reproducible baseline training on frozen records."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time
import tomllib

import numpy as np
import torch
from torch.utils.data import DataLoader

from rhythmlens.data import LABELS, sha256, utc_now, write_json
from rhythmlens.configuration import (
    PROFILES,
    load_experiment,
    read_document,
    relative_path,
    validate_experiment,
)
from rhythmlens.preprocessing import PreprocessingState, fit_preprocessing
from rhythmlens.rhythm_data import (
    FrozenRecords,
    LengthBucketSampler,
    RhythmDataset,
    collate_records,
    project_path,
)
from rhythmlens.rhythm_model import (
    ModelConfig,
    RhythmCNN,
    build_model,
    temporal_capability,
)
from rhythmlens.lifecycle import (
    atomic_json,
    capture_rng,
    restore_rng,
    run_lifecycle,
    resume_signature,
)

DEFAULT_CONFIG = {
    "schema_version": 1,
    "split_dir": "splits/challenge2017-v3-seed42-v1",
    "seed": 42,
    "device": "cpu",
    "gpu_index": 0,
    "cpu_threads": 2,
    "batch_size": 16,
    "cache_signals": True,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "epochs": 50,
    "patience": 8,
    "model": ModelConfig().to_dict(),
    "smoke_records_per_class": 4,
    "smoke_batch_size": 4,
    "smoke_max_steps": 200,
    "smoke_max_seconds": 1200.0,
    "smoke_target_accuracy": 1.0,
    "smoke_target_loss": 0.1,
    "model_version": "rhythm-cnn-v1",
    "noncanonical_development": False,
    "_experiment": None,
}


def load_config(path: Path, *, smoke: bool = False) -> dict:
    value = read_document(path)
    if value.get("schema_version") == 2:
        value, _ = load_experiment(path, Path.cwd(), smoke=smoke)
    return validate_config(value)


def validate_config(value, *, validate_context=True):
    if not isinstance(value, dict) or set(value) - set(DEFAULT_CONFIG):
        raise ValueError(
            "Configuration must be a mapping with recognized baseline keys"
        )
    config = {**DEFAULT_CONFIG, **value}
    config["model"] = ModelConfig.from_dict(config["model"]).to_dict()
    if (
        type(config["schema_version"]) is not int
        or config["schema_version"] != 1
        or config["device"] not in ("cpu", "cuda", "auto")
    ):
        raise ValueError("Unsupported schema or device")
    for key in (
        "cpu_threads",
        "batch_size",
        "epochs",
        "patience",
        "smoke_records_per_class",
        "smoke_batch_size",
        "smoke_max_steps",
    ):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["gpu_index"]) is not int or config["gpu_index"] < 0:
        raise ValueError("gpu_index must be a nonnegative integer")
    if type(config["seed"]) is not int or config["seed"] < 0 or config["seed"] >= 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    for key in (
        "learning_rate",
        "weight_decay",
        "smoke_max_seconds",
        "smoke_target_loss",
        "smoke_target_accuracy",
    ):
        if (
            isinstance(config[key], bool)
            or not isinstance(config[key], (int, float))
            or not np.isfinite(config[key])
        ):
            raise ValueError(f"{key} must be finite")
    if (
        config["learning_rate"] <= 0
        or config["weight_decay"] < 0
        or config["smoke_target_loss"] < 0
    ):
        raise ValueError("Invalid learning rate, weight decay or smoke target loss")
    if not 0 < config["smoke_max_seconds"] <= 1200 or config["smoke_max_steps"] > 200:
        raise ValueError("Smoke checks are limited to 200 steps and 1200 seconds")
    if (
        not 0 < config["smoke_target_accuracy"] <= 1
        or type(config["cache_signals"]) is not bool
    ):
        raise ValueError("Invalid smoke accuracy target or cache setting")
    if not isinstance(config["split_dir"], str):
        raise ValueError("split_dir must be a relative path")
    if config["model_version"] not in ("rhythm-cnn-v1", "rhythm-dilated-v1"):
        raise ValueError("Unsupported model version")
    if type(config["noncanonical_development"]) is not bool:
        raise ValueError("noncanonical_development must be boolean")
    context = config["_experiment"]
    if context is not None and validate_context:
        if not isinstance(context, dict) or not isinstance(
            context.get("document"), dict
        ):
            raise ValueError("Invalid experiment provenance")
        validate_experiment(context["document"], Path.cwd())
    return config


def setup_runtime(config: dict) -> torch.device:
    context = config.get("_experiment")
    if context is not None:
        profile = context["profile"]
        source_root = Path(__file__).resolve().parents[2]
        lock = Path.cwd() / PROFILES[profile]["lock"]
        if not lock.exists():
            lock = source_root / PROFILES[profile]["lock"]
        versions = {
            p["version"]
            for p in tomllib.loads(lock.read_text())["package"]
            if p["name"] == "torch"
        }
        suffix = (
            "+cu128"
            if profile == "cuda128"
            else ("" if platform.system() == "Darwin" else "+cpu")
        )
        expected = {
            v for v in versions if (v.endswith(suffix) if suffix else "+" not in v)
        }
        pin = Path.cwd() / ".python-version"
        if not pin.exists():
            pin = source_root / ".python-version"
        python_pin = (
            pin.read_text().strip() if pin.exists() else platform.python_version()
        )
        if (
            str(torch.__version__) not in expected
            or platform.python_version() != python_pin
        ):
            raise ValueError(
                f"Wrong environment: profile {profile} requires Python {python_pin} and torch {sorted(expected)}; "
                "run python3 scripts/manage.py setup for this config"
            )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable in this environment")
    if device == "cuda":
        index = config.get("gpu_index", 0)
        if index >= torch.cuda.device_count():
            raise ValueError(f"GPU index {index} is unavailable")
        torch.cuda.set_device(index)
        return torch.device("cuda", index)
    return torch.device(device)


def create_output(root: Path, output: str | Path) -> Path:
    path = project_path(root, output)
    relative = path.relative_to(root.resolve())
    if len(relative.parts) < 2 or relative.parts[0] not in (
        "outputs",
        "runs",
        "checkpoints",
    ):
        raise ValueError(
            "Run outputs must be in a new subdirectory of outputs/, runs/ or checkpoints/"
        )
    if path.exists():
        raise ValueError(f"Output already exists; refusing overwrite: {relative}")
    path.mkdir(parents=True)
    return path


def code_metadata(root: Path, config: dict | None = None) -> dict:
    candidates = set()
    for folder in ("src", "scripts", "configs", "environments"):
        for parent, dirs, files in os.walk(root / folder):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            candidates.update(
                Path(parent) / name
                for name in files
                if Path(name).suffix in (".py", ".json", ".yaml", ".toml", ".lock")
            )
    candidates.update(
        root / name
        for name in ("pyproject.toml", "uv.lock", ".python-version")
        if (root / name).is_file()
    )
    context = config.get("_experiment") if config else None
    if context:
        # Invoked YAML has its own immutable snapshot and reviewed resume signature.
        candidates.discard(root / context["source_path"])
    hashes = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(candidates)}
    revision, dirty = None, None
    try:
        top = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if Path(top).resolve() == root.resolve():
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "-C", str(root), "status", "--porcelain"], text=True
                ).strip()
            )
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "revision": revision,
        "working_tree_dirty": dirty,
        "source_sha256": hashes,
        "source_bundle_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
    }


def environment_metadata(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
    }


def reproduction_metadata(root: Path, config: dict) -> dict:
    """Bind the active environment lock and exact installed versions to the run."""
    context = config.get("_experiment")
    locks = {"uv.lock": sha256(root / "uv.lock")} if (root / "uv.lock").exists() else {}
    profile = context["profile"] if context is not None else None
    if profile is not None:
        name = PROFILES[profile]["lock"]
        locks[name] = sha256(relative_path(root, name))
    return {
        "experiment": context,
        "dependency_locks_sha256": locks,
        "installed_versions": dict(
            sorted(
                (dist.metadata["Name"], dist.version)
                for dist in distributions()
                if dist.metadata.get("Name")
            )
        ),
    }


def save_checkpoint(
    path: Path,
    model: RhythmCNN,
    optimizer: torch.optim.Optimizer,
    state: PreprocessingState,
    config: dict,
    data_identity: dict,
    *,
    epoch: int,
    steps: int,
    kind: str,
    metadata: dict,
    continuation: dict | None = None,
    rng: dict | None = None,
) -> None:
    value = {
        "schema_version": 1,
        "task": "rhythm",
        "model_version": model.model_version,
        "temporal_capability": temporal_capability(model),
        "continuation": continuation,
        "kind": kind,
        "label_order": list(LABELS),
        "model_config": model.config.to_dict(),
        "model_state": {
            key: tensor.detach().cpu() for key, tensor in model.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "preprocessing": state.to_dict(),
        "config": config,
        "data_identity": data_identity,
        "data_identity_sha256": hashlib.sha256(
            json.dumps(data_identity, sort_keys=True).encode()
        ).hexdigest(),
        "preprocessing_identity_sha256": hashlib.sha256(
            json.dumps(state.to_dict(), sort_keys=True).encode()
        ).hexdigest(),
        "epoch": epoch,
        "steps": steps,
        "rng": rng if rng is not None else capture_rng(),
        "metadata": metadata,
    }
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as saved:
            os.fsync(saved.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
    expected_data_identity: dict | None = None,
) -> tuple[RhythmCNN, PreprocessingState, dict]:
    # Tensor/primitive-only format; never opt into unrestricted pickle loading.
    value = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema_version",
        "task",
        "model_version",
        "kind",
        "label_order",
        "model_config",
        "model_state",
        "optimizer_state",
        "preprocessing",
        "config",
        "data_identity",
        "epoch",
        "steps",
        "rng",
        "metadata",
    }
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError("Malformed checkpoint: missing required fields")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["task"] != "rhythm"
        or value["kind"] not in ("baseline", "smoke")
    ):
        raise ValueError("Unsupported rhythm checkpoint")
    if value["label_order"] != list(LABELS):
        raise ValueError("Checkpoint label order differs from N,A,O,~")
    for key in (
        "config",
        "data_identity",
        "metadata",
        "model_config",
        "rng",
        "optimizer_state",
        "model_state",
        "preprocessing",
    ):
        if not isinstance(value[key], dict):
            raise ValueError(f"Malformed checkpoint {key}")
    if any(type(value[key]) is not int or value[key] < 0 for key in ("epoch", "steps")):
        raise ValueError("Malformed checkpoint epoch/steps")
    if (
        expected_data_identity is not None
        and value["data_identity"] != expected_data_identity
    ):
        raise ValueError("Checkpoint/frozen data identity mismatch")
    identity = value["data_identity"]
    for field, target in (
        ("data_identity_sha256", identity),
        ("preprocessing_identity_sha256", value["preprocessing"]),
    ):
        if (
            field in value
            and hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()
            != value[field]
        ):
            raise ValueError(f"Checkpoint {field} checksum mismatch")
    if not identity or any(
        not isinstance(k, str) or not isinstance(v, str) or not v
        for k, v in identity.items()
        if v is not None
    ):
        raise ValueError("Malformed data identity")
    try:
        state = PreprocessingState.from_dict(value["preprocessing"])
        model = build_model(value["model_version"], value["model_config"])
        configured = value["config"]
        validate_config(configured, validate_context=False)
        if (
            ModelConfig.from_dict(configured["model"]) != model.config
            or configured.get("model_version", "rhythm-cnn-v1") != model.model_version
        ):
            raise ValueError("Checkpoint model/config agreement failed")
        if state.training_manifest_sha256 != identity["train_sha256"]:
            raise ValueError("Checkpoint preprocessing training-manifest mismatch")
        if "preprocessing_sha256" in value["metadata"]:
            encoded = json.dumps(state.to_dict(), indent=2) + "\n"
            if (
                hashlib.sha256(encoded.encode()).hexdigest()
                != value["metadata"]["preprocessing_sha256"]
            ):
                raise ValueError("Checkpoint preprocessing checksum mismatch")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed checkpoint metadata: {exc}") from exc
    continuation = value.get("continuation")
    if continuation is not None:
        if not isinstance(continuation, dict) or set(continuation) != {
            "history",
            "best",
            "best_epoch",
            "stale",
        }:
            raise ValueError("Malformed epoch continuation state")
        if (
            not isinstance(continuation["history"], list)
            or len(continuation["history"]) != value["epoch"]
            or any(
                not isinstance(entry, dict) or entry.get("epoch") != i + 1
                for i, entry in enumerate(continuation["history"])
            )
            or type(continuation["best_epoch"]) is not int
            or not 1 <= continuation["best_epoch"] <= value["epoch"]
            or type(continuation["stale"]) is not int
            or continuation["stale"] < 0
            or isinstance(continuation["best"], bool)
            or not isinstance(continuation["best"], (int, float))
            or not np.isfinite(continuation["best"])
        ):
            raise ValueError("Malformed epoch selection/history state")
        required_rng = {
            "python",
            "numpy",
            "torch_cpu",
            "cuda",
            "train_loader",
            "validation_loader",
            "sampler_epoch",
        }
        if not required_rng <= set(value["rng"]):
            raise ValueError("Missing resume RNG states")
        for key in ("torch_cpu", "train_loader", "validation_loader"):
            tensor = value["rng"][key]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.uint8
                or tensor.ndim != 1
            ):
                raise ValueError(f"Malformed {key} RNG state")
    model.load_state_dict(value["model_state"], strict=True)
    if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise ValueError("Checkpoint contains nonfinite model parameters")
    return model.to(device).eval(), state, value


def make_loader(
    dataset: RhythmDataset,
    config: dict,
    *,
    shuffle: bool,
    batch_size: int | None = None,
) -> DataLoader:
    sampler = LengthBucketSampler(
        dataset.lengths,
        batch_size or config["batch_size"],
        seed=config["seed"],
        shuffle=shuffle,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_records,
        num_workers=0,
        generator=torch.Generator().manual_seed(
            config["seed"] + (1001 if shuffle else 2001)
        ),
    )


def classification_metrics(truth: list[int], predicted: list[int]) -> dict:
    if (
        len(truth) != len(predicted)
        or not truth
        or any(type(x) is not int or x not in range(4) for x in truth + predicted)
    ):
        raise ValueError("Expected paired nonempty four-class labels")
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    support = confusion.sum(axis=1)
    outputs = confusion.sum(axis=0)
    precision = np.divide(
        confusion.diagonal(), outputs, out=np.zeros(4), where=outputs != 0
    )
    recall = np.divide(
        confusion.diagonal(), support, out=np.zeros(4), where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(4),
        where=precision + recall != 0,
    )
    return {
        "records": len(truth),
        "label_order": list(LABELS),
        "confusion_matrix": confusion.tolist(),
        "confusion_orientation": "rows=true, columns=predicted",
        "accuracy": float(confusion.trace() / len(truth)),
        "macro_f1_four_classes": float(f1.mean()),
        "zero_division_policy": "zero for undefined per-class precision/recall/F1",
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "af_recall_all_labeled_af": float(recall[1]) if support[1] else None,
        "automatic_af_sensitivity": float(recall[1]) if support[1] else None,
        "coverage": 1.0,
        "assessed_records": len(truth),
        "deferred_records": 0,
        "deferral_reasons": {},
        "af_correct": int(confusion[1, 1]),
        "af_denominator": int(support[1]),
        "deferral_policy": "not implemented; baseline classifier only",
        "calibration": "not fitted; probabilities are uncalibrated model softmax",
    }


def evaluate_batches(
    model: RhythmCNN, loader: DataLoader, device: torch.device
) -> tuple[dict, list[dict]]:
    model.eval()
    truth, predicted, rows = [], [], []
    loss_sum = 0.0
    with torch.inference_mode():
        for batch in loader:
            labels = batch["labels"].to(device)
            logits = model(batch["signals"].to(device), batch["mask"].to(device))
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite evaluation logits")
            loss_sum += float(
                torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
            )
            probabilities = logits.softmax(dim=1).cpu().numpy()
            guesses = logits.argmax(dim=1).cpu().tolist()
            truths = labels.cpu().tolist()
            truth.extend(truths)
            predicted.extend(guesses)
            for rid, actual, guess, probability, logit in zip(
                batch["record_ids"],
                truths,
                guesses,
                probabilities,
                logits.cpu().tolist(),
                strict=True,
            ):
                row = {
                    "record_id": rid,
                    "true_label": LABELS[actual],
                    "predicted_label": LABELS[guess],
                }
                row.update(
                    {
                        f"probability_{label}": float(probability[i])
                        for i, label in enumerate(LABELS)
                    }
                )
                row.update(
                    {
                        f"logit_{label}": float(logit[i])
                        for i, label in enumerate(LABELS)
                    }
                )
                rows.append(row)
    metrics = classification_metrics(truth, predicted)
    metrics["cross_entropy"] = loss_sum / len(truth)
    return metrics, sorted(rows, key=lambda row: row["record_id"])


def train_batch(
    model: RhythmCNN,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    device: torch.device,
) -> tuple[float, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    labels = batch["labels"].to(device)
    logits = model(batch["signals"].to(device), batch["mask"].to(device))
    loss = torch.nn.functional.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    loss.backward()
    if any(
        param.grad is not None and not torch.isfinite(param.grad).all()
        for param in model.parameters()
    ):
        raise ValueError("Nonfinite training gradients")
    optimizer.step()
    return float(loss.detach()), int((logits.argmax(dim=1) == labels).sum())


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_validation_artifacts(output: Path, metrics: dict, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    write_json(output / "validation-metrics.json", metrics)
    write_predictions(output / "validation-predictions.csv", rows)
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=range(4),
        yticks=range(4),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted class",
        ylabel="True class",
        title="Development validation; not final holdout",
    )
    for i in range(4):
        for j in range(4):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "validation-confusion.png", dpi=140)
    plt.close(fig)
    lines = [
        "# Baseline development validation",
        "",
        "This is development validation, not final holdout or clinical validation.",
        "Probabilities are uncalibrated. No deferral policy has been fitted.",
        "",
        f"Records: {metrics['records']}; four-class macro F1: {metrics['macro_f1_four_classes']:.6f}.",
        "",
        "| Class | Support | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in LABELS:
        item = metrics["per_class"][label]
        lines.append(
            f"| {label} | {item['support']} | {item['precision']:.6f} | {item['recall']:.6f} | {item['f1']:.6f} |"
        )
    lines += [
        "",
        "AF recall denominator includes every AF-labeled validation record.",
        "The naturally noisy class has no known underlying rhythm.",
        "See validation-metrics.json, validation-predictions.csv and validation-confusion.png.",
        "",
    ]
    (output / "validation-report.md").write_text("\n".join(lines))


def preprocessing_for_run(
    records: FrozenRecords, supplied: Path | None, *, deadline: float | None = None
) -> PreprocessingState:
    if supplied is None:
        return fit_preprocessing(records, deadline=deadline)
    state = PreprocessingState.from_dict(json.loads(supplied.read_text()))
    ids = sorted(row["record_id"] for row in records.rows)
    expected = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        state.training_manifest_sha256 != records.partition_sha256
        or state.fitted_records != len(ids)
        or state.fitted_record_ids_sha256 != expected
    ):
        raise ValueError(
            "Saved preprocessing must have been fitted on the entire frozen training partition"
        )
    return state


def fit_command(root: Path, config: dict, output: str | Path) -> dict:
    records = FrozenRecords(root, config["split_dir"], "train")
    destination = create_output(root, output)
    started = time.monotonic()
    state = fit_preprocessing(records)
    write_json(destination / "preprocessing.json", state.to_dict())
    write_json(destination / "config.json", config)
    if config.get("_experiment") is not None:
        (destination / "experiment.yaml").write_text(
            config["_experiment"]["source_yaml"]
        )
    report = {
        "kind": "training-only preprocessing fit",
        "created_at_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "preprocessing": state.to_dict(),
        "data_identity": records.identity,
        "reproduction": reproduction_metadata(root, config),
    }
    write_json(destination / "fit-report.json", report)
    return report


def _run_training(
    root: Path,
    config: dict,
    output: str | Path,
    *,
    smoke: bool,
    supplied_preprocessing: Path | None = None,
    resume: bool = False,
) -> dict:
    started = time.monotonic()
    device = setup_runtime(config)
    records = FrozenRecords(root, config["split_dir"], "train")
    context = config.get("_experiment")
    if supplied_preprocessing is None and context is not None:
        reuse = context["document"]["preprocessing"]["reuse"]
        if reuse is not None:
            supplied_preprocessing = relative_path(root, reuse)
    destination = project_path(root, output)
    if not resume:
        write_json(destination / "config.json", config)
    else:
        supplied_preprocessing = destination / "preprocessing.json"
    deadline = started + config["smoke_max_seconds"] if smoke else float("inf")
    state = preprocessing_for_run(
        records, supplied_preprocessing, deadline=deadline if smoke else None
    )
    if not resume:
        write_json(destination / "preprocessing.json", state.to_dict())
    fit_seconds = time.monotonic() - started
    indices = None
    if smoke:
        indices, rng, used_groups = [], random.Random(config["seed"]), set()
        for label in LABELS:
            candidates = [
                i for i, row in enumerate(records.rows) if row["label"] == label
            ]
            rng.shuffle(candidates)
            chosen = []
            for index in candidates:
                group = records.rows[index]["duplicate_group"]
                if group not in used_groups:
                    chosen.append(index)
                    used_groups.add(group)
                if len(chosen) == config["smoke_records_per_class"]:
                    break
            if len(chosen) != config["smoke_records_per_class"]:
                raise ValueError(
                    f"Not enough distinct training groups for smoke class {label}"
                )
            indices.extend(chosen)
    dataset = RhythmDataset(records, state, indices, cache=config["cache_signals"])
    loader = make_loader(
        dataset,
        config,
        shuffle=True,
        batch_size=config["smoke_batch_size"] if smoke else None,
    )
    evaluation_loader = (
        make_loader(
            dataset, config, shuffle=False, batch_size=config["smoke_batch_size"]
        )
        if smoke
        else None
    )
    validation_records = (
        None if smoke else FrozenRecords(root, config["split_dir"], "validation")
    )
    if validation_records is not None:
        validation_dataset = RhythmDataset(
            validation_records, state, cache=config["cache_signals"]
        )
        evaluation_loader = make_loader(validation_dataset, config, shuffle=False)
    model = build_model(config["model_version"], config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    metadata = {
        "created_at_utc": utc_now(),
        "configuration_sha256": sha256(destination / "config.json"),
        "preprocessing_sha256": sha256(destination / "preprocessing.json"),
        "code": code_metadata(root, config),
        "data_compatibility": records.compatibility,
        "temporal_capability": temporal_capability(model),
        "environment": environment_metadata(device),
        "reproduction": reproduction_metadata(root, config),
    }
    if config.get("_experiment") is not None and not resume:
        (destination / "experiment.yaml").write_text(
            config["_experiment"]["source_yaml"]
        )
    initial, _ = (
        evaluate_batches(model, evaluation_loader, device) if smoke else ({}, [])
    )
    history, best, stale, steps, processed = (
        [],
        float("inf") if smoke else -float("inf"),
        0,
        0,
        0,
    )
    batch_seconds, padded_slots, valid_slots = 0.0, 0, 0
    training_start = time.monotonic()
    stop_reason = "epoch_limit"
    epoch_limit = config["smoke_max_steps"] if smoke else config["epochs"]
    start_epoch, best_epoch = 0, 0
    if resume:
        restored_model, restored_state, checkpoint = load_checkpoint(
            destination / "last.pt",
            device=device,
            expected_data_identity=records.identity,
        )
        if (
            resume_signature(checkpoint["config"]) != resume_signature(config)
            or restored_state != state
        ):
            raise ValueError("Incompatible resume configuration/preprocessing")
        if checkpoint["metadata"]["environment"] != environment_metadata(device):
            raise ValueError("Incompatible resume runtime")
        if {
            k: v
            for k, v in checkpoint["metadata"]["reproduction"].items()
            if k != "experiment"
        } != {
            k: v
            for k, v in reproduction_metadata(root, config).items()
            if k != "experiment"
        }:
            raise ValueError("Incompatible resume environment/lock")
        if (
            checkpoint["metadata"]["code"]["source_bundle_sha256"]
            != code_metadata(root, config)["source_bundle_sha256"]
        ):
            raise ValueError("Incompatible resume source provenance")
        model.load_state_dict(restored_model.state_dict(), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        continuation = checkpoint.get("continuation")
        if not isinstance(continuation, dict):
            raise ValueError("Checkpoint has no epoch-boundary continuation state")
        history, best, best_epoch, stale = (
            continuation[k] for k in ("history", "best", "best_epoch", "stale")
        )
        start_epoch, steps = checkpoint["epoch"], checkpoint["steps"]
        metadata = checkpoint["metadata"]
        config = checkpoint["config"]
        restore_rng(checkpoint["rng"], loader, evaluation_loader)
    if not smoke and stale >= config["patience"]:
        start_epoch = epoch_limit
    for epoch in range(start_epoch, epoch_limit):
        loader.batch_sampler.set_epoch(epoch)
        total_loss, total_correct, epoch_records = 0.0, 0, 0
        for batch in loader:
            if smoke and (
                steps >= config["smoke_max_steps"] or time.monotonic() >= deadline
            ):
                stop_reason = (
                    "step_limit" if steps >= config["smoke_max_steps"] else "time_limit"
                )
                break
            tick = time.monotonic()
            loss, correct = train_batch(model, optimizer, batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_seconds += time.monotonic() - tick
            count = len(batch["record_ids"])
            steps += 1
            status_path = destination / "status.json"
            status = json.loads(status_path.read_text())
            status.update(optimizer_steps=steps, updated_at_utc=utc_now())
            atomic_json(status_path, status)
            processed += count
            epoch_records += count
            total_loss += loss * count
            total_correct += correct
            valid_slots += int(batch["mask"].sum())
            padded_slots += batch["mask"].numel()
        metrics, prediction_rows = evaluate_batches(model, evaluation_loader, device)
        entry = {
            "epoch": epoch + 1,
            "steps": steps,
            "training_cross_entropy": total_loss / epoch_records
            if epoch_records
            else None,
            "training_accuracy_with_dropout": total_correct / epoch_records
            if epoch_records
            else None,
            "selection_partition": "training_smoke_subset" if smoke else "validation",
            "selection_metrics": metrics,
        }
        history.append(entry)
        write_json(destination / "history.json", history)
        score = metrics["cross_entropy"] if smoke else metrics["macro_f1_four_classes"]
        improved = score < best if smoke else score > best
        if improved:
            best, stale, best_epoch = score, 0, epoch + 1
            save_checkpoint(
                destination / "best.pt",
                model,
                optimizer,
                state,
                config,
                records.identity,
                epoch=epoch + 1,
                steps=steps,
                kind="smoke" if smoke else "baseline",
                metadata=metadata,
                continuation={
                    "history": history,
                    "best": best,
                    "best_epoch": best_epoch,
                    "stale": stale,
                },
                rng=capture_rng(loader, evaluation_loader),
            )
        else:
            stale += 1
        save_checkpoint(
            destination / "last.pt",
            model,
            optimizer,
            state,
            config,
            records.identity,
            epoch=epoch + 1,
            steps=steps,
            kind="smoke" if smoke else "baseline",
            metadata=metadata,
            continuation={
                "history": history,
                "best": best,
                "best_epoch": best_epoch,
                "stale": stale,
            },
            rng=capture_rng(loader, evaluation_loader),
        )
        status_path = destination / "status.json"
        status = json.loads(status_path.read_text())
        status.update(
            last_completed_epoch=epoch + 1,
            optimizer_steps=steps,
            last_usable_checkpoint="last.pt",
            updated_at_utc=utc_now(),
        )
        atomic_json(status_path, status)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "steps": steps,
                    "partition": entry["selection_partition"],
                    "loss": metrics["cross_entropy"],
                    "macro_f1": metrics["macro_f1_four_classes"],
                }
            ),
            flush=True,
        )
        if smoke:
            if (
                metrics["accuracy"] >= config["smoke_target_accuracy"]
                and metrics["cross_entropy"] <= config["smoke_target_loss"]
            ):
                stop_reason = "training_subset_target_met"
                break
            if steps >= config["smoke_max_steps"] or time.monotonic() >= deadline:
                stop_reason = (
                    "step_limit" if steps >= config["smoke_max_steps"] else "time_limit"
                )
                break
        elif stale >= config["patience"]:
            stop_reason = "early_stopping_validation_macro_f1"
            break
    best_model, loaded_state, payload = load_checkpoint(
        destination / "best.pt", device=device, expected_data_identity=records.identity
    )
    final, final_rows = evaluate_batches(best_model, evaluation_loader, device)
    # Same saved weights and shared preprocessing must survive an independent reload.
    same_model_metrics, original_rows = evaluate_batches(
        model, evaluation_loader, device
    )
    reloaded_last, _, _ = load_checkpoint(
        destination / "last.pt", device=device, expected_data_identity=records.identity
    )
    reloaded_metrics, reloaded_rows = evaluate_batches(
        reloaded_last, evaluation_loader, device
    )
    if (
        not np.isclose(
            same_model_metrics["cross_entropy"],
            reloaded_metrics["cross_entropy"],
            atol=1e-7,
            rtol=1e-6,
        )
        or loaded_state != state
    ):
        raise ValueError("Checkpoint reload verification failed")
    original_logits = [
        [row[f"logit_{label}"] for label in LABELS] for row in original_rows
    ]
    reloaded_logits = [
        [row[f"logit_{label}"] for label in LABELS] for row in reloaded_rows
    ]
    if [row["record_id"] for row in original_rows] != [
        row["record_id"] for row in reloaded_rows
    ] or not np.allclose(original_logits, reloaded_logits, atol=1e-7, rtol=1e-6):
        raise ValueError("Checkpoint reload logits differ")
    elapsed_training = time.monotonic() - training_start
    compute_rate = processed / batch_seconds if batch_seconds else None
    report = {
        "kind": "training-only wiring check; not generalization evidence"
        if smoke
        else "baseline development training",
        "created_at_utc": utc_now(),
        "stop_reason": stop_reason,
        "steps": steps,
        "best_epoch": payload["epoch"],
        "training_records": len(dataset),
        "class_counts": dict(
            Counter(records.rows[index]["label"] for index in dataset.indices)
        ),
        "smoke_record_ids": [
            records.rows[index]["record_id"] for index in dataset.indices
        ]
        if smoke
        else None,
        "initial_training_subset_metrics": initial if smoke else None,
        "best_selection_metrics": final,
        "overfit_target_met": bool(
            final["accuracy"] >= config["smoke_target_accuracy"]
            and final["cross_entropy"] <= config["smoke_target_loss"]
        )
        if smoke
        else None,
        "holdout_signals_accessed": False,
        "validation_signals_accessed": not smoke,
        "preprocessing_seconds_including_manifest_checks": fit_seconds,
        "training_and_evaluation_seconds": elapsed_training,
        "total_seconds": time.monotonic() - started,
        "optimization_step_seconds": batch_seconds,
        "processed_training_record_visits": processed,
        "optimization_records_per_second": compute_rate,
        "padding_fraction": 1 - valid_slots / padded_slots if padded_slots else None,
        "rough_training_compute_seconds_per_full_epoch": len(records.rows)
        / compute_rate
        if compute_rate
        else None,
        "runtime_estimate_limitations": "Smoke subset durations may not represent all records; compute estimate excludes decoding, validation, checkpointing and cold-start overhead. Wall budget is checked between batches and can overrun by one batch plus evaluation/checkpointing.",
        "checkpoint_reload_verified": True,
        "model_parameters": sum(param.numel() for param in model.parameters()),
        "preprocessing": state.to_dict(),
        "data_identity": records.identity,
        "config": config,
        "metadata": metadata,
    }
    write_json(destination / "run-report.json", report)
    if smoke:
        write_predictions(destination / "smoke-training-predictions.csv", final_rows)
    else:
        write_validation_artifacts(destination, final, final_rows)
    return report


def evaluate_command(
    root: Path,
    checkpoint: Path,
    output: str | Path,
    *,
    runtime_config: dict | None = None,
) -> dict:
    model, state, payload = load_checkpoint(checkpoint)
    if payload["kind"] != "baseline":
        raise ValueError(
            "Smoke checkpoints cannot be presented as validation baselines"
        )
    config = payload["config"]
    runtime = (
        runtime_config
        if runtime_config is not None
        else {**config, "device": "cpu", "gpu_index": 0, "_experiment": None}
    )
    device = setup_runtime(runtime)
    if runtime.get("_experiment") is not None:
        from rhythmlens.preflight import lock_environment

        lock_environment(root, runtime["_experiment"]["profile"])
    records = FrozenRecords(root, config["split_dir"], "validation")
    if payload["data_identity"] != records.identity:
        raise ValueError("Checkpoint/frozen data identity mismatch")
    destination = create_output(root, output)
    dataset = RhythmDataset(records, state, cache=config["cache_signals"])
    metrics, rows = evaluate_batches(
        model.to(device), make_loader(dataset, config, shuffle=False), device
    )
    write_validation_artifacts(destination, metrics, rows)
    report = {
        "partition": "validation",
        "checkpoint_sha256": sha256(checkpoint),
        "data_identity": records.identity,
        "metrics": metrics,
        "holdout_signals_accessed": False,
        "environment": environment_metadata(device),
        "training_provenance": payload["metadata"],
        "training_config": config,
        "evaluation_runtime_config": runtime,
        "reproduction": reproduction_metadata(root, runtime),
    }
    write_json(destination / "evaluation-report.json", report)
    return report


def run_training(
    root, config, output, *, smoke, supplied_preprocessing=None, resume=False
):
    from rhythmlens.preflight import preflight

    config = validate_config(config)
    checked = preflight(
        root,
        config,
        output=output,
        smoke=smoke,
        resume=resume,
        synthetic_execution=False,
    )
    destination = project_path(root, output)
    if resume:
        # Validate before updating status or any artifact.
        _, _, payload = load_checkpoint(destination / "last.pt")
        if (
            resume_signature(payload["config"]) != resume_signature(config)
            or payload.get("continuation") is None
        ):
            raise ValueError("Incompatible resume configuration or checkpoint")
    else:
        destination = create_output(root, output)
    with run_lifecycle(destination, config, resume=resume):
        atomic_json(
            destination / ("resume-preflight.json" if resume else "preflight.json"),
            checked,
        )
        return _run_training(
            root,
            config,
            output,
            smoke=smoke,
            supplied_preprocessing=supplied_preprocessing,
            resume=resume,
        )
