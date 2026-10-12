"""Strict research configuration without importing numerical or GPU libraries."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import re


PROFILES = {
    "cpu": {"project": ".", "venv": ".venv", "lock": "uv.lock"},
    "cuda128": {
        "project": "environments/cuda128",
        "venv": "environments/cuda128/.venv",
        "lock": "environments/cuda128/uv.lock",
    },
}

SECTIONS = {
    "experiment": {"name", "seed"},
    "environment": {"profile"},
    "runtime": {"device", "gpu_index", "cpu_threads"},
    "data": {"split_dir"},
    "preprocessing": {"method", "reuse"},
    "model": {"name", "stem_channels", "stem_kernel", "channels", "kernel", "dropout"},
    "optimizer": {"name", "learning_rate", "weight_decay"},
    "training": {
        "batch_size",
        "epochs",
        "patience",
        "cache_signals",
        "selection_metric",
    },
    "smoke": {
        "records_per_class",
        "batch_size",
        "max_steps",
        "max_seconds",
        "target_accuracy",
        "target_loss",
        "dropout",
    },
    "paths": {"output_root"},
    "execution": {"resume", "noncanonical_development"},
    "evaluation": {"profile", "device", "gpu_index", "cpu_threads"},
}


def read_document(path: Path, *, content: str | None = None) -> dict:
    """Reject duplicate YAML keys so a setting cannot silently replace another."""
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Configuration keys must be strings")
            if key in result:
                raise ValueError(f"Duplicate configuration key: {key}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping
    )
    try:
        value = yaml.load(
            path.read_text() if content is None else content, Loader=UniqueLoader
        )
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    return value


def relative_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Paths must be nonempty repository-relative strings")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Expected a repository-relative path: {value}")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the repository: {value}")
    return target


def integer(value, name, minimum=1, maximum=None):
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"Invalid integer for {name}")


def number(value, name, minimum=0, maximum=None, strict_minimum=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (strict_minimum and value == minimum)
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"Invalid numeric value for {name}")


def validate_experiment(document: dict, root: Path) -> None:
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 2
    ):
        raise ValueError("Unified YAML configuration requires schema_version: 2")
    if set(document) != {"schema_version", *SECTIONS}:
        raise ValueError("Missing or unknown top-level configuration sections")
    for name, keys in SECTIONS.items():
        section = document[name]
        if not isinstance(section, dict) or set(section) != keys:
            raise ValueError(
                f"Missing or unknown settings in {name}; expected {sorted(keys)}"
            )
    execution = document["execution"]
    if any(type(v) is not bool for v in execution.values()):
        raise ValueError("execution settings must be booleans")
    evaluation = document["evaluation"]
    if evaluation["profile"] not in PROFILES or evaluation["device"] not in (
        "cpu",
        "cuda",
        "auto",
    ):
        raise ValueError("Invalid evaluation runtime")
    if evaluation["profile"] == "cpu" and evaluation["device"] == "cuda":
        raise ValueError("CUDA evaluation requires cuda128 profile")
    integer(evaluation["gpu_index"], "evaluation.gpu_index", 0)
    integer(evaluation["cpu_threads"], "evaluation.cpu_threads")
    exp, runtime, smoke, model = (
        document[k] for k in ("experiment", "runtime", "smoke", "model")
    )
    if (
        not isinstance(exp["name"], str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", exp["name"]) is None
    ):
        raise ValueError(
            "experiment.name must be a short name using letters, digits, _ or -"
        )
    integer(exp["seed"], "experiment.seed", 0, 2**32 - 1)
    profile = document["environment"]["profile"]
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError("environment.profile must be cpu or cuda128")
    if runtime["device"] not in ("cpu", "cuda", "auto"):
        raise ValueError("runtime.device must be cpu, cuda or auto")
    if profile == "cpu" and runtime["device"] == "cuda":
        raise ValueError("CUDA execution requires environment.profile: cuda128")
    integer(runtime["gpu_index"], "runtime.gpu_index", 0)
    integer(runtime["cpu_threads"], "runtime.cpu_threads")
    relative_path(root, document["data"]["split_dir"])
    if document["preprocessing"]["method"] != "median-fixed-training-rms-v1":
        raise ValueError("Unsupported preprocessing method")
    reuse = document["preprocessing"]["reuse"]
    if reuse is not None:
        relative_path(root, reuse)
    output = document["paths"]["output_root"]
    relative_path(root, output)
    if Path(output).parts[0] not in ("outputs", "runs", "checkpoints"):
        raise ValueError(
            "paths.output_root must be inside outputs/, runs/ or checkpoints/"
        )
    if model["name"] not in ("rhythm-cnn-v1", "rhythm-dilated-v1"):
        raise ValueError("Unsupported model.name")
    if not isinstance(model["channels"], list) or len(model["channels"]) != 3:
        raise ValueError("model.channels requires three integer stage widths")
    for c in (model["stem_channels"], *model["channels"]):
        integer(c, "model channels", 2)
    for key in ("stem_kernel", "kernel"):
        integer(model[key], f"model.{key}")
        if model[key] % 2 != 1:
            raise ValueError("Convolution kernels must be odd")
    for section in (model, smoke):
        number(section["dropout"], "dropout", maximum=1)
        if section["dropout"] == 1:
            raise ValueError("dropout must be less than 1")
    optimizer, training = document["optimizer"], document["training"]
    if optimizer["name"] != "AdamW":
        raise ValueError("Only optimizer.name: AdamW is implemented")
    number(optimizer["learning_rate"], "optimizer.learning_rate", strict_minimum=True)
    number(optimizer["weight_decay"], "optimizer.weight_decay")
    for key in ("batch_size", "epochs", "patience"):
        integer(training[key], f"training.{key}")
    if type(training["cache_signals"]) is not bool:
        raise ValueError("training.cache_signals must be a boolean")
    if training["selection_metric"] != "macro_f1_four_classes":
        raise ValueError(
            "Only validation macro_f1_four_classes checkpoint selection is implemented"
        )
    integer(smoke["records_per_class"], "smoke.records_per_class")
    integer(smoke["batch_size"], "smoke.batch_size")
    integer(smoke["max_steps"], "smoke.max_steps", maximum=200)
    number(smoke["max_seconds"], "smoke.max_seconds", maximum=1200, strict_minimum=True)
    number(
        smoke["target_accuracy"],
        "smoke.target_accuracy",
        maximum=1,
        strict_minimum=True,
    )
    number(smoke["target_loss"], "smoke.target_loss")


def flatten_experiment(document: dict, *, smoke=False) -> dict:
    runtime, train, check = (document[k] for k in ("runtime", "training", "smoke"))
    model = {k: copy.deepcopy(v) for k, v in document["model"].items() if k != "name"}
    if smoke:
        model["dropout"] = check["dropout"]
    return {
        "schema_version": 1,
        "split_dir": document["data"]["split_dir"],
        "seed": document["experiment"]["seed"],
        "device": runtime["device"],
        "gpu_index": runtime["gpu_index"],
        "cpu_threads": runtime["cpu_threads"],
        "batch_size": train["batch_size"],
        "cache_signals": train["cache_signals"],
        "learning_rate": document["optimizer"]["learning_rate"],
        "weight_decay": document["optimizer"]["weight_decay"],
        "epochs": train["epochs"],
        "patience": train["patience"],
        "model": model,
        "model_version": document["model"]["name"],
        "noncanonical_development": document["execution"]["noncanonical_development"],
        "smoke_records_per_class": check["records_per_class"],
        "smoke_batch_size": check["batch_size"],
        "smoke_max_steps": check["max_steps"],
        "smoke_max_seconds": check["max_seconds"],
        "smoke_target_accuracy": check["target_accuracy"],
        "smoke_target_loss": check["target_loss"],
    }


def load_experiment(path: Path, root: Path, *, smoke=False) -> tuple[dict, dict]:
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Experiment YAML must be inside the repository")
    source = path.read_bytes()
    text = source.decode("utf-8")
    document = read_document(path, content=text)
    validate_experiment(document, root)
    flat = flatten_experiment(document, smoke=smoke)
    flat["_experiment"] = {
        "source_path": path.relative_to(root.resolve()).as_posix(),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_yaml": text,
        "document": document,
        "profile": document["environment"]["profile"],
        "mode": "smoke" if smoke else "baseline",
    }
    return flat, document


def output_path(document: dict, action: str) -> str:
    return (
        Path(document["paths"]["output_root"])
        / document["experiment"]["name"]
        / document["environment"]["profile"]
        / f"seed{document['experiment']['seed']}"
        / action
    ).as_posix()


def evaluation_config(config, document):
    """Resolve inference runtime without modifying saved training provenance."""
    value = copy.deepcopy(config)
    runtime = document["evaluation"]
    value.update({key: runtime[key] for key in ("device", "gpu_index", "cpu_threads")})
    value["_experiment"]["profile"] = runtime["profile"]
    return value
