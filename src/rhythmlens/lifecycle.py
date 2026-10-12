"""Atomic run records and safe tensor/primitive epoch-boundary RNG snapshots."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import uuid

import numpy as np
import torch

from rhythmlens.data import utc_now


def atomic_json(path: Path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def configuration_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def capture_rng(train_loader=None, validation_loader=None):
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
        "torch_cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_loader": train_loader.generator.get_state() if train_loader else None,
        "validation_loader": validation_loader.generator.get_state()
        if validation_loader
        else None,
        "sampler_epoch": train_loader.batch_sampler.epoch if train_loader else None,
    }


def restore_rng(value, train_loader, validation_loader):
    random.setstate(value["python"])
    n = value["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(value["torch_cpu"])
    if value["cuda"]:
        torch.cuda.set_rng_state_all(value["cuda"])
    train_loader.generator.set_state(value["train_loader"])
    validation_loader.generator.set_state(value["validation_loader"])
    train_loader.batch_sampler.set_epoch(value["sampler_epoch"])


@contextmanager
def run_lifecycle(destination, config, *, resume=False):
    path = destination / "status.json"
    previous = json.loads(path.read_text()) if resume else {}
    record = {
        **previous,
        "state": "running",
        "started_at_utc": previous.get("started_at_utc", utc_now()),
        "updated_at_utc": utc_now(),
        "invocation": sys.argv,
        "run_identity": previous.get("run_identity", uuid.uuid4().hex),
        "configuration_sha256": previous.get(
            "configuration_sha256", configuration_hash(config)
        ),
        "invocation_configuration_sha256": configuration_hash(config),
        "last_completed_epoch": previous.get("last_completed_epoch", 0),
        "optimizer_steps": previous.get("optimizer_steps", 0),
        "last_usable_checkpoint": previous.get("last_usable_checkpoint"),
        "details": None,
    }
    atomic_json(path, record)
    try:
        yield record
    except BaseException as exc:
        # Read epoch updates written by the training loop.
        record = json.loads(path.read_text())
        record.update(
            state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            updated_at_utc=utc_now(),
            details={"type": type(exc).__name__, "message": str(exc)},
        )
        atomic_json(path, record)
        raise
    else:
        record = json.loads(path.read_text())
        record.update(state="completed", updated_at_utc=utc_now())
        atomic_json(path, record)


def resume_signature(config):
    value = json.loads(json.dumps(config))
    context = value.get("_experiment")
    if context:
        for key in ("source_yaml", "source_sha256", "source_path"):
            context.pop(key, None)
        context["document"]["execution"]["resume"] = False
    return value
