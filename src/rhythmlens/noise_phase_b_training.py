"""Phase B training primitives; production construction is authorization-gated."""

import hashlib
import time

import numpy as np

from .calibration import LABELS, read_json, require, sha, write_new
from .noise import apply_plan, array_hash, training_plan
from .noise_phase_a import atomic
from .preprocessing import PreprocessingState, transform_physical


def epoch_plans(rows, pools, seed, epoch):
    require(
        rows and len({r["record_id"] for r in rows}) == len(rows), "Invalid train IDs"
    )
    require(all(r["split"] == "train" for r in rows), "Training partition required")
    return [
        training_plan(
            r["record_id"], seed, epoch, int(r["samples"]), pools, partition="train"
        )
        for r in rows
    ]


class AugmentedDataset:
    """Cache only original physical inputs; transform a newly planned view per visit."""

    def __init__(self, records, state, pools, seed):
        require(records.partition == "train", "Training-only augmentation")
        require(
            state.training_manifest_sha256 == records.identity["train_sha256"],
            "Original preprocessing training binding changed",
        )
        self.records, self.state, self.pools, self.seed = records, state, pools, seed
        self.lengths = [int(r["samples"]) for r in records.rows]
        self.cache, self.details, self.plans = {}, [], None

    def __len__(self):
        return len(self.lengths)

    def start_epoch(self, epoch):
        self.plans = epoch_plans(self.records.rows, self.pools, self.seed, epoch)
        self.details = []
        return [
            dict(
                record_id=r["record_id"],
                label=r["label"],
                duplicate_group=r["duplicate_group"],
                split=r["split"],
                **p.metadata(),
            )
            for r, p in zip(self.records.rows, self.plans, strict=True)
        ]

    def __getitem__(self, index):
        import torch

        require(self.plans is not None, "Freeze epoch plans before training")
        row = self.records.rows[index]
        if index not in self.cache:
            x = np.asarray(self.records.load_physical(index), dtype=np.float64).copy()
            require(
                x.shape == (self.lengths[index],) and np.isfinite(x).all(),
                "Invalid original training input",
            )
            x.flags.writeable = False
            self.cache[index] = x
        x = self.cache[index]
        value, meta = apply_plan(x, self.plans[index], self.pools, partition="train")
        processed = transform_physical(
            value, self.state, sampling_rate_hz=300, units="mV"
        )
        self.details.append(
            dict(
                record_id=row["record_id"],
                index=index,
                original_physical_sha256=array_hash(x),
                corrupted_physical_sha256=array_hash(value),
                post_transform_sha256=hashlib.sha256(
                    np.asarray(processed, dtype="<f4").tobytes()
                ).hexdigest(),
                corruption=meta,
            )
        )
        return dict(
            record_id=row["record_id"],
            signal=torch.from_numpy(processed.copy()),
            label=LABELS.index(row["label"]),
        )


def selection_update(best, stale, metric):
    """Original strict improvement rule: ties increment patience; clean macro F1 only."""
    require(np.isfinite(metric), "Nonfinite clean-validation metric")
    return (metric, 0, True) if metric > best else (best, stale + 1, False)


def train_seed(folder, seed, trainer, progress):
    """Generated adapters may exercise lifecycle/selection without any real model I/O."""
    folder.mkdir(parents=True, exist_ok=False)
    history, best, stale, steps, selected_epoch = [], -float("inf"), 0, 0, None
    atomic(folder / "status.json", dict(state="running", seed=seed))
    try:
        trainer.initialize(folder)
        write_new(folder / "config.json", trainer.config)
        write_new(folder / "preprocessing.json", trainer.state.to_dict())
        for epoch in range(1, trainer.config["epochs"] + 1):
            epoch_root = folder / f"epoch{epoch}"
            epoch_root.mkdir(exist_ok=False)
            plans = trainer.dataset.start_epoch(epoch)
            write_new(
                epoch_root / "plans.json", dict(epoch=epoch, seed=seed, plans=plans)
            )
            trainer.loader.batch_sampler.set_epoch(epoch - 1)
            seen = []
            progress(
                "training",
                seed=seed,
                epoch=epoch,
                completed=0,
                total=len(trainer.loader),
            )
            for number, batch in enumerate(trainer.loader):
                loss, correct = trainer.step(batch)
                require(np.isfinite(loss), "Nonfinite optimizer loss")
                steps += 1
                details = trainer.dataset.details
                require(
                    [d["record_id"] for d in details] == batch["record_ids"],
                    "Training batch evidence mismatch",
                )
                write_new(
                    epoch_root / f"batch{number}.json",
                    dict(
                        record_ids=batch["record_ids"],
                        cases=details,
                        loss=loss,
                        correct=correct,
                        optimizer_step=steps,
                    ),
                )
                seen.extend(batch["record_ids"])
                trainer.dataset.details = []
                progress(
                    "training",
                    seed=seed,
                    epoch=epoch,
                    completed=number + 1,
                    total=len(trainer.loader),
                )
            expected = [r["record_id"] for r in trainer.dataset.records.rows]
            require(
                len(seen) == len(expected) and set(seen) == set(expected),
                "Incomplete/duplicate training epoch accounting",
            )
            progress(
                "clean_validation", seed=seed, epoch=epoch, completed=0, total=None
            )
            metrics = trainer.validate()
            best, stale, improved = selection_update(
                best, stale, metrics["macro_f1_four_classes"]
            )
            history.append(
                dict(
                    epoch=epoch,
                    selection_partition="clean_validation",
                    selection_metrics=metrics,
                    optimizer_steps=steps,
                )
            )
            atomic(folder / "history.json", history)
            if improved:
                selected_epoch = epoch
                trainer.save(
                    folder / "best.pt",
                    epoch,
                    steps,
                    history,
                    best,
                    stale,
                    selected_epoch,
                )
            trainer.save(
                folder / "last.pt", epoch, steps, history, best, stale, selected_epoch
            )
            atomic(
                folder / "status.json",
                dict(
                    state="running",
                    seed=seed,
                    last_completed_epoch=epoch,
                    optimizer_steps=steps,
                    last_usable_checkpoint="last.pt",
                ),
            )
            if stale >= trainer.config["patience"]:
                break
        require(selected_epoch is not None, "No selected augmented checkpoint")
        report = dict(
            seed=seed,
            experiment="temporal-noise-v1",
            best_epoch=selected_epoch,
            selection="strict_clean_validation_macro_f1",
            steps=steps,
            completed_epochs=len(history),
            resume_supported=False,
            best_checkpoint_sha256=sha(folder / "best.pt"),
            config_sha256=sha(folder / "config.json"),
            preprocessing_sha256=sha(folder / "preprocessing.json"),
        )
        write_new(folder / "run-report.json", report)
        atomic(folder / "status.json", dict(state="completed", **report))
        return report
    except BaseException as exc:
        atomic(
            folder / "failure-context.json",
            dict(
                error=str(exc),
                optimizer_steps=steps,
                available_corruption_metadata=getattr(trainer.dataset, "details", []),
            ),
        )
        atomic(
            folder / "status.json",
            dict(
                state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                seed=seed,
                error=str(exc),
                optimizer_steps=steps,
            ),
        )
        raise


class CUDATrainer:
    """Original model/optimizer/bucketing/selection, with physical augmentation only."""

    def __init__(self, root, c, seed, records, validation, pools, registry):
        self.root, self.contract, self.seed = root, c, seed
        self.records, self.validation, self.pools = records, validation, pools
        self.registry = registry
        self.dataset = None

    def initialize(self, folder):
        import torch
        from . import rhythm_training as rt
        from .configuration import load_experiment
        from .rhythm_data import RhythmDataset
        from .rhythm_model import build_model

        config, document = load_experiment(
            self.root / self.contract["training_configs"][self.seed], self.root
        )
        self.config = rt.validate_config(config)
        require(
            document["experiment"]["name"] == "temporal-v1"
            and config["seed"] == self.seed
            and config["device"] == "cuda"
            and document["training"]["selection_metric"] == "macro_f1_four_classes"
            and config["cache_signals"] is True,
            "Original CUDA profile changed",
        )
        m = next(m for m in self.contract["models"] if m["seed"] == self.seed)
        original = read_json(self.root / m["resolved_config"])
        # Historical _experiment provenance differs; every numerical flat setting is fixed.
        require(
            {k: v for k, v in config.items() if k != "_experiment"}
            == {k: v for k, v in original.items() if k != "_experiment"},
            "Original resolved numerical training configuration changed",
        )
        self.state = PreprocessingState.from_dict(
            read_json(self.root / m["preprocessing"])
        )
        require(self.state.scale_mv == 0.18008390945457742, "Original scale changed")
        self.device = rt.setup_runtime(config)
        self.dataset = AugmentedDataset(self.records, self.state, self.pools, self.seed)
        self.loader = rt.make_loader(self.dataset, config, shuffle=True)
        self.validation_loader = rt.make_loader(
            RhythmDataset(self.validation, self.state, cache=True),
            config,
            shuffle=False,
        )
        self.model = build_model(config["model_version"], config["model"]).to(
            self.device
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        self.metadata = dict(
            phase_b=True,
            experiment="temporal-noise-v1",
            source_revision=self.registry["source_revision"],
            code=rt.code_metadata(self.root, config),
            environment=rt.environment_metadata(self.device),
            reproduction=rt.reproduction_metadata(self.root, config),
            original_preprocessing_sha256=m["preprocessing_sha256"],
        )

    def step(self, batch):
        import torch
        from .rhythm_training import train_batch

        result = train_batch(self.model, self.optimizer, batch, self.device)
        torch.cuda.synchronize(self.device)
        return result

    def validate(self):
        from .rhythm_training import evaluate_batches

        metrics, _ = evaluate_batches(self.model, self.validation_loader, self.device)
        return metrics

    def save(self, path, epoch, steps, history, best, stale, best_epoch):
        from .rhythm_training import save_checkpoint
        from .lifecycle import capture_rng

        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.state,
            self.config,
            self.records.identity,
            epoch=epoch,
            steps=steps,
            kind="baseline",
            metadata=self.metadata,
            continuation=dict(
                history=history, best=best, stale=stale, best_epoch=best_epoch
            ),
            rng=capture_rng(self.loader, self.validation_loader),
        )


class Progress:
    """Measured epoch-local ETA only; validation/scoring remain indeterminate."""

    def __init__(self, destination):
        self.destination, self.started = destination, time.monotonic()
        self.key, self.tick = None, self.started

    def __call__(self, stage, *, seed=None, epoch=None, completed=0, total=None):
        key = stage, seed, epoch
        if key != self.key:
            self.key, self.tick = key, time.monotonic()
        elapsed = time.monotonic() - self.tick
        rate = completed / elapsed if completed and elapsed > 0 else None
        eta = (total - completed) / rate if total and rate else None
        value = dict(
            stage=stage,
            seed=seed,
            epoch=epoch,
            completed=completed,
            total=total,
            elapsed_seconds=time.monotonic() - self.started,
            stage_elapsed_seconds=elapsed,
            batches_per_second=rate,
            remaining_stage_seconds=eta,
        )
        atomic(self.destination / "progress.json", value)
        bar = (
            "[" + ("#" * int(20 * completed / total)).ljust(20) + "]"
            if total
            else "[...]"
        )
        print(
            f"{stage} seed={seed} epoch={epoch} {bar} {completed}/{total or '?'} "
            f"elapsed={value['elapsed_seconds']:.1f}s ETA={eta if eta is not None else '?'}s",
            flush=True,
        )
