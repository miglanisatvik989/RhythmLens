"""Disabled-by-default Phase B: explicit CUDA training and CPU paired evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

from . import noise_phase_a as phase_a
from .calibration import (
    local,
    read_json,
    require,
    require_committed_source,
    sha,
    write_new,
)
from .noise import apply_plan, array_hash
from .noise_phase_a import atomic, canonical, save_array
from .noise_phase_a_analysis import SEEDS, conditions
from .noise_phase_b_analysis import comparison, paired_bootstrap
from .noise_phase_b_training import CUDATrainer, Progress, train_seed
from .nstdb import now
from .preprocessing import PreprocessingState, transform_physical

POLICY_SHA256 = "71bb8347193a0f9e67bdf34686c0c48e42c5fd4525b99a83068c2088d1401b82"
OUTPUT = "outputs/noise-robustness/noise-robustness-v1/phase-b"


def load_config(path):
    c = yaml.load(Path(path).read_text(), Loader=phase_a.UniqueLoader)
    require(
        isinstance(c, dict)
        and c.get("protocol_schema") == "rhythmlens-noise-phase-b-v1",
        "Unsupported Phase B schema",
    )
    require(
        type(c.get("execution_authorized")) is bool,
        "Literal boolean authorization required",
    )
    require(
        canonical({k: v for k, v in c.items() if k != "execution_authorized"})
        == POLICY_SHA256,
        "Phase B execution contract changed; versioned review required",
    )
    return c


def inventory(folder, *, exclude=()):
    result = {}
    for p in sorted(folder.rglob("*")):
        require(not p.is_symlink(), "Symlink in evidence inventory")
        if p.is_file() and str(p.relative_to(folder)) not in exclude:
            result[str(p.relative_to(folder))] = sha(p)
    return result


def verify_index(folder, index_name, digest, *, exclude=()):
    require(sha(folder / index_name) == digest, "Artifact index byte mismatch")
    expected = read_json(folder / index_name)
    require(
        inventory(folder, exclude=(*exclude, index_name)) == expected,
        "Artifact inventory/bytes mismatch",
    )


def verify_baseline(root, c, rows, *, arrays=False):
    """Full indexed evidence, not just cached logits. Never launches Phase A."""
    folder = local(root, c["baseline"]["root"])
    for name, v in c["baseline"]["identities"].items():
        path = local(folder, name)
        require(
            path.stat().st_size == v["bytes"] and sha(path) == v["sha256"],
            f"Baseline reuse identity mismatch: {name}",
        )
    status = read_json(folder / "status.json")
    require(status["state"] == "completed", "Baseline reuse requires completed Phase A")
    verify_index(
        folder,
        "artifacts-sha256.json",
        status["artifacts_index_sha256"],
        exclude=("status.json",),
    )
    registry = read_json(folder / "registry.json")
    require(
        registry["source_revision"] == c["baseline"]["source_revision"],
        "Baseline execution revision mismatch",
    )
    original = phase_a.load_config(local(root, c["baseline"]["config"]))
    require(
        canonical(registry["configuration"]) == canonical(original),
        "Baseline numerical/input contract mismatch",
    )
    require(
        read_json(folder / "plans.json")["rows"] == rows,
        "Baseline row/label/group mismatch",
    )
    phase_a.check_identities(root, registry["input_byte_sha256"])
    # Historical documentation/test edits do not affect inference compatibility;
    # any change to an existing baseline source module requires explicit review.
    phase_a.check_identities(
        root,
        {
            n: h
            for n, h in registry["committed_files_sha256"].items()
            if n.startswith("src/rhythmlens/")
        },
    )
    if arrays:
        return phase_a.sealed_predictions(folder, rows)
    return read_json(folder / "plans.json")


def reviewed_pools(root, c, role):
    from .nstdb import load_pools

    pools = load_pools(
        local(root, c["noise"]["raw"]),
        read_json(local(root, c["noise"]["provenance"])),
        role,
    )
    descriptors = read_json(local(root, c["noise"]["roles"]))["roles"]
    require(set(pools) == {"bw", "ma", "em"}, "All noise records required")
    for n, pool in pools.items():
        require(
            canonical(pool.metadata())
            == canonical(
                next(r for r in descriptors if r["record"] == n and r["role"] == role)
            ),
            "Reconstructed role/pool mismatch",
        )
    return pools


def compatible_plans(rows, pools, baseline):
    plans = phase_a.freeze_plans(rows, pools)
    expected = dict(
        rows=rows,
        conditions=conditions(),
        plans=[
            dict(
                index=i,
                repeat=r,
                noise_release="nstdb-1.0.0",
                role_descriptor=pools[n].metadata(),
                **p.metadata(),
            )
            for (i, n, r), p in plans.items()
        ],
    )
    require(
        canonical(expected) == canonical(baseline),
        "Phase A baseline reuse incompatible with frozen stress maps; no rerun",
    )
    return plans


def completed_training(folder):
    status = read_json(folder / "status.json")
    require(status["state"] == "completed", "All augmented training must be completed")
    verify_index(
        folder,
        "artifacts-sha256.json",
        status["artifacts_index_sha256"],
        exclude=("status.json",),
    )
    for seed in SEEDS:
        child = folder / f"seed{seed}"
        report = read_json(child / "run-report.json")
        require(
            read_json(child / "status.json")["state"] == "completed"
            and report["seed"] == seed
            and report["selection"] == "strict_clean_validation_macro_f1"
            and report["best_checkpoint_sha256"] == sha(child / "best.pt"),
            "Augmented seed/selection/checkpoint mismatch",
        )
    return read_json(folder / "registry.json")


class EvaluationBackend(phase_a.Backend):
    def __init__(self, root, c, records, training):
        super().__init__(root, c, records)
        self.training = training

    def state_and_model(self, seed):
        if seed not in self.loaded:
            from .rhythm_training import load_checkpoint
            from .configuration import load_experiment

            folder = self.training / f"seed{seed}"
            report = read_json(folder / "run-report.json")
            model, state, payload = load_checkpoint(
                folder / "best.pt", expected_data_identity=self.records.identity
            )
            config, _ = load_experiment(
                local(self.root, self.c["training_configs"][seed]), self.root
            )
            require(
                payload["kind"] == "baseline"
                and payload["metadata"].get("phase_b") is True
                and payload["config"] == config
                and payload["epoch"] == report["best_epoch"]
                and payload["model_version"] == "rhythm-dilated-v1"
                and payload["metadata"]["source_revision"]
                == read_json(self.training / "registry.json")["source_revision"],
                "Augmented candidate seed/config/source mismatch",
            )
            original = next(m for m in self.c["models"] if m["seed"] == seed)
            require(
                state.to_dict()
                == read_json(local(self.root, original["preprocessing"]))
                and state.to_dict() == read_json(folder / "preprocessing.json"),
                "Augmented preprocessing differs from original",
            )
            self.loaded[seed] = state, model.eval()
        return self.loaded[seed]


def evaluate_augmented(destination, rows, backend, plans, progress, verify_inputs):
    """Produce 165 complete arrays and evidence. No scoring occurs here."""
    from .rhythm_data import LengthBucketSampler

    batches = list(
        LengthBucketSampler([int(r["samples"]) for r in rows], 16, shuffle=False)
    )
    accounting = [
        dict(
            seed=s,
            condition=n,
            record_id=r["record_id"],
            label=r["label"],
            duplicate_group=r["duplicate_group"],
            status="not_started",
            reason="pending",
        )
        for s in SEEDS
        for n, *_ in conditions()
        for r in rows
    ]
    index = {(r["seed"], r["condition"], r["record_id"]): r for r in accounting}
    atomic(destination / "accounting.json", accounting)
    active, details = [], []
    predictions = {}
    try:
        for c, (name, record, snr, repeat) in enumerate(conditions()):
            for seed in SEEDS:
                stage = "clean_evaluation" if c == 0 else "stress_evaluation"
                progress(stage, seed=seed, completed=0, total=len(batches))
                folder = destination / f"seed{seed}" / name
                folder.mkdir(parents=True, exist_ok=False)
                state, model = backend.state_and_model(seed)
                require(
                    isinstance(state, PreprocessingState),
                    "Explicit original state required",
                )
                z = np.empty((len(rows), 4), dtype=np.float32)
                for number, indices in enumerate(batches):
                    active = [index[seed, name, rows[i]["record_id"]] for i in indices]
                    for item in active:
                        item.update(status="started", reason="batch_in_progress")
                    with (destination / "events.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps(
                                dict(
                                    seed=seed,
                                    condition=name,
                                    indices=indices,
                                    event="started",
                                    utc=now(),
                                )
                            )
                            + "\n"
                        )
                    items, details = [], []
                    tick = time.monotonic()
                    for i in indices:
                        row = rows[i]
                        x = backend.physical(i)
                        require(
                            np.asarray(x).shape == (int(row["samples"]),),
                            "Source length mismatch",
                        )
                        value, meta = (
                            (x, dict(corruption_applied=False, reason="clean_control"))
                            if record is None
                            else apply_plan(
                                x,
                                plans[i, record, repeat],
                                backend.pools,
                                partition="validation",
                                snr_db=snr,
                            )
                        )
                        transformed = transform_physical(
                            value, state, sampling_rate_hz=300, units="mV"
                        )
                        items.append((row, transformed))
                        details.append(
                            dict(
                                record_id=row["record_id"],
                                original_physical_sha256=array_hash(x),
                                corrupted_physical_sha256=array_hash(value),
                                post_transform_sha256=hashlib.sha256(
                                    np.asarray(transformed, dtype="<f4").tobytes()
                                ).hexdigest(),
                                corruption=meta,
                            )
                        )
                    logits = backend.predict(model, items)
                    require(
                        logits.dtype == np.float32
                        and logits.shape == (len(indices), 4)
                        and np.isfinite(logits).all(),
                        "Invalid augmented logits",
                    )
                    z[indices] = logits
                    save_array(folder / f"batch{number}.npy", logits)
                    write_new(
                        folder / f"batch{number}.json",
                        dict(
                            indices=indices,
                            cases=details,
                            elapsed_seconds=time.monotonic() - tick,
                        ),
                    )
                    for item in active:
                        item.update(status="completed", reason=None)
                    active = []
                    progress(stage, seed=seed, completed=number + 1, total=len(batches))
                save_array(folder / "logits.npy", z)
                name_path = str((folder / "logits.npy").relative_to(destination))
                predictions[name_path] = sha(folder / "logits.npy")
        atomic(destination / "accounting.json", accounting)
        verify_inputs()
        evidence = inventory(destination, exclude=("status.json", "progress.json"))
        write_new(
            destination / "seal.json", dict(predictions=predictions, evidence=evidence)
        )
    except BaseException as exc:
        for item in active:
            item.update(status="technical_failure", reason=str(exc))
        for item in accounting:
            if item["status"] == "not_started":
                item["reason"] = "blocked_after_failure: " + str(exc)
        atomic(destination / "accounting.json", accounting)
        atomic(
            destination / "failure-context.json",
            dict(active=active, available_corruption_metadata=details, error=str(exc)),
        )
        raise


def sealed_augmented(destination, rows):
    seal = read_json(destination / "seal.json")
    expected = {f"seed{s}/{n}/logits.npy" for s in SEEDS for n, *_ in conditions()}
    require(
        set(seal["predictions"]) == expected,
        "All augmented candidates/conditions required",
    )
    batches = (len(rows) + 15) // 16
    expected_evidence = {
        "registry.json",
        "plans.json",
        "accounting.json",
        "events.jsonl",
        *expected,
        *(
            f"seed{s}/{n}/batch{b}.{ext}"
            for s in SEEDS
            for n, *_ in conditions()
            for b in range(batches)
            for ext in ("npy", "json")
        ),
    }
    require(
        set(seal["evidence"]) == expected_evidence,
        "Incomplete augmented evidence inventory",
    )
    require(
        all(seal["evidence"][n] == h for n, h in seal["predictions"].items()),
        "Prediction/evidence seal mismatch",
    )
    phase_a.check_identities(destination, {**seal["evidence"], **seal["predictions"]})
    ledger = read_json(destination / "accounting.json")
    keys = {
        (s, n, r["record_id"]) for s in SEEDS for n, *_ in conditions() for r in rows
    }
    require(
        len(ledger) == len(keys)
        and {(r["seed"], r["condition"], r["record_id"]) for r in ledger} == keys
        and all(r["status"] == "completed" for r in ledger),
        "Incomplete augmented accounting",
    )
    row_map = {r["record_id"]: r for r in rows}
    require(
        all(
            r["label"] == row_map[r["record_id"]]["label"]
            and r["duplicate_group"] == row_map[r["record_id"]]["duplicate_group"]
            for r in ledger
        ),
        "Accounting labels/groups mismatch",
    )
    require(
        read_json(destination / "plans.json")["rows"] == rows, "Scoring rows changed"
    )
    values = []
    for seed in SEEDS:
        by_condition = []
        for name, *_ in conditions():
            z = np.load(
                destination / f"seed{seed}/{name}/logits.npy", allow_pickle=False
            )
            require(
                z.dtype == np.float32
                and z.shape == (len(rows), 4)
                and np.isfinite(z).all(),
                "Invalid sealed augmented array",
            )
            by_condition.append(z)
        values.append(by_condition)
    return np.asarray(values)


def verify_pairing(destination, baseline, rows):
    """Both arms must see the same physical and transformed inputs, not just keys."""
    batches = (len(rows) + 15) // 16
    for seed in SEEDS:
        for name, *_ in conditions():
            for number in range(batches):
                path = f"seed{seed}/{name}/batch{number}.json"
                old, new = read_json(baseline / path), read_json(destination / path)
                require(
                    old["indices"] == new["indices"]
                    and canonical(old["cases"]) == canonical(new["cases"]),
                    "Original/augmented physical inputs or corruption maps differ",
                )


def comparative_seal(
    destination, rows, baseline_root, baseline, training, verify_inputs
):
    """Verify both arms, freeze cross-root byte identities, then permit scoring."""
    verify_inputs()
    completed_training(training)
    original = phase_a.sealed_predictions(baseline_root, rows)
    augmented = sealed_augmented(destination, rows)
    verify_pairing(destination, baseline_root, rows)
    value = dict(
        original_status_sha256=sha(baseline_root / "status.json"),
        original_index_sha256=sha(baseline_root / "artifacts-sha256.json"),
        original_seal_sha256=sha(baseline_root / "seal.json"),
        training_status_sha256=sha(training / "status.json"),
        training_index_sha256=sha(training / "artifacts-sha256.json"),
        augmented_seal_sha256=sha(destination / "seal.json"),
        baseline_binding=baseline,
        seed_pairing=list(SEEDS),
        arms=["original", "augmented"],
        conditions=conditions(),
        rows_sha256=canonical(rows),
    )
    write_new(destination / "comparison-seal.json", value)
    return np.stack((original, augmented)), value


def metadata_preflight(root, c):
    records, rows = phase_a.metadata_preflight(root, c)
    from .rhythm_data import FrozenRecords

    train = FrozenRecords(root, c["split_dir"], "train")
    require(
        train.identity == c["data_identity"] and len(train.rows) == 5969,
        "Training eligibility changed",
    )
    require(
        all(
            r["split"] == "train"
            and r["subject_id"] == "unknown"
            and float(r["sampling_rate_hz"]) == 300
            and r["physical_units"] == "mV"
            for r in train.rows
        ),
        "Training grouping/units changed",
    )
    return train, records, rows


def run_training_stage(destination, registry, work):
    """Exclusive creation and no retries. Work may be a generated fixture callback."""
    destination.mkdir(parents=True, exist_ok=False)
    atomic(destination / "status.json", dict(state="running", started_utc=now()))
    try:
        write_new(destination / "registry.json", registry)
        work()
        write_new(
            destination / "artifacts-sha256.json",
            inventory(destination, exclude=("status.json",)),
        )
        atomic(
            destination / "status.json",
            dict(
                state="completed",
                completed_utc=now(),
                artifacts_index_sha256=sha(destination / "artifacts-sha256.json"),
            ),
        )
    except BaseException as exc:
        atomic(
            destination / "status.json",
            dict(
                state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error=str(exc),
                updated_utc=now(),
            ),
        )
        raise


def execute(root, config_path, stage, authorized=False):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    c = load_config(config_path)
    require(
        c["execution_authorized"] is True,
        "execution_authorized must be true before protected I/O",
    )
    require(
        authorized is True,
        "Explicit --authorize-execution required before protected I/O",
    )
    require(stage in ("train", "evaluate"), "Explicit train/evaluate stage required")
    require_committed_source(root, config_path)
    require(
        phase_a.runtime_versions() == c["environments"][stage],
        "Pinned stage environment mismatch",
    )
    # Check the lexical path before local() resolves symlinks (including dangling ones).
    require(not (root / c["output_root"]).is_symlink(), "Output symlink collision")
    destination = local(root, c["output_root"])
    require(not destination.is_symlink(), "Output symlink collision")
    if stage == "train":
        require(not destination.exists(), "Output collision; no retry/resume")
    else:
        require(
            read_json(destination / "status.json")["state"] == "training_completed",
            "Evaluation requires exactly one completed training stage",
        )
        require(
            not (destination / "evaluation").exists(), "Evaluation collision; no retry"
        )
    train, validation, rows = metadata_preflight(root, c)
    identities = {
        **phase_a.protected_identities(root, c, train),
        **phase_a.protected_identities(root, c, validation),
    }
    committed = {
        name: sha(root / name)
        for name in subprocess.check_output(
            ["git", "ls-files"], cwd=root, text=True
        ).splitlines()
    }
    registry = dict(
        configuration=c,
        config_byte_sha256=sha(config_path),
        config_canonical_sha256=canonical(c),
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        committed_files_sha256=committed,
        input_byte_sha256=identities,
        environment=phase_a.runtime_versions(),
        stage=stage,
        created_utc=now(),
        invocation=sys.argv,
    )

    def verify_inputs():
        phase_a.check_identities(root, {**committed, **identities})
        verify_baseline(root, c, rows)

    if stage == "train":
        destination.mkdir(parents=True, exist_ok=False)
    # A stage claim is exclusive even when two processes race to evaluate.
    with (destination / f"{stage}-claim.json").open("x") as stream:
        json.dump(dict(stage=stage, utc=now()), stream)
    atomic(
        destination / "status.json", dict(state="running_" + stage, started_utc=now())
    )
    progress = Progress(destination)
    try:
        if stage == "train":
            folder = destination / "training"

            def work():
                verify_inputs()
                pools = reviewed_pools(root, c, "augmentation-train")
                stress = reviewed_pools(root, c, "stress")
                write_new(folder / "rows.json", train.rows)
                write_new(
                    folder / "noise-pools.json",
                    {
                        "augmentation-train": {
                            n: p.metadata() for n, p in pools.items()
                        },
                        "stress": {n: p.metadata() for n, p in stress.items()},
                    },
                )
                compatible_plans(rows, stress, verify_baseline(root, c, rows))
                for seed in SEEDS:
                    verify_inputs()
                    trainer = CUDATrainer(
                        root, c, seed, train, validation, pools, registry
                    )
                    train_seed(folder / f"seed{seed}", seed, trainer, progress)
                    del trainer
                verify_inputs()

            run_training_stage(folder, registry, work)
            completed_training(folder)
            atomic(
                destination / "status.json",
                dict(state="training_completed", completed_utc=now()),
            )
        else:
            old_registry = completed_training(destination / "training")
            require(
                old_registry["source_revision"] == registry["source_revision"]
                and old_registry["config_byte_sha256"] == registry["config_byte_sha256"]
                and old_registry["committed_files_sha256"] == committed,
                "Training/evaluation source or execution configuration changed",
            )
            folder = destination / "evaluation"
            folder.mkdir(exist_ok=False)
            atomic(folder / "status.json", dict(state="running", started_utc=now()))
            write_new(folder / "registry.json", registry)
            verify_inputs()
            import torch

            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True)
            backend = EvaluationBackend(root, c, validation, destination / "training")
            baseline_plans = verify_baseline(root, c, rows)
            plans = compatible_plans(rows, backend.pools, baseline_plans)
            write_new(folder / "plans.json", baseline_plans)
            evaluate_augmented(folder, rows, backend, plans, progress, verify_inputs)
            progress("sealing")
            predictions, seal = comparative_seal(
                folder,
                rows,
                local(root, c["baseline"]["root"]),
                c["baseline"],
                destination / "training",
                verify_inputs,
            )
            seal_hash = sha(folder / "comparison-seal.json")
            progress("bootstrap_scoring")
            result = comparison(rows, predictions)
            result["flat_cases"] = {
                "original": phase_a.flat_findings(
                    local(root, c["baseline"]["root"]), rows, predictions[0]
                ),
                "augmented": phase_a.flat_findings(folder, rows, predictions[1]),
            }
            draws, stats, uncertainty = paired_bootstrap(rows, predictions)
            save_array(folder / "bootstrap-draws.npy", draws)
            save_array(folder / "bootstrap-statistics.npy", stats)
            write_new(folder / "bootstrap.json", uncertainty)
            verify_inputs()
            completed_training(destination / "training")
            phase_a.sealed_predictions(local(root, c["baseline"]["root"]), rows)
            sealed_augmented(folder, rows)
            require(
                sha(folder / "comparison-seal.json") == seal_hash,
                "Comparison seal changed",
            )
            require(
                sha(destination / "training/status.json")
                == seal["training_status_sha256"]
                and sha(folder / "seal.json") == seal["augmented_seal_sha256"],
                "Comparison inputs changed after scoring",
            )
            write_new(folder / "metrics.json", result)
            write_new(
                folder / "artifacts-sha256.json",
                inventory(folder, exclude=("status.json",)),
            )
            atomic(
                folder / "status.json",
                dict(
                    state="completed",
                    completed_utc=now(),
                    artifacts_index_sha256=sha(folder / "artifacts-sha256.json"),
                ),
            )
            atomic(
                destination / "status.json",
                dict(
                    state="completed",
                    completed_utc=now(),
                    evaluation_status_sha256=sha(folder / "status.json"),
                ),
            )
            progress("completion")
    except BaseException as exc:
        status = dict(
            state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            stage=stage,
            error=str(exc),
            updated_utc=now(),
        )
        if stage == "evaluate" and (destination / "evaluation").is_dir():
            atomic(destination / "evaluation/status.json", status)
        atomic(destination / "status.json", status)
        progress("failure")
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stage", choices=("train", "evaluate"), required=True)
    p.add_argument("--authorize-execution", action="store_true")
    args = p.parse_args()
    execute(args.root, args.config, args.stage, args.authorize_execution)


if __name__ == "__main__":
    main()
