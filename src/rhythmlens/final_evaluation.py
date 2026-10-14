"""One-time qualified internal evaluation of three frozen rhythm-only models.

Preflight reads metadata and opaque checkpoint hashes only. Real signal access
is confined to execute(), behind committed YAML and explicit runtime authority.
Development FrozenRecords remains holdout-locked.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import yaml

from rhythmlens.calibration import (
    LABELS,
    csv_rows,
    encoded,
    ledger,
    local,
    metrics,
    read_json,
    rate,
    require,
    require_committed_source,
    sha,
    write_new,
)

POLICY_SHA256 = "2d20095a1db344318e1bbc983fa05ef7ca356f6f80a4595b2acaa179479eca6d"
SEEDS = (42, 43, 44)


def load_config(path):
    c = yaml.safe_load(Path(path).read_text())
    require(
        type(c.get("execution_authorized")) is bool,
        "Authorization must be a literal boolean",
    )
    policy = {k: v for k, v in c.items() if k != "execution_authorized"}
    require(
        hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
        == POLICY_SHA256,
        "Frozen internal evaluation policy changed",
    )
    return c


def metadata_preflight(root, c):
    """No checkpoint deserialization, signal reads, or prediction reads."""
    from rhythmlens.rhythm_data import FrozenRecords

    for name, digest in c["protected_sha256"].items():
        require(sha(local(root, name)) == digest, f"Protected input changed: {name}")
    require(
        sha(local(root, c["report"])) == c["report_sha256"], "Multiseed report changed"
    )
    train = FrozenRecords(root, c["split_dir"], "train")
    require(train.identity == c["data_identity"], "Original data identity changed")
    split = local(root, c["split_dir"])
    require(
        sha(split / "holdout.csv") == c["holdout_sha256"], "Holdout manifest changed"
    )
    manifest = csv_rows(split / "holdout.csv")
    require(len(manifest) == c["eligible_records"], "Eligibility changed")
    require(
        [sum(r["label"] == label for r in manifest) for label in LABELS]
        == c["class_supports"],
        "Class supports changed",
    )
    # Existing unknown subjects must stay unknown. Newly verified relations need
    # a protocol amendment, rather than silently retaining recording intervals.
    source = csv_rows(root / "data/manifests/challenge2017-v3/records.csv")
    require(
        all(r["subject_id"] == "unknown" for r in source),
        "New subject linkage requires review",
    )
    require(all(r["split"] == "holdout" for r in manifest), "Wrong holdout tags")
    groups = defaultdict(set)
    for r in manifest:
        groups[r["duplicate_group"]].add(r["label"])
    require(all(len(v) == 1 for v in groups.values()), "Conflicting group labels")
    report = read_json(local(root, c["report"]))
    runs = {
        r["seed"]: r for r in report["runs"] if r["architecture"] == "rhythm-dilated-v1"
    }
    require([m["seed"] for m in c["models"]] == list(SEEDS), "All fixed seeds required")
    inputs = {}
    for m in c["models"]:
        r = runs[m["seed"]]
        for key in (
            "checkpoint_sha256",
            "best_epoch",
            "preprocessing_sha256",
            "source_revision",
            "source_bundle_sha256",
        ):
            require(m[key] == r[key], f"Seed provenance changed: {key}")
        for key in (
            "checkpoint",
            "preprocessing",
            "config",
            "resolved_config",
            "training_report",
        ):
            path = local(root, m[key])
            require(sha(path) == m[key + "_sha256"], f"Input changed: {m[key]}")
            inputs[m[key]] = m[key + "_sha256"]
        require(
            m["config_sha256"] == r["yaml_sha256"]
            and m["resolved_config_sha256"] == r["resolved_config_sha256"]
            and m["training_report_sha256"] == r["run_report_sha256"],
            "Training provenance mismatch",
        )
    identities = {**c["protected_sha256"], **inputs, c["report"]: c["report_sha256"]}
    for name in (c["protocol_document"], c["integrity_audit"]):
        identities[name] = sha(local(root, name))
    for name in (
        "frozen.json",
        "train.csv",
        "validation.csv",
        "holdout.csv",
        "config.json",
        "summary.json",
        "README.md",
    ):
        identities[str(Path(c["split_dir"]) / name)] = sha(split / name)
    for name in ("records.csv", "audit.json", "provenance.json"):
        path = f"data/manifests/challenge2017-v3/{name}"
        identities[path] = sha(root / path)
    return dict(
        inputs_sha256=identities,
        data_identity=train.identity,
        eligible=len(manifest),
        class_supports=c["class_supports"],
        holdout_signals_accessed=False,
        checkpoint_deserialized=False,
    ), manifest


def runtime_preflight(root, c):
    from rhythmlens.preflight import lock_environment
    import torch

    require(
        platform.python_version() == c["runtime"]["python"], "Python version mismatch"
    )
    require(torch.__version__ == c["runtime"]["torch"], "Torch version mismatch")
    require(np.__version__ == c["runtime"]["numpy"], "NumPy version mismatch")
    contract = lock_environment(root, "cpu")  # Offline checks; never installs.
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    return dict(
        platform=platform.platform(),
        python=sys.version,
        interpreter=sys.executable,
        installed_versions={
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        lock_contract=contract,
        cpu_threads=torch.get_num_threads(),
        deterministic_algorithms=True,
        inference_dtype="float32",
        scoring_dtype="float64",
    )


def authorize(root, config_path, c, authorized):
    require(
        c["execution_authorized"] is True,
        "Real evaluation requires execution_authorized: true",
    )
    require(authorized is True, "Real evaluation requires --authorize-evaluation")
    require_committed_source(root, config_path)


def raw_metrics(rows, z):
    cases = ledger(
        [dict(record_id=r["record_id"], true_label=r["label"]) for r in rows],
        z,
        1.0,
        0.0,
    )
    result = metrics(cases)
    # Baseline prespecifies zero-filled class F1, including a flag for zero support.
    result["macro_f1"] = result["legacy_zero_fill_macro_f1"]
    result["macro_f1_undefined_reason"] = None
    result["zero_support_classes"] = [
        k for k, v in result["per_class"].items() if v["support"] + v["predicted"] == 0
    ]
    result["accuracy"] = result["accepted_accuracy"]
    result["other_policy_deferrals"] = 0
    require(
        result["complete"] and result["accepted"] == len(rows),
        "Incomplete raw assessment",
    )
    return cases, result


def primary_from_confusion(cm):
    cm = np.asarray(cm)
    support = cm.sum(axis=1)
    denominator = support + cm.sum(axis=0)
    f1 = np.divide(
        2 * np.diag(cm), denominator, out=np.zeros(4), where=denominator != 0
    )
    return [float(f1.mean()), float(cm[1, 1] / support[1]) if support[1] else np.nan]


def bootstrap(rows, logits, *, replicates=10000, seed=20261014):
    """Class-stratified duplicate groups, shared draws across all fixed seeds."""
    require(
        len(logits) == 3 and replicates > 0,
        "Three seeds and positive replicates required",
    )
    require(len({r["record_id"] for r in rows}) == len(rows), "Repeated bootstrap ID")
    y = np.array([LABELS.index(r["label"]) for r in rows])
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        require(bool(r["duplicate_group"]), "Missing duplicate group")
        groups[r["duplicate_group"]].append(i)
    require(
        all(len(set(y[ix])) == 1 for ix in groups.values()),
        "Conflicting bootstrap group labels",
    )
    strata = [
        [groups[g] for g in sorted(groups) if y[groups[g][0]] == k] for k in range(4)
    ]
    guesses = []
    for z in logits:
        require(
            z.shape == (len(rows), 4) and np.isfinite(z).all(),
            "Invalid bootstrap predictions",
        )
        guesses.append(z.argmax(axis=1))
    rng = np.random.Generator(np.random.PCG64(seed))
    stats = np.full((replicates, 4, 2), np.nan)
    for b in range(replicates):
        indices = [
            i
            for stratum in strata
            if stratum
            for j in rng.integers(0, len(stratum), size=len(stratum))
            for i in stratum[j]
        ]
        for s, guess in enumerate(guesses):
            cm = np.bincount(4 * y[indices] + guess[indices], minlength=16).reshape(
                4, 4
            )
            stats[b, s] = primary_from_confusion(cm)
        stats[b, 3] = stats[b, :3].mean(axis=0)
    require(np.isfinite(stats[:, :, 0]).all(), "Bootstrap numerical failure")
    has_af = bool(np.any(y == 1))
    require(
        not has_af or np.isfinite(stats[:, :, 1]).all(),
        "AF bootstrap numerical failure",
    )
    intervals = {}
    for s, key in enumerate(("42", "43", "44", "equal_weight_mean")):
        intervals[key] = {}
        for j, name in enumerate(("macro_f1", "automatic_af_sensitivity")):
            intervals[key][name] = dict(
                interval=np.quantile(
                    stats[:, s, j], [0.025, 0.975], method="linear"
                ).tolist()
                if j == 0 or has_af
                else None,
                undefined_reason=None if j == 0 or has_af else "no_AF_labels",
            )
    return stats, dict(
        replicates=replicates,
        seed=seed,
        generator="PCG64",
        axes=[
            "replicate",
            "seed42_seed43_seed44_equal_mean",
            "macro_f1_AF_sensitivity",
        ],
        groups={g: [rows[i]["record_id"] for i in groups[g]] for g in sorted(groups)},
        intervals=intervals,
        limitation="Conditions on observed class mix and fixed checkpoints; unknown patients, source shifts and training sampling unaccounted for",
    )


def verify_signal_files(root, rows):
    """Opaque hashes only, after exclusive run creation, before any prediction."""
    source = {
        r["record_id"]: r
        for r in csv_rows(root / "data/manifests/challenge2017-v3/records.csv")
    }
    audit = read_json(root / "data/manifests/challenge2017-v3/audit.json")
    for r in rows:
        for key in ("mat_path", "header_path"):
            name = source[r["record_id"]][key]
            require(
                sha(local(root, name)) == audit["inputs_sha256"].get(name),
                f"Missing or changed audited signal input: {name}",
            )


def verify_registry(root, registry):
    for name, digest in registry["inputs_sha256"].items():
        require(
            sha(local(root, name)) == digest,
            f"Frozen input changed during execution: {name}",
        )
    for name, digest in registry["committed_files_sha256"].items():
        require(
            sha(local(root, name)) == digest,
            f"Committed source changed during execution: {name}",
        )


def predict_seed(root, c, m, rows, on_batch):
    """Called only after authorization/preflight and exclusive lifecycle creation."""
    import torch
    from rhythmlens.rhythm_data import FrozenRecords, RhythmDataset, collate_records
    from rhythmlens.rhythm_training import load_checkpoint

    # Reuse the complete split/source audit without relaxing the development loader.
    records = FrozenRecords(root, c["split_dir"], "train")
    source = {
        r["record_id"]: r
        for r in csv_rows(root / "data/manifests/challenge2017-v3/records.csv")
    }
    records.rows = [{**source[r["record_id"]], "split": "holdout"} for r in rows]
    records.partition = "holdout"
    records.partition_sha256 = c["holdout_sha256"]
    model, state, checkpoint = load_checkpoint(
        local(root, m["checkpoint"]), expected_data_identity=records.identity
    )
    require(
        checkpoint["kind"] == "baseline"
        and checkpoint["epoch"] == m["best_epoch"]
        and checkpoint["model_version"] == "rhythm-dilated-v1"
        and checkpoint["config"]["seed"] == m["seed"],
        "Checkpoint candidate mismatch",
    )
    require(
        state.to_dict() == read_json(local(root, m["preprocessing"])),
        "Saved preprocessing mismatch",
    )
    require(
        checkpoint["config"] == read_json(local(root, m["resolved_config"])),
        "Resolved checkpoint config mismatch",
    )
    code = checkpoint["metadata"]["code"]
    require(
        code["revision"] == m["source_revision"]
        and code["source_bundle_sha256"] == m["source_bundle_sha256"],
        "Checkpoint training source mismatch",
    )
    model.eval()
    dataset = RhythmDataset(records, state, cache=False)
    result = []
    for start in range(0, len(rows), 16):
        indices = range(start, min(start + 16, len(rows)))
        batch_rows = rows[start : start + 16]
        on_batch(batch_rows, "started")
        batch = collate_records([dataset[i] for i in indices])
        require(
            batch["record_ids"] == [r["record_id"] for r in batch_rows],
            "Batch ID mismatch",
        )
        with torch.inference_mode():
            z = model(batch["signals"], batch["mask"]).cpu().numpy().copy()
        require(
            z.dtype == np.float32
            and z.shape == (len(batch_rows), 4)
            and np.isfinite(z).all(),
            "Invalid inference logits",
        )
        result.append(z)
        on_batch(batch_rows, "completed")
    return np.concatenate(result)


def score_sealed(destination, rows, seals):
    require(
        set(seals) == {"42", "43", "44"},
        "All three predictions must be sealed before scoring",
    )
    logits = []
    for seed in SEEDS:
        path = destination / f"seed{seed}/logits.npy"
        require(sha(path) == seals[str(seed)], "Sealed predictions changed")
        with path.open("rb") as stream:
            z = np.load(stream, allow_pickle=False)
        require(
            z.dtype == np.float32
            and z.shape == (len(rows), 4)
            and np.isfinite(z).all(),
            "Incomplete sealed predictions",
        )
        logits.append(z)
    results = {}
    for seed, z in zip(SEEDS, logits):
        cases, result = raw_metrics(rows, z)
        write_new(destination / f"seed{seed}/cases.json", cases)
        write_new(destination / f"seed{seed}/metrics.json", result)
        results[str(seed)] = result
    stats, uncertainty = bootstrap(rows, logits)
    with (destination / "bootstrap-statistics.npy").open("xb") as stream:
        np.save(stream, stats, allow_pickle=False)
    write_new(destination / "bootstrap.json", uncertainty)

    # Aggregate the entire fixed metric suite, retaining undefined denominators.
    def summarize(values):
        if all(isinstance(v, dict) for v in values):
            return {k: summarize([v[k] for v in values]) for k in values[0]}
        if all(v is None or type(v) in (float, int) for v in values):
            defined = [v for v in values if v is not None]
            return dict(
                n=len(defined),
                mean=float(np.mean(defined)) if defined else None,
                sample_sd=float(np.std(defined, ddof=1)) if len(defined) > 1 else None,
            )
        return None

    return dict(
        scope="qualified_internal_frozen_rhythm_only",
        per_seed=results,
        equal_weight_summary=summarize(list(results.values())),
        uncertainty=uncertainty["intervals"],
        quality_aware_validation=False,
        holdout_consumed_for_future_development=True,
    )


def execute(root, config_path, c, authorized=False):
    authorize(root, config_path, c, authorized)  # Before inputs or output creation.
    destination = local(root, c["output_root"])
    require(
        not destination.exists(), "Output collision: no overwrite or automatic retry"
    )
    registry, rows = metadata_preflight(root, c)
    rows = sorted(rows, key=lambda r: r["record_id"])
    runtime = runtime_preflight(root, c)
    # Freeze every committed byte identity, including the executor, protocol and config.
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    registry["committed_files_sha256"] = {
        p: sha(root / p) for p in tracked if p and (root / p).is_file()
    }
    registry["revision"] = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    )
    registry["config_sha256"] = sha(config_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # Exclusive: racing invocations cannot share a run.
    from rhythmlens.lifecycle import atomic_json
    from rhythmlens.data import utc_now

    status = dict(
        state="running",
        stage="prediction",
        started_at_utc=utc_now(),
        prediction_seals={},
        holdout_signals_accessed=False,
    )
    accounting = {
        str(s): [
            dict(
                record_id=r["record_id"],
                true_label=r["label"],
                disposition="technical_failure",
                reason="not_attempted",
                confidence_deferred=0,
                quality_deferred=0,
            )
            for r in rows
        ]
        for s in SEEDS
    }

    def save_status():
        atomic_json(
            destination / "status.json", {**status, "updated_at_utc": utc_now()}
        )

    def save_accounting():
        atomic_json(destination / "execution-accounting.json", accounting)
        summaries = {}
        for seed, cases in accounting.items():
            af = [r for r in cases if r["true_label"] == "A"]
            accepted = sum(r["disposition"] == "accepted" for r in cases)
            af_accepted = sum(r["disposition"] == "accepted" for r in af)
            summaries[seed] = dict(
                eligible=len(cases),
                labeled_af=len(af),
                accepted=accepted,
                technical_failure_or_unresolved=len(cases) - accepted,
                af_technical_failure_or_unresolved=len(af) - af_accepted,
                coverage=rate(accepted, len(cases)),
                af_coverage=rate(af_accepted, len(af)),
                confidence_deferred=0,
                quality_deferred=0,
                other_policy_deferred=0,
                deferral_rate=rate(0, len(cases)),
                af_deferral_rate=rate(0, len(af)),
                unresolved_reasons=dict(
                    Counter(r["reason"] for r in cases if r["reason"])
                ),
                af_unresolved_reasons=dict(
                    Counter(r["reason"] for r in af if r["reason"])
                ),
                performance="pending all-three-seed seal; see status.json for completeness",
            )
        atomic_json(destination / "accounting-summary.json", summaries)

    save_status()
    try:
        save_accounting()
        write_new(destination / "original-config.yaml", config_path.read_bytes())
        write_new(destination / "resolved-config.json", c)
        write_new(destination / "input-registry.json", registry)
        write_new(destination / "eligible-cases.json", rows)
        write_new(destination / "environment.json", {**runtime, "invocation": sys.argv})
        write_new(
            destination / "audit-log.json",
            dict(
                scope=c["scope"],
                preflight=registry,
                exposure_audit_sha256=sha(local(root, c["integrity_audit"])),
            ),
        )
        verify_signal_files(root, rows)
        for m in c["models"]:
            verify_registry(root, registry)
            seed = str(m["seed"])

            def on_batch(batch_rows, phase):
                ids = {r["record_id"] for r in batch_rows}
                for case in accounting[seed]:
                    if case["record_id"] in ids:
                        case.update(
                            disposition="accepted"
                            if phase == "completed"
                            else "technical_failure",
                            reason=None
                            if phase == "completed"
                            else "batch_in_progress_or_failed",
                        )
                status["holdout_signals_accessed"] = True
                save_accounting()
                save_status()

            z = predict_seed(root, c, m, rows, on_batch)
            require(
                z.dtype == np.float32
                and z.shape == (len(rows), 4)
                and np.isfinite(z).all(),
                "Incomplete predictions",
            )
            path = destination / f"seed{seed}/logits.npy"
            path.parent.mkdir()
            with path.open("xb") as stream:
                np.save(stream, z, allow_pickle=False)
            status["prediction_seals"][seed] = sha(path)
            save_status()
        require(
            all(
                case["disposition"] == "accepted"
                for cases in accounting.values()
                for case in cases
            ),
            "Incomplete case ledger",
        )
        write_new(destination / "prediction-seals.json", status["prediction_seals"])
        status["stage"] = "scoring"
        save_status()
        verify_registry(root, registry)
        summary = score_sealed(destination, rows, status["prediction_seals"])
        write_new(destination / "summary.json", summary)
        write_new(
            destination / "completion.json",
            dict(
                completed_at_utc=utc_now(),
                revision=registry["revision"],
                config_sha256=registry["config_sha256"],
                holdout_consumed_for_future_development=True,
                external_archival="pending",
            ),
        )
        status.update(state="completed", stage="complete")
        save_status()
        write_new(
            destination / "artifact-sha256.json",
            {
                str(p.relative_to(destination)): sha(p)
                for p in sorted(destination.rglob("*"))
                if p.is_file()
            },
        )
    except BaseException as exc:
        for seed, cases in accounting.items():
            if seed not in status["prediction_seals"]:
                for case in cases:
                    if case["disposition"] == "accepted":
                        case.update(
                            disposition="technical_failure",
                            reason="prediction_not_sealed",
                        )
                    if case["reason"] != "not_attempted":
                        case["error"] = dict(type=type(exc).__name__, message=str(exc))
        save_accounting()
        status.update(
            state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=dict(type=type(exc).__name__, message=str(exc)),
            reporting="incomplete; no headline claims; preserve all partial artifacts",
        )
        save_status()
        raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "evaluate"])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/temporal-final-evaluation-v1.yaml")
    )
    parser.add_argument("--authorize-evaluation", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        path = local(root, str(args.config))
        c = load_config(path)
        if args.action == "preflight":
            registry, _ = metadata_preflight(root, c)
            registry["runtime"] = runtime_preflight(root, c)
            print(encoded(registry).decode())
        else:
            print(execute(root, path, c, args.authorize_evaluation))
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"rhythmlens final evaluation: {exc}\n")


if __name__ == "__main__":
    main()
