"""Separate Phase A runner. Real I/O requires committed config and runtime authority."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import yaml

from .calibration import (
    LABELS,
    csv_rows,
    local,
    probabilities,
    read_json,
    require,
    require_committed_source,
    sha,
    validate_predictions,
    write_new,
)
from .noise import apply_plan, array_hash, stress_plan
from .noise_phase_a_analysis import SEEDS, bootstrap, conditions, score
from .nstdb import now
from .preprocessing import PreprocessingState, transform_physical

POLICY_SHA256 = "32222409868f6b055baa3fc262bc90f2129c8b2910747338ca3e2a52f25f22ea"
OUTPUT = "outputs/noise-robustness/noise-robustness-v1/phase-a"


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        require(key not in result, "Duplicate configuration key")
        result[key] = loader.construct_object(value_node)
    return result


UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping
)


def canonical(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def load_config(path):
    c = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    require(
        isinstance(c, dict)
        and c.get("protocol_schema") == "rhythmlens-noise-phase-a-v1",
        "Unsupported Phase A schema",
    )
    require(
        type(c.get("execution_authorized")) is bool,
        "Literal boolean authorization required",
    )
    require(
        canonical({k: v for k, v in c.items() if k != "execution_authorized"})
        == POLICY_SHA256,
        "Phase A execution contract changed; requires a versioned review",
    )
    return c


def runtime_versions():
    return dict(
        python=platform.python_version(),
        **{
            k: importlib.metadata.version(k)
            for k in ("numpy", "scipy", "torch", "wfdb", "PyYAML")
        },
    )


def metadata_preflight(root, c):
    """Only explicitly bound compact metadata. No signal/export/checkpoint reads."""
    for name, digest in c["metadata_sha256"].items():
        require(sha(local(root, name)) == digest, f"Changed metadata: {name}")
    p = read_json(local(root, c["noise"]["provenance"]))
    a = read_json(local(root, c["noise"]["audit"]))
    r = read_json(local(root, c["noise"]["roles"]))
    require(a["status"] == "passed" and not a["errors"], "Noise audit failed")
    require(
        a["provenance_sha256"] == canonical(p), "Canonical provenance binding mismatch"
    )
    require(
        r["audit_sha256"] == sha(local(root, c["noise"]["audit"])),
        "Audit byte binding mismatch",
    )
    require(
        r["roles"] == a["roles"] and len(r["roles"]) == 9, "Role descriptors mismatch"
    )
    require(
        {f"{v['record']}/{v['role']}": v["pool_sha256"] for v in r["roles"]}
        == c["noise"]["pool_sha256"],
        "Reviewed pool identities changed",
    )
    from .rhythm_data import FrozenRecords

    records = FrozenRecords(root, c["split_dir"], "validation")
    rows = [{**row, "samples": int(row["samples"])} for row in records.rows]
    require(
        records.identity == c["data_identity"], "Frozen split/data identity changed"
    )
    require(
        len(rows) == 1280
        and [sum(r["label"] == k for r in rows) for k in LABELS] == [762, 114, 362, 42],
        "Eligibility/supports changed",
    )
    require(
        all(
            r["split"] == "validation"
            and r["subject_id"] == "unknown"
            and float(r["sampling_rate_hz"]) == 300
            and r["physical_units"] == "mV"
            for r in rows
        ),
        "Source grouping/units require review",
    )
    return records, rows


def check_identities(root, identities):
    for name, digest in identities.items():
        require(sha(local(root, name)) == digest, f"Input changed: {name}")


def protected_identities(root, c, records):
    identities = dict(c["metadata_sha256"])
    for model in c["models"]:
        for key in (
            "checkpoint",
            "preprocessing",
            "resolved_config",
            "training_report",
            "historical_export",
        ):
            identities[model[key]] = model[key + "_sha256"]
    for row in records.rows:
        for key in ("mat_path", "header_path"):
            name = row[key]
            identities[name] = records.audit["inputs_sha256"][name]
    p = read_json(local(root, c["noise"]["provenance"]))
    for name, item in p["files"].items():
        identities[str(Path(c["noise"]["raw"]) / name)] = item["sha256"]
    return identities


class Backend:
    """Production I/O adapter, constructed only inside authorized execution."""

    def __init__(self, root, c, records):
        from .nstdb import load_pools

        self.root, self.c, self.records = root, c, records
        self.pools = load_pools(
            local(root, c["noise"]["raw"]),
            read_json(local(root, c["noise"]["provenance"])),
            "stress",
        )
        descriptors = read_json(local(root, c["noise"]["roles"]))["roles"]
        for record, pool in self.pools.items():
            require(
                canonical(pool.metadata())
                == canonical(
                    next(
                        r
                        for r in descriptors
                        if r["record"] == record and r["role"] == "stress"
                    )
                ),
                "Reconstructed stress descriptor mismatch",
            )
        self.loaded = {}

    def historical(self, seed):
        m = next(m for m in self.c["models"] if m["seed"] == seed)
        return csv_rows(local(self.root, m["historical_export"]))

    def physical(self, index):
        return self.records.load_physical(index)

    def state_and_model(self, seed):
        if seed not in self.loaded:
            from .rhythm_training import load_checkpoint

            m = next(m for m in self.c["models"] if m["seed"] == seed)
            model, state, checkpoint = load_checkpoint(
                local(self.root, m["checkpoint"]),
                expected_data_identity=self.records.identity,
            )
            require(
                checkpoint["kind"] == "baseline"
                and checkpoint["epoch"] == m["selected_epoch"]
                and checkpoint["model_version"] == "rhythm-dilated-v1"
                and checkpoint["config"]["seed"] == seed,
                "Checkpoint candidate mismatch",
            )
            require(
                state.to_dict() == read_json(local(self.root, m["preprocessing"]))
                and state.scale_mv == 0.18008390945457742
                and state.training_manifest_sha256
                == self.records.identity["train_sha256"],
                "Saved preprocessing mismatch",
            )
            require(
                checkpoint["config"]
                == read_json(local(self.root, m["resolved_config"])),
                "Resolved checkpoint configuration mismatch",
            )
            report = read_json(local(self.root, "reports/rhythm-multiseed-v1.json"))
            run = next(
                r
                for r in report["runs"]
                if r["seed"] == seed and r["architecture"] == "rhythm-dilated-v1"
            )
            require(
                checkpoint["metadata"]["code"]["revision"] == run["source_revision"]
                and checkpoint["metadata"]["code"]["source_bundle_sha256"]
                == run["source_bundle_sha256"],
                "Training source mismatch",
            )
            model.eval()
            self.loaded[seed] = state, model
        return self.loaded[seed]

    def predict(self, model, items):
        import torch
        from .rhythm_data import collate_records

        batch = collate_records(
            [
                dict(
                    record_id=r["record_id"],
                    label=LABELS.index(r["label"]),
                    signal=torch.from_numpy(x.copy()),
                )
                for r, x in items
            ]
        )
        require(not model.training, "Model must be in evaluation mode")
        with torch.inference_mode():
            return model(batch["signals"], batch["mask"]).cpu().numpy().copy()


def freeze_plans(rows, pools):
    require(
        len({r["record_id"] for r in rows}) == len(rows) and len(rows) > 0,
        "Invalid eligible IDs",
    )
    plans = {}
    for i, row in enumerate(rows):
        require(
            row["label"] in LABELS
            and row["duplicate_group"]
            and row["split"] == "validation",
            "Invalid eligible metadata",
        )
        for record in ("bw", "ma", "em"):
            require(pools[record].role == "stress", "Only stress pools permitted")
            for repeat in range(3):
                plans[i, record, repeat] = stress_plan(
                    row["record_id"],
                    record,
                    repeat,
                    row["samples"],
                    pools,
                    partition="validation",
                )
    return plans


def baseline_gate(rows, generated, historical):
    saved, z, y = validate_predictions(historical, rows)
    expected_ids = [r["record_id"] for r in rows]
    require(
        [r["record_id"] for r in saved] == expected_ids, "Historical IDs/order mismatch"
    )
    require(
        generated.shape == z.shape and np.isfinite(generated).all(),
        "Incomplete clean predictions",
    )
    require(
        np.allclose(generated, z, atol=1e-7, rtol=1e-5),
        "Clean logits reproduction mismatch",
    )
    p = probabilities(generated)
    saved_p = np.array([[float(r[f"probability_{k}"]) for k in LABELS] for r in saved])
    require(
        np.allclose(p, saved_p, atol=1e-7, rtol=1e-5),
        "Clean probability reproduction mismatch",
    )
    require(
        np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0),
        "Clean probability sums mismatch",
    )
    require(
        np.array_equal(generated.argmax(axis=1), z.argmax(axis=1)),
        "Clean predictions reproduction mismatch",
    )

    def cm(x):
        return np.bincount(4 * y + x.argmax(axis=1), minlength=16).reshape(4, 4)

    require(
        np.array_equal(cm(generated), cm(z)), "Clean confusion reproduction mismatch"
    )
    return dict(
        status="passed",
        exact_ids_labels_predictions_confusion=True,
        logit_probability_atol=1e-7,
        logit_probability_rtol=1e-5,
        probability_sum_atol=1e-6,
        maximum_absolute_logit_difference=float(np.max(np.abs(generated - z))),
        maximum_absolute_probability_difference=float(np.max(np.abs(p - saved_p))),
        confusion_matrix=cm(generated).tolist(),
    )


def atomic(path, value):
    from .lifecycle import atomic_json

    atomic_json(path, value)


def save_array(path, values):
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)


def sealed_predictions(destination, rows):
    seal = read_json(destination / "seal.json")
    require(
        read_json(destination / "plans.json")["rows"] == rows,
        "Scoring rows differ from frozen plans",
    )
    expected = {
        f"seed{seed}/{name}/logits.npy" for seed in SEEDS for name, *_ in conditions()
    }
    require(
        set(seal["predictions"]) == expected,
        "All three seeds/all conditions must be sealed",
    )
    batches = (len(rows) + 15) // 16
    expected_evidence = {
        "registry.json",
        "plans.json",
        "accounting.json",
        "events.jsonl",
        *(f"seed{seed}/clean/baseline-gate.json" for seed in SEEDS),
        *(
            f"seed{seed}/{name}/batch{batch}.{ext}"
            for seed in SEEDS
            for name, *_ in conditions()
            for batch in range(batches)
            for ext in ("npy", "json")
        ),
    }
    require(
        set(seal["evidence"]) == expected_evidence,
        "Incomplete sealed evidence inventory",
    )
    for name, digest in {**seal["predictions"], **seal["evidence"]}.items():
        require(
            sha(local(destination, name)) == digest, f"Sealed bytes changed: {name}"
        )
    accounting = read_json(destination / "accounting.json")
    require(
        all(
            read_json(destination / f"seed{seed}/clean/baseline-gate.json")["status"]
            == "passed"
            for seed in SEEDS
        ),
        "Missing clean reproduction gate",
    )
    require(
        len(accounting) == 3 * 55 * len(rows)
        and all(r["status"] == "completed" for r in accounting),
        "Incomplete accounting; scoring blocked",
    )
    expected_keys = {
        (seed, name, r["record_id"])
        for seed in SEEDS
        for name, *_ in conditions()
        for r in rows
    }
    require(
        {(r["seed"], r["condition"], r["record_id"]) for r in accounting}
        == expected_keys,
        "Missing/duplicate eligible accounting",
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
                "Invalid sealed logits",
            )
            by_condition.append(z)
        values.append(by_condition)
    return np.asarray(values)


def flat_findings(destination, rows, predictions):
    """Report inapplicable-SNR counts and unchanged outcomes, separate from deferral."""
    from .calibration import rate

    results = {}
    for s, seed in enumerate(SEEDS):
        results[str(seed)] = {}
        for c, (name, *_rest) in enumerate(conditions()):
            flags = set()
            for path in (destination / f"seed{seed}" / name).glob("batch*.json"):
                for case in read_json(path)["cases"]:
                    if (
                        case["corruption"].get("reason")
                        == "flat_or_near_flat_reference"
                    ):
                        require(
                            case["original_physical_sha256"]
                            == case["corrupted_physical_sha256"],
                            "Flat signal changed",
                        )
                        flags.add(case["record_id"])
            indices = [i for i, row in enumerate(rows) if row["record_id"] in flags]
            counts = np.zeros((4, 4), dtype=int)
            for i in indices:
                counts[
                    LABELS.index(rows[i]["label"]), predictions[s, c, i].argmax()
                ] += 1
            results[str(seed)][name] = dict(
                count=len(indices),
                per_eligible=rate(len(indices), len(rows)),
                source_label_by_predicted_label=counts.tolist(),
                reason="flat_or_near_flat_reference",
                achieved_snr_db=None,
                deferrals=0,
            )
    return results


def run_verified(destination, rows, backend, registry, verify_inputs):
    """Internal engine for authorized inputs or generated fixture backends only.

    It cannot discover data or load a checkpoint on its own. Public CLI callers
    must go through execute's two-part authorization and committed-source gate.
    """
    destination.mkdir(parents=True, exist_ok=False)
    status = dict(state="running", started_utc=now(), invocation=sys.argv, details=None)
    accounting = [
        dict(
            seed=seed,
            condition=name,
            record_id=r["record_id"],
            label=r["label"],
            duplicate_group=r["duplicate_group"],
            status="not_started",
            reason="pending",
        )
        for seed in SEEDS
        for name, *_ in conditions()
        for r in rows
    ]
    index = {(r["seed"], r["condition"], r["record_id"]): r for r in accounting}
    atomic(destination / "status.json", status)
    active = []
    details = []
    try:
        write_new(destination / "registry.json", registry)
        atomic(destination / "accounting.json", accounting)
        verify_inputs()
        if callable(backend):
            backend = backend()
        plans = freeze_plans(rows, backend.pools)
        write_new(
            destination / "plans.json",
            dict(
                rows=rows,
                conditions=conditions(),
                plans=[
                    dict(
                        index=i,
                        repeat=r,
                        noise_release="nstdb-1.0.0",
                        role_descriptor=backend.pools[n].metadata(),
                        **p.metadata(),
                    )
                    for (i, n, r), p in plans.items()
                ],
            ),
        )
        atomic(destination / "accounting.json", accounting)
        verify_inputs()
        historical = {seed: backend.historical(seed) for seed in SEEDS}
        for seed in SEEDS:
            validate_predictions(historical[seed], rows)
        from .rhythm_data import LengthBucketSampler

        batches = list(
            LengthBucketSampler([r["samples"] for r in rows], 16, shuffle=False)
        )
        seals = {}
        for c, (name, record, snr, repeat) in enumerate(conditions()):
            for seed in SEEDS:
                folder = destination / f"seed{seed}" / name
                folder.mkdir(parents=True, exist_ok=False)
                z = np.empty((len(rows), 4), dtype=np.float32)
                state, model = backend.state_and_model(seed)
                require(
                    isinstance(state, PreprocessingState),
                    "Explicit saved preprocessing required",
                )
                for batch_number, indices in enumerate(batches):
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
                    started = time.monotonic()
                    for i in indices:
                        row = rows[i]
                        x = backend.physical(i)
                        require(
                            np.asarray(x).shape == (row["samples"],),
                            "Source length mismatch",
                        )
                        if record is None:
                            value, meta = (
                                x,
                                dict(corruption_applied=False, reason="clean_control"),
                            )
                        else:
                            value, meta = apply_plan(
                                x,
                                plans[i, record, repeat],
                                backend.pools,
                                partition="validation",
                                snr_db=snr,
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
                        "Invalid batch logits",
                    )
                    z[indices] = logits
                    save_array(folder / f"batch{batch_number}.npy", logits)
                    write_new(
                        folder / f"batch{batch_number}.json",
                        dict(
                            indices=indices,
                            cases=details,
                            elapsed_seconds=time.monotonic() - started,
                        ),
                    )
                    for item in active:
                        item.update(status="completed", reason=None)
                    active = []
                if c == 0:
                    write_new(
                        folder / "baseline-gate.json",
                        baseline_gate(rows, z, historical[seed]),
                    )
                save_array(folder / "logits.npy", z)
                seals[str((folder / "logits.npy").relative_to(destination))] = sha(
                    folder / "logits.npy"
                )
        atomic(destination / "accounting.json", accounting)
        verify_inputs()
        evidence = {
            name: sha(destination / name)
            for name in (
                "registry.json",
                "plans.json",
                "accounting.json",
                "events.jsonl",
            )
        }
        evidence.update(
            {
                f"seed{seed}/clean/baseline-gate.json": sha(
                    destination / f"seed{seed}/clean/baseline-gate.json"
                )
                for seed in SEEDS
            }
        )
        # Bind all case-level corruption evidence and partial batch predictions too.
        evidence.update(
            {
                str(p.relative_to(destination)): sha(p)
                for p in destination.glob("seed*/*/batch*")
            }
        )
        write_new(
            destination / "seal.json",
            dict(predictions=seals, evidence=evidence, created_utc=now()),
        )
        predictions = sealed_predictions(destination, rows)
        result = score(rows, predictions)
        result["flat_cases"] = flat_findings(destination, rows, predictions)
        draws, statistics, uncertainty = bootstrap(rows, predictions)
        save_array(destination / "bootstrap-draws.npy", draws)
        save_array(destination / "bootstrap-statistics.npy", statistics)
        write_new(destination / "bootstrap.json", uncertainty)
        verify_inputs()
        sealed_predictions(destination, rows)
        write_new(destination / "metrics.json", result)
        write_new(
            destination / "artifacts-sha256.json",
            {
                str(p.relative_to(destination)): sha(p)
                for p in sorted(destination.rglob("*"))
                if p.is_file()
                and p.name not in ("status.json", "artifacts-sha256.json")
            },
        )
        status.update(
            state="completed",
            completed_utc=now(),
            artifacts_index_sha256=sha(destination / "artifacts-sha256.json"),
        )
        atomic(destination / "status.json", status)
    except BaseException as exc:
        write_new(
            destination / "failure-context.json",
            dict(active=active, available_corruption_metadata=details, error=str(exc)),
        )
        for item in active:
            item.update(status="technical_failure", reason=str(exc))
        for item in accounting:
            if item["status"] == "not_started":
                item["reason"] = "blocked_after_failure: " + str(exc)
        atomic(destination / "accounting.json", accounting)
        status.update(
            state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            details=dict(type=type(exc).__name__, message=str(exc)),
            updated_utc=now(),
        )
        atomic(destination / "status.json", status)
        raise


def execute(root, config_path, authorized=False):
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
    require_committed_source(root, config_path)
    require(runtime_versions() == c["environment"], "Environment version mismatch")
    destination = local(root, c["output_root"])
    require(
        not destination.exists() and not destination.is_symlink(),
        "Output collision; preserve, never retry",
    )
    records, rows = metadata_preflight(root, c)
    identities = protected_identities(root, c, records)
    import torch

    old_threads, deterministic = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
    )
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
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
            environment=runtime_versions(),
            runtime=dict(
                cpu_threads=torch.get_num_threads(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                inference_dtype="float32",
                scoring_dtype="float64",
                batch_size=16,
                shuffle=False,
                calibration=False,
                confidence_deferral=False,
                quality_gating=False,
            ),
            created_utc=now(),
        )
        run_verified(
            destination,
            rows,
            lambda: Backend(root, c, records),
            registry,
            lambda: check_identities(root, {**committed, **identities}),
        )
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(deterministic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorize-execution", action="store_true")
    args = parser.parse_args()
    execute(args.root, args.config, args.authorize_execution)


if __name__ == "__main__":
    main()
