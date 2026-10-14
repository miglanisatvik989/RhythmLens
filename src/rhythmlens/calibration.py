"""Logits-only exploratory analysis. Never imports signal/model/training code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

LABELS = ("N", "A", "O", "~")
ROLES = ("calibration-fit", "threshold-select", "dev-assess")
ARMS = ("raw_no_deferral", "calibrated_no_deferral", "calibrated_confidence_deferral")
TOLERANCES = dict(
    probability_sum_atol=1e-6,
    probability_atol=1e-7,
    probability_rtol=1e-5,
    baseline_metric_atol=1e-5,
    baseline_metric_rtol=1e-5,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else encoded(value))


def local(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), "Path escapes repository")
    return path


def load_config(path):
    c = yaml.safe_load(Path(path).read_text())
    require(
        c["protocol_schema"] == "rhythmlens-calibration-deferral-v1",
        "Unsupported analysis schema",
    )
    require(
        c["numerical_verification"] == TOLERANCES, "Unsupported verification tolerances"
    )
    require(c["inputs"]["label_order"] == list(LABELS), "Label order mismatch")
    require(
        [m["seed"] for m in c["inputs"]["models"]] == [42, 43, 44],
        "All three seeds required",
    )
    require(c["partition"]["role_order"] == list(ROLES), "Role order mismatch")
    require(
        c["partition"]["algorithm"] == "validation-group-greedy-sha256-v1",
        "Unsupported grouping algorithm",
    )
    require(
        c["partition"]["fractions"] == [0.4, 0.3, 0.3], "Unsupported role fractions"
    )
    require(
        c["decision"]["signal_quality_gate"] == "disabled_unavailable",
        "Quality gate unsupported",
    )
    require(
        c["decision"]["noisy_prediction_is_deferral"] is False,
        "Noisy routing unsupported",
    )
    require(
        c["calibration"]["method"] == "scalar_temperature_four_class_nll",
        "Unsupported calibrator",
    )
    require(
        c["calibration"]["solver"] == "deterministic_golden_section",
        "Unsupported solver",
    )
    require(
        c["selection"]["threshold_grid"]
        == dict(integer_start=0, integer_stop_inclusive=100, divisor=100),
        "Unsupported threshold grid",
    )
    require(
        c["selection"]["objective"] == "minimize_accepted_error_fraction",
        "Unsupported objective",
    )
    require(c["assessment"]["arms"] == list(ARMS), "Unsupported arms")
    require(
        c["assessment"]["binary_af_scope"]
        == "true_N_A_O_existing_p_A_no_renormalization",
        "Invalid binary AF scope",
    )
    require(
        c["assessment"]["reliability_bins"]
        == dict(count=15, spacing="equal_width", range=[0.0, 1.0]),
        "Unsupported bins",
    )
    require(
        c["runtime"]
        == dict(
            device="cpu",
            dtype="float64",
            cpu_threads=2,
            new_dependencies="forbidden_without_review",
        ),
        "Unsupported runtime",
    )
    fixed = {
        "partition": {
            "group_key": "duplicate_group",
            "related_records": "frozen_exact_groups_new_relations_block_for_review",
            "require_all_classes_per_role": True,
            "same_roles_for_all_seeds": True,
        },
        "calibration": {
            "fit_role": ROLES[0],
            "per_seed": True,
            "weighting": "equal_per_recording",
            "additional_candidates": [
                "lower_bound",
                "upper_bound",
                "beta_one",
                "final_midpoint",
            ],
            "ties": ["closest_beta_to_one", "smaller_beta"],
            "nonconvergence": "block_affected_arm",
        },
        "selection": {
            "role": ROLES[1],
            "per_seed": True,
            "confidence": "max_calibrated_probability",
            "accept_operator": "greater_than_or_equal",
            "min_overall_coverage": 0.9,
            "min_af_coverage": 0.9,
            "ties": ["higher_overall_coverage", "smaller_threshold"],
            "ratio_comparisons": "exact_integer_cross_products",
        },
        "decision": {
            "argmax_tie_order": list(LABELS),
            "noisy_prediction": "assessed_four_class_outcome",
            "technical_failures": "explicit_ledger_and_block_complete_assessment",
        },
        "assessment": {
            "role": ROLES[2],
            "unlock_after": "all_three_inputs_verified_and_calibrators_and_thresholds_frozen",
            "retuning": "forbidden",
            "multiclass_brier": "sum_over_classes_no_class_normalization",
            "undefined_denominators": "null_with_reason",
            "bootstrap": "disabled",
        },
        "provenance": {"overwrite_existing": False},
    }
    for section, settings in fixed.items():
        for key, expected in settings.items():
            require(c[section][key] == expected, f"Unsupported {section}.{key}")
    require(
        c["calibration"]["inverse_temperature_bounds"] == [0.05, 20.0],
        "Unsupported beta bounds",
    )
    require(
        c["calibration"]["absolute_bracket_tolerance"] == 1e-8
        and c["calibration"]["max_iterations"] == 256,
        "Unsupported solver limits",
    )
    require(
        c["calibration"]["fitted_temperatures"] is None
        and c["selection"]["selected_thresholds"] is None,
        "Fitted artifacts belong in immutable outputs",
    )
    return c


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def probabilities(z, beta=1.0):
    scaled = np.asarray(z, dtype=np.float64) * beta
    require(
        scaled.ndim == 2 and scaled.shape[1] == 4 and np.isfinite(scaled).all(),
        "Invalid logits",
    )
    centered = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(z, y, beta=1.0):
    if not len(y):
        return None
    scaled = np.asarray(z, dtype=np.float64) * beta
    centered = scaled - scaled.max(axis=1, keepdims=True)
    return float(
        np.mean(np.log(np.exp(centered).sum(axis=1)) - centered[np.arange(len(y)), y])
    )


def validate_predictions(rows, manifest, tolerance=TOLERANCES):
    expected = {r["record_id"]: r["label"] for r in manifest}
    require(len(expected) == len(manifest), "Duplicate manifest IDs")
    require(
        len(rows) == len(expected) and len({r["record_id"] for r in rows}) == len(rows),
        "Missing/duplicate prediction IDs",
    )
    require(
        {r["record_id"] for r in rows} == set(expected), "Prediction ID set mismatch"
    )
    rows = sorted(rows, key=lambda r: r["record_id"])
    require(
        all(r["true_label"] == expected[r["record_id"]] for r in rows),
        "Prediction labels mismatch",
    )
    z = np.array([[float(r[f"logit_{k}"]) for k in LABELS] for r in rows])
    p = probabilities(z)
    saved = np.array([[float(r[f"probability_{k}"]) for k in LABELS] for r in rows])
    require(
        np.isfinite(saved).all() and ((saved >= 0) & (saved <= 1)).all(),
        "Invalid saved probabilities",
    )
    require(
        np.allclose(
            saved.sum(axis=1), 1, atol=tolerance["probability_sum_atol"], rtol=0
        ),
        "Probability sums mismatch",
    )
    require(
        np.allclose(
            saved,
            p,
            atol=tolerance["probability_atol"],
            rtol=tolerance["probability_rtol"],
        ),
        "Saved softmax mismatch",
    )
    guesses = z.argmax(axis=1)
    require(
        all(r["predicted_label"] == LABELS[g] for r, g in zip(rows, guesses)),
        "Predicted classes mismatch",
    )
    require(
        np.array_equal(saved.argmax(axis=1), guesses),
        "Saved probability argmax mismatch",
    )
    return rows, z, np.array([LABELS.index(r["true_label"]) for r in rows])


def preflight(root, c):
    """Read only original metadata and prediction exports; no role assignment."""
    report_path = local(root, c["inputs"]["report"])
    report = read_json(report_path)
    manifest_path = local(root, c["inputs"]["validation_manifest"])
    require(
        sha(manifest_path) == report["data_identity"]["validation_sha256"],
        "Validation hash mismatch",
    )
    require(
        c["inputs"]["assignment_sha256"]
        == report["data_identity"]["assignment_sha256"],
        "Assignment identity mismatch",
    )
    manifest = csv_rows(manifest_path)
    split_dir = local(root, c["inputs"]["split_dir"])
    frozen_path = split_dir / "frozen.json"
    require(
        sha(frozen_path) == report["data_identity"]["frozen_metadata_sha256"],
        "Frozen metadata hash mismatch",
    )
    frozen = read_json(frozen_path)
    require(
        frozen["assignment_sha256"] == c["inputs"]["assignment_sha256"],
        "Frozen assignment mismatch",
    )
    source_hashes = {}
    for name in ("source_manifest", "source_audit", "source_provenance"):
        digest = sha(local(root, c["inputs"][name]))
        require(
            digest == report["data_identity"][f"{name}_sha256"],
            "Source metadata hash mismatch",
        )
        source_hashes[name] = digest
    owners, identifiers = {}, set()
    for partition in ("train", "validation", "holdout"):
        metadata = split_dir / f"{partition}.csv"
        require(
            sha(metadata) == frozen["files_sha256"][metadata.name],
            "Split metadata hash mismatch",
        )
        for row in csv_rows(metadata):
            require(row["record_id"] not in identifiers, "Original split ID overlap")
            identifiers.add(row["record_id"])
            require(
                row.get("subject_id") == "unknown",
                "Verified subjects require grouping review",
            )
            require(
                row["duplicate_group"] not in owners
                or owners[row["duplicate_group"]] == partition,
                "Original split group overlap",
            )
            owners[row["duplicate_group"]] = partition
    require(
        len(manifest) == c["inputs"]["eligible_records"], "Validation count mismatch"
    )
    require(
        sum(r["label"] == "A" for r in manifest) == c["inputs"]["eligible_af_records"],
        "Validation AF count mismatch",
    )
    for name, digest in c["provenance"]["dependency_locks"].items():
        require(
            sha(local(root, name)) == digest == report["dependency_locks_sha256"][name],
            "Lock hash mismatch",
        )
    runs = {
        r["seed"]: r
        for r in report["runs"]
        if r["architecture"] == c["inputs"]["architecture"]
    }
    registry = {
        "report_sha256": sha(report_path),
        "manifest_sha256": sha(manifest_path),
        "frozen_metadata_sha256": sha(frozen_path),
        "source_metadata_sha256": source_hashes,
        "holdout_signals_accessed": False,
        "role_manifest_generated": False,
        "inputs": {},
    }
    data = {}
    for m in c["inputs"]["models"]:
        seed = m["seed"]
        r = runs[seed]
        for key in ("checkpoint_sha256", "best_epoch"):
            require(m[key] == r[key], f"Seed {seed}: {key} mismatch")
        config_path = local(root, m["config"])
        require(
            m["config"] == r["config_path"] and sha(config_path) == r["yaml_sha256"],
            "Training config hash mismatch",
        )
        training = yaml.safe_load(config_path.read_text())
        # Same deterministic path formula as the launcher; authenticated report
        # reproduction.document below must agree with these path inputs.
        base = local(
            root,
            str(
                Path(training["paths"]["output_root"])
                / training["experiment"]["name"]
                / training["environment"]["profile"]
                / f"seed{seed}"
            ),
        )
        evaluation_path = base / "evaluate/evaluation-report.json"
        train_path = base / "train/run-report.json"
        require(
            sha(evaluation_path) == r["evaluation_report_sha256"],
            "Evaluation report hash mismatch",
        )
        require(
            sha(train_path) == r["run_report_sha256"], "Training report hash mismatch"
        )
        ev = read_json(evaluation_path)
        require(
            ev["reproduction"]["experiment"]["document"] == training,
            "Recorded configuration mismatch",
        )
        require(
            ev["partition"] == "validation" and ev["environment"]["device"] == "cpu",
            "Not CPU validation provenance",
        )
        require(ev["holdout_signals_accessed"] is False, "Holdout provenance failure")
        require(
            ev["checkpoint_sha256"] == m["checkpoint_sha256"],
            "Checkpoint provenance mismatch",
        )
        require(
            ev["data_identity"] == report["data_identity"], "Data provenance mismatch"
        )
        tp = ev["training_provenance"]
        require(
            tp["preprocessing_sha256"]
            == c["inputs"]["preprocessing_sha256"]
            == r["preprocessing_sha256"],
            "Preprocessing mismatch",
        )
        require(
            tp["code"]["revision"] == r["source_revision"]
            and tp["code"]["source_bundle_sha256"] == r["source_bundle_sha256"],
            "Training source mismatch",
        )
        require(
            tp["configuration_sha256"] == r["resolved_config_sha256"],
            "Resolved training config mismatch",
        )
        prediction_path = evaluation_path.parent / "validation-predictions.csv"
        if m["prediction_path"] is not None:
            require(
                local(root, m["prediction_path"]) == prediction_path,
                "Prediction path mismatch",
            )
        if m["prediction_sha256"] is not None:
            require(
                sha(prediction_path) == m["prediction_sha256"],
                "Prediction hash mismatch",
            )
        rows, z, y = validate_predictions(csv_rows(prediction_path), manifest)
        confusion = np.zeros((4, 4), dtype=int)
        np.add.at(confusion, (y, z.argmax(axis=1)), 1)
        require(
            confusion.tolist() == ev["metrics"]["confusion_matrix"],
            "Baseline confusion mismatch",
        )
        require(ev["metrics"]["records"] == len(y), "Baseline support mismatch")
        require(
            math.isclose(
                nll(z, y),
                ev["metrics"]["cross_entropy"],
                abs_tol=TOLERANCES["baseline_metric_atol"],
                rel_tol=TOLERANCES["baseline_metric_rtol"],
            ),
            "Baseline NLL mismatch",
        )
        registry["inputs"][str(seed)] = {
            "prediction_path": str(prediction_path.relative_to(root)),
            "prediction_sha256": sha(prediction_path),
            "evaluation_report_sha256": sha(evaluation_path),
            "training_report_sha256": sha(train_path),
            "checkpoint_sha256": m["checkpoint_sha256"],
            "yaml_sha256": sha(config_path),
            "source_revision": r["source_revision"],
            "source_bundle_sha256": r["source_bundle_sha256"],
            "resolved_config_sha256": r["resolved_config_sha256"],
            "records": len(rows),
            "supports": dict(Counter(row["true_label"] for row in rows)),
            "required_logits_verified": True,
            "baseline_nll_verified": True,
        }
        data[seed] = (rows, z, y)
    return registry, manifest, data


def assign_roles(manifest, seed):
    groups = defaultdict(list)
    require(
        len({r["record_id"] for r in manifest}) == len(manifest),
        "Duplicate manifest IDs",
    )
    for row in manifest:
        require(
            row.get("subject_id") == "unknown",
            "Verified subjects require reviewed grouping",
        )
        require(
            row.get("duplicate_group") and row["label"] in LABELS,
            "Invalid group metadata",
        )
        groups[row["duplicate_group"]].append(row)
    require(
        all(len({r["label"] for r in g}) == 1 for g in groups.values()),
        "Conflicting duplicate labels",
    )
    result = []
    for label in LABELS:
        candidates = [(gid, g) for gid, g in groups.items() if g[0]["label"] == label]
        require(len(candidates) >= 3, "At least three groups per class required")
        candidates.sort(
            key=lambda item: (
                -len(item[1]),
                hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest(),
                item[0],
            )
        )
        total = sum(len(g) for _, g in candidates)
        counts = [0, 0, 0]
        for index, (gid, group) in enumerate(candidates):
            empty = [j for j in range(3) if counts[j] == 0]
            allowed = empty if len(candidates) - index == len(empty) else range(3)
            j = max(allowed, key=lambda j: (0.4, 0.3, 0.3)[j] * total - counts[j])
            counts[j] += len(group)
            result.extend(
                dict(
                    record_id=r["record_id"],
                    label=label,
                    subject_id="unknown",
                    duplicate_group=gid,
                    original_split="validation",
                    role=ROLES[j],
                    grouping_provenance="frozen_exact_duplicate_group",
                )
                for r in group
            )
    return sorted(result, key=lambda r: r["record_id"])


def manifest_bytes(roles):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(roles[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(roles)
    return stream.getvalue().encode()


def fit_temperature(z, y, settings):
    require(len(y) > 0, "Empty calibration role")
    lo, hi = settings["inverse_temperature_bounds"]
    require(0 < lo < hi, "Invalid beta bounds")
    original = (lo, hi)
    ratio = (math.sqrt(5) - 1) / 2
    left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fl, fr = nll(z, y, left), nll(z, y, right)
    iterations = 0
    while (
        hi - lo > settings["absolute_bracket_tolerance"]
        and iterations < settings["max_iterations"]
    ):
        if fl <= fr:
            hi, right, fr = right, left, fl
            left = hi - ratio * (hi - lo)
            fl = nll(z, y, left)
        else:
            lo, left, fl = left, right, fr
            right = lo + ratio * (hi - lo)
            fr = nll(z, y, right)
        iterations += 1
    require(
        hi - lo <= settings["absolute_bracket_tolerance"],
        "Temperature fit did not converge",
    )
    candidates = [*original, 1.0, (lo + hi) / 2]
    values = [(nll(z, y, b), abs(b - 1), b) for b in candidates]
    require(all(math.isfinite(v[0]) for v in values), "Nonfinite temperature objective")
    objective, _, beta = min(values)
    return dict(
        beta=beta,
        temperature=1 / beta,
        objective=objective,
        iterations=iterations,
        final_bracket=[lo, hi],
        converged=True,
        boundary=beta in original,
    )


def select_threshold(z, y, beta, settings):
    p = probabilities(z, beta)
    guess, confidence = z.argmax(axis=1), p.max(axis=1)
    af = y == 1
    require(len(y) and af.sum(), "Selection needs labeled AF support")
    table, feasible = [], []
    # Decimal coverage constraints are converted to exact integer fractions.
    from fractions import Fraction

    coverage = Fraction(str(settings["min_overall_coverage"]))
    af_coverage = Fraction(str(settings["min_af_coverage"]))
    require(0 <= coverage <= 1 and 0 <= af_coverage <= 1, "Invalid coverage target")
    for j in range(101):
        accepted = confidence >= j / 100
        count, af_count = int(accepted.sum()), int((accepted & af).sum())
        errors = int((accepted & (guess != y)).sum())
        ok = (
            count * coverage.denominator >= len(y) * coverage.numerator
            and af_count * af_coverage.denominator
            >= int(af.sum()) * af_coverage.numerator
        )
        row = dict(
            threshold=j / 100,
            accepted=count,
            eligible=len(y),
            af_accepted=af_count,
            af_eligible=int(af.sum()),
            errors=errors,
            automatic_af_correct=int((accepted & af & (guess == 1)).sum()),
            feasible=ok,
        )
        table.append(row)
        if ok:
            feasible.append((Fraction(errors, count), -count, j))
    require(feasible, "No feasible confidence threshold")
    return dict(threshold=min(feasible)[2] / 100, table=table)


def rate(num, den):
    return dict(
        numerator=num,
        denominator=int(den),
        value=num / den if den else None,
        undefined_reason=None if den else "zero_denominator",
    )


def ledger(rows, z, beta, threshold):
    p = probabilities(z, beta)
    require(
        np.array_equal(z.argmax(axis=1), p.argmax(axis=1)), "Calibration changed argmax"
    )
    return [
        dict(
            record_id=r["record_id"],
            true_label=r["true_label"],
            predicted_label=LABELS[int(z[i].argmax())],
            probabilities=p[i].tolist(),
            nll=nll(z[i : i + 1], np.array([LABELS.index(r["true_label"])]), beta),
            disposition="accepted"
            if p[i].max() >= threshold
            else "confidence_deferred",
        )
        for i, r in enumerate(rows)
    ]


def reliability(pairs, bins=15):
    groups = [[] for _ in range(bins)]
    for confidence, outcome in pairs:
        groups[min(int(confidence * bins), bins - 1)].append((confidence, outcome))
    records = []
    ece = 0.0
    for index, group in enumerate(groups):
        count = len(group)
        mean = sum(c for c, _ in group) / count if count else None
        successes = sum(y for _, y in group)
        observed = successes / count if count else None
        if count:
            ece += count / len(pairs) * abs(mean - observed)
        records.append(
            dict(
                left=index / bins,
                right=(index + 1) / bins,
                count=count,
                successes=successes,
                mean_probability=mean,
                observed_fraction=observed,
            )
        )
    return dict(
        denominator=len(pairs),
        bins=records,
        ece=ece if pairs else None,
        undefined_reason=None if pairs else "zero_denominator",
    )


def metrics(cases):
    """Recompute all metrics from saved case ledger, including every eligible case."""
    require(
        len({r["record_id"] for r in cases}) == len(cases), "Duplicate ledger cases"
    )
    dispositions = (
        "accepted",
        "confidence_deferred",
        "quality_deferred",
        "technical_failure",
    )
    matrix = np.zeros((4, 7), dtype=int)
    for row in cases:
        require(row["disposition"] in dispositions, "Unknown disposition")
        column = (
            LABELS.index(row["predicted_label"])
            if row["disposition"] == "accepted"
            else 4 + dispositions.index(row["disposition"]) - 1
        )
        matrix[LABELS.index(row["true_label"]), column] += 1
    confusion = matrix[:, :4]
    E, C, A, CA = (
        len(cases),
        int(confusion.sum()),
        int(matrix[1].sum()),
        int(confusion[1].sum()),
    )
    counts = matrix.sum(axis=0).tolist()
    require(sum(counts) == E, "Eligible accounting mismatch")
    require(sum(matrix[1]) == A, "AF accounting mismatch")
    failures = counts[6]
    per_class = {}
    for k, label in enumerate(LABELS):
        tp, support, predicted = (
            int(confusion[k, k]),
            int(confusion[k].sum()),
            int(confusion[:, k].sum()),
        )
        per_class[label] = dict(
            support=support,
            predicted=predicted,
            precision=rate(tp, predicted),
            recall=rate(tp, support),
            f1=rate(2 * tp, support + predicted),
        )
    f1 = [v["f1"]["value"] for v in per_class.values()]
    top = [
        (max(r["probabilities"]), int(r["predicted_label"] == r["true_label"]))
        for r in cases
        if r["disposition"] != "technical_failure"
    ]
    known = [
        r
        for r in cases
        if r["true_label"] in LABELS[:3] and r["disposition"] != "technical_failure"
    ]
    binary = [(r["probabilities"][1], int(r["true_label"] == "A")) for r in known]
    D = counts[4] + counts[5]
    result = dict(
        complete=not failures,
        eligible=E,
        accepted=C,
        labeled_af=A,
        outcome_columns=[
            *LABELS,
            "confidence_deferred",
            "quality_deferred",
            "technical_failure",
        ],
        outcome_matrix=matrix.tolist(),
        confusion_matrix=confusion.tolist(),
        af_outcomes=matrix[1].tolist(),
        technical_failures=failures,
        coverage=rate(C, E),
        rhythm_output_coverage=rate(sum(counts[:3]), E),
        deferral_rate=rate(D, E),
        af_coverage=rate(CA, A),
        af_deferral_rate=rate(int(matrix[1, 4:6].sum()), A),
        automatic_af_sensitivity=rate(int(confusion[1, 1]), A),
        accepted_af_recall=rate(int(confusion[1, 1]), CA),
        accepted_af_precision=rate(int(confusion[1, 1]), int(confusion[:, 1].sum())),
        accepted_accuracy=rate(int(np.trace(confusion)), C),
        selective_risk=rate(C - int(np.trace(confusion)), C),
        per_class=per_class,
        macro_f1=sum(f1) / 4 if all(v is not None for v in f1) else None,
        macro_f1_undefined_reason=None
        if all(v is not None for v in f1)
        else "undefined_class_f1",
        legacy_zero_fill_macro_f1=sum(v or 0 for v in f1) / 4,
        reasons={},
        probability_metrics=None,
    )
    for column, reason in ((4, "confidence"), (5, "signal_quality")):
        dc, da = counts[column], int(matrix[1, column])
        result["reasons"][reason] = dict(
            count=dc,
            per_eligible=rate(dc, E),
            share_deferred=rate(dc, D),
            af_count=da,
            per_af=rate(da, A),
            share_af_deferred=rate(da, int(matrix[1, 4:6].sum())),
        )
    if not failures:
        brier = sum(
            sum(
                (p - int(k == LABELS.index(r["true_label"]))) ** 2
                for k, p in enumerate(r["probabilities"])
            )
            for r in cases
        )
        result["probability_metrics"] = dict(
            nll=rate(sum(r["nll"] for r in cases), E),
            multiclass_brier=rate(brier, E),
            top_label_reliability=reliability(top),
            binary_af_eligible=len(known),
            binary_af_excluded_noisy=sum(r["true_label"] == "~" for r in cases),
            binary_af_brier=rate(sum((p - y) ** 2 for p, y in binary), len(binary)),
            binary_af_reliability=reliability(binary),
        )
    return result


def check_gate(frozen, registry, roles_sha, config_sha):
    require(
        set(registry["inputs"]) == {"42", "43", "44"},
        "All three verified inputs required",
    )
    require(
        all(
            r["required_logits_verified"] and r["baseline_nll_verified"]
            for r in registry["inputs"].values()
        ),
        "All input verification must pass",
    )
    require(
        frozen["config_sha256"] == config_sha and frozen["roles_sha256"] == roles_sha,
        "Frozen config/roles mismatch",
    )
    require(frozen["input_registry"] == registry, "Frozen inputs changed")
    require(
        set(frozen["policies"]) == {"42", "43", "44"},
        "All three policies required before assessment",
    )
    for policy in frozen["policies"].values():
        require(
            policy["fit"]["converged"]
            and math.isfinite(policy["fit"]["beta"])
            and policy["fit"]["beta"] > 0,
            "Unfrozen calibrator",
        )
        require(0 <= policy["threshold"] <= 1, "Unfrozen threshold")


def environment(root):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root).decode().strip()

    sources = ["src/rhythmlens/calibration.py", "scripts/manage.py"]
    return dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        invocation=sys.argv,
        python=sys.version,
        platform=platform.platform(),
        cpu_threads=2,
        dtype="float64",
        revision=git("rev-parse", "HEAD"),
        working_tree_status=git("status", "--porcelain"),
        diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
        ).hexdigest(),
        source_sha256={p: sha(root / p) for p in sources},
        installed_versions={
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
    )


def require_authorization(
    root, config_path, authorized, document, derived_manifest=None
):
    require(
        document.get("execution_authorized") is True,
        "Real analysis requires YAML execution_authorized to be the literal boolean true",
    )
    require(
        authorized is True,
        "Real analysis requires explicit --authorize-analysis after protocol commitment",
    )
    require_committed_source(root, config_path, derived_manifest)


def require_committed_source(root, config_path, derived_manifest=None):
    paths = ["."]
    if derived_manifest:
        paths.extend(
            [
                f":(exclude){derived_manifest}",
                f":(exclude){Path(derived_manifest).with_suffix('.provenance.json')}",
            ]
        )
    require(
        not subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
            cwd=root,
        ).strip(),
        "Commit reviewed protocol and implementation with a clean tree before real analysis",
    )
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(config_path.relative_to(root))],
        cwd=root,
        check=True,
        capture_output=True,
    )


def aggregate(results):
    paths = [
        ("accepted_accuracy", "value"),
        ("selective_risk", "value"),
        ("coverage", "value"),
        ("rhythm_output_coverage", "value"),
        ("automatic_af_sensitivity", "value"),
        ("accepted_af_recall", "value"),
        ("accepted_af_precision", "value"),
        ("af_coverage", "value"),
        ("af_deferral_rate", "value"),
        ("macro_f1",),
        ("probability_metrics", "nll", "value"),
        ("probability_metrics", "multiclass_brier", "value"),
        ("probability_metrics", "binary_af_brier", "value"),
    ]
    paths += [(name, "value") for name in ("deferral_rate",)]
    paths += [
        ("per_class", label, metric, "value")
        for label in LABELS
        for metric in ("precision", "recall", "f1")
    ]
    paths += [
        ("probability_metrics", "top_label_reliability", "ece"),
        ("legacy_zero_fill_macro_f1",),
    ]
    paths += [
        ("reasons", reason, metric, "value")
        for reason in ("confidence", "signal_quality")
        for metric in ("per_eligible", "share_deferred", "per_af", "share_af_deferred")
    ]
    output = {}
    for path in paths:
        by_arm = {}
        for arm in ARMS:
            values, missing = [], []
            for seed in ("42", "43", "44"):
                value = results[seed][arm]
                for key in path:
                    value = value[key] if value is not None else None
                if value is None:
                    missing.append(seed)
                else:
                    values.append(value)
            by_arm[arm] = dict(
                n=len(values),
                mean=float(np.mean(values)) if values else None,
                sample_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                undefined_seeds=missing,
            )
        differences = {}
        for seed in ("42", "43", "44"):
            values = []
            for arm in ARMS:
                value = results[seed][arm]
                for key in path:
                    value = value[key] if value is not None else None
                values.append(value)
            differences[seed] = {
                f"{ARMS[i]}_minus_{ARMS[0]}": values[i] - values[0]
                if values[i] is not None and values[0] is not None
                else None
                for i in (1, 2)
            }
        output[".".join(path)] = dict(arms=by_arm, paired_differences=differences)
    return output


def run(root, config_path, action, execution_id=None, authorized=False):
    c = load_config(config_path)
    if action != "analysis-preflight":
        require_authorization(
            root,
            config_path,
            authorized,
            c,
            c["partition"]["manifest_path"] if action == "analysis-assess" else None,
        )
        require(
            execution_id
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", execution_id),
            "Safe execution ID required",
        )
    registry, manifest, data = preflight(root, c)
    if action == "analysis-preflight":
        return registry  # Read-only: no output directory, roles or fit artifacts.
    roles = assign_roles(manifest, c["partition"]["seed"])
    role_bytes = manifest_bytes(roles)
    role_digest = hashlib.sha256(role_bytes).hexdigest()
    require(
        c["partition"]["manifest_sha256"] in (None, role_digest),
        "Configured role hash mismatch",
    )
    role_path = local(root, c["partition"]["manifest_path"])
    root_template = c["provenance"]["local_output_template"].split("/seed{seed}")[0]
    destination = local(root, root_template.replace("<execution-id>", execution_id))
    config_digest = sha(config_path)
    if action == "analysis-fit":
        destination.mkdir(parents=True, exist_ok=False)
        if role_path.exists():
            require(
                role_path.read_bytes() == role_bytes,
                "Existing role manifest differs; never overwrite",
            )
        else:
            write_new(role_path, role_bytes)
        role_provenance = dict(
            roles_sha256=role_digest,
            parent_validation_sha256=registry["manifest_sha256"],
            config_sha256=config_digest,
            partition=c["partition"],
            input_registry=registry,
        )
        role_provenance_path = role_path.with_suffix(".provenance.json")
        if role_provenance_path.exists():
            require(
                role_provenance_path.read_bytes() == encoded(role_provenance),
                "Existing role provenance differs",
            )
        else:
            write_new(role_provenance_path, role_provenance)
        write_new(destination / "roles.csv", role_bytes)
        write_new(destination / "input-registry.json", registry)
        policies = {}
        roles_by_id = {r["record_id"]: r["role"] for r in roles}
        for seed, (rows, z, y) in data.items():
            fit_mask = np.array([roles_by_id[r["record_id"]] == ROLES[0] for r in rows])
            selection_mask = np.array(
                [roles_by_id[r["record_id"]] == ROLES[1] for r in rows]
            )
            fitted = fit_temperature(z[fit_mask], y[fit_mask], c["calibration"])
            selected = select_threshold(
                z[selection_mask], y[selection_mask], fitted["beta"], c["selection"]
            )
            policies[str(seed)] = dict(fit=fitted, threshold=selected["threshold"])
            write_new(destination / f"seed{seed}/selection.json", selected)
            write_new(destination / f"seed{seed}/calibration.json", fitted)
            for role, mask in ((ROLES[0], fit_mask), (ROLES[1], selection_mask)):
                subset = [r for r, use in zip(rows, mask) if use]
                for arm, beta, threshold in (
                    (ARMS[0], 1.0, 0.0),
                    (ARMS[1], fitted["beta"], 0.0),
                    (ARMS[2], fitted["beta"], selected["threshold"]),
                ):
                    cases = ledger(subset, z[mask], beta, threshold)
                    write_new(
                        destination / f"seed{seed}/{role}/{arm}-ledger.json", cases
                    )
                    write_new(
                        destination / f"seed{seed}/{role}/{arm}-metrics.json",
                        metrics(cases),
                    )
        freeze = dict(
            config_sha256=config_digest,
            roles_sha256=role_digest,
            policies=policies,
            input_registry=registry,
            environment=environment(root),
            role_supports={
                role: dict(Counter(r["label"] for r in roles if r["role"] == role))
                for role in ROLES
            },
            protocol_sha256=sha(local(root, c["protocol_document"])),
        )
        freeze["role_provenance_sha256"] = sha(role_provenance_path)
        check_gate(freeze, registry, role_digest, config_digest)
        freeze["artifact_sha256"] = {
            str(p.relative_to(destination)): sha(p)
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        }
        write_new(
            destination / "frozen-policy.json", freeze
        )  # Last write opens assessment gate.
        return dict(
            status="frozen",
            frozen_policy_sha256=sha(destination / "frozen-policy.json"),
            dev_assess_inspected=False,
        )
    require(action == "analysis-assess", "Unknown analysis action")
    frozen = read_json(destination / "frozen-policy.json")
    require(role_path.read_bytes() == role_bytes, "Role manifest changed")
    require(
        sha(role_path.with_suffix(".provenance.json"))
        == frozen["role_provenance_sha256"],
        "Role provenance changed",
    )
    check_gate(frozen, registry, role_digest, config_digest)
    require(
        frozen["protocol_sha256"] == sha(local(root, c["protocol_document"])),
        "Protocol changed",
    )
    require(
        frozen["environment"]["revision"] == environment(root)["revision"],
        "Analysis implementation revision changed",
    )
    for name, digest in frozen["artifact_sha256"].items():
        require(
            sha(local(destination, name)) == digest, "Frozen analysis artifact changed"
        )
    for seed, policy in frozen["policies"].items():
        require(
            read_json(destination / f"seed{seed}/calibration.json") == policy["fit"],
            "Frozen calibrator changed",
        )
        require(
            read_json(destination / f"seed{seed}/selection.json")["threshold"]
            == policy["threshold"],
            "Frozen threshold changed",
        )
    assessment = destination / "assessment"
    assessment.mkdir(exist_ok=False)
    results = {}
    assessment_ids = {r["record_id"] for r in roles if r["role"] == ROLES[2]}
    for seed, (rows, z, y) in data.items():
        policy = frozen["policies"][str(seed)]
        mask = np.array([r["record_id"] in assessment_ids for r in rows])
        subset = [r for r, use in zip(rows, mask) if use]
        results[str(seed)] = {}
        for arm, beta, threshold in (
            (ARMS[0], 1.0, 0.0),
            (ARMS[1], policy["fit"]["beta"], 0.0),
            (ARMS[2], policy["fit"]["beta"], policy["threshold"]),
        ):
            path = assessment / f"seed{seed}/{arm}-ledger.json"
            write_new(path, ledger(subset, z[mask], beta, threshold))
            # Reports derive from persisted ledgers, not transient metric accumulators.
            result = metrics(read_json(path))
            require(
                result["complete"] and result["eligible"] == len(assessment_ids),
                "Incomplete assessment",
            )
            write_new(path.with_name(f"{arm}-metrics.json"), result)
            results[str(seed)][arm] = result
        require(
            results[str(seed)][ARMS[0]]["confusion_matrix"]
            == results[str(seed)][ARMS[1]]["confusion_matrix"],
            "Calibration changed baseline decisions",
        )
    summary = dict(
        status="exploratory",
        seeds=results,
        aggregates=aggregate(results),
        parent_frozen_policy_sha256=sha(destination / "frozen-policy.json"),
        environment=environment(root),
        holdout_signals_accessed=False,
    )
    write_new(assessment / "summary.json", summary)
    write_new(
        assessment / "artifact-provenance.json",
        {
            str(p.relative_to(assessment)): sha(p)
            for p in sorted(assessment.rglob("*"))
            if p.is_file()
        },
    )
    return dict(
        status="exploratory_assessment_complete",
        summary_path=str((assessment / "summary.json").relative_to(root)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["analysis-preflight", "analysis-fit", "analysis-assess"]
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution-id")
    parser.add_argument("--authorize-analysis", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=2):
            result = run(
                root,
                args.config.resolve(),
                args.action,
                args.execution_id,
                args.authorize_analysis,
            )
        print(json.dumps(result, indent=2, allow_nan=False))
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(1, f"rhythmlens analysis: {exc}\n")


if __name__ == "__main__":
    main()
