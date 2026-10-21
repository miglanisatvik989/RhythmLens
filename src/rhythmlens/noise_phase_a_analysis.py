"""Phase A scoring of complete sealed predictions; no signal/model I/O."""

from collections import defaultdict

import numpy as np

from .calibration import LABELS, require
from .final_evaluation import raw_metrics

SEEDS = (42, 43, 44)
PRIMARY = ("macro_f1", "automatic_af_sensitivity", "automatic_error")


def conditions():
    return [("clean", None, None, None)] + [
        (f"{noise}-snr{snr}-repeat{repeat}", noise, snr, repeat)
        for noise in ("bw", "ma", "em")
        for snr in (24, 18, 12, 6, 0, -6)
        for repeat in range(3)
    ]


def primary(cm):
    cm = np.asarray(cm, dtype=np.float64)
    support = cm.sum(axis=-1)
    denominator = support + cm.sum(axis=-2)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    af = np.divide(
        tp[..., 1],
        support[..., 1],
        out=np.full_like(tp[..., 1], np.nan),
        where=support[..., 1] != 0,
    )
    error = 1 - tp.sum(axis=-1) / support.sum(axis=-1)
    return np.stack((f1.mean(axis=-1), af, error), axis=-1)


def summarize(values):
    if all(isinstance(v, dict) for v in values):
        return {k: summarize([v[k] for v in values]) for k in values[0]}
    if all(v is None or type(v) in (int, float) for v in values):
        defined = [v for v in values if v is not None]
        return dict(
            n=len(defined),
            mean=float(np.mean(defined)) if defined else None,
            sample_sd=float(np.std(defined, ddof=1)) if len(defined) > 1 else None,
        )
    return None


def score_condition(rows, z):
    cases, result = raw_metrics(rows, z)
    cm = np.asarray(result["confusion_matrix"])
    for k, label in enumerate(LABELS):
        result["per_class"][label].update(
            tp=int(cm[k, k]),
            fp=int(cm[:, k].sum() - cm[k, k]),
            fn=int(cm[k].sum() - cm[k, k]),
        )
    from .calibration import rate

    false_af = int(cm[0, 1] + cm[2, 1])
    result.update(
        automatic_error=rate(int(cm.sum() - np.trace(cm)), len(rows)),
        false_af_on_N_O=rate(false_af, int(cm[[0, 2]].sum())),
        false_af_share_predictions=rate(false_af, int(cm[:, 1].sum())),
        true_noisy_predicted_A=int(cm[3, 1]),
        review_workload_mean=0.0,
        review_workload_max=0.0,
    )
    return cases, result


def grouped_indices():
    return [[i] for i in range(55)] + [
        list(range(start, start + 3)) for start in range(1, 55, 3)
    ]


def bootstrap(rows, predictions, *, replicates=10000):
    """Shared class-stratified group draws; all seeds/conditions remain paired."""
    require(type(replicates) is int and replicates > 0, "Invalid bootstrap count")
    z = np.asarray(predictions)
    require(
        z.shape == (3, 55, len(rows), 4) and np.isfinite(z).all(),
        "Incomplete bootstrap predictions",
    )
    require(len({r["record_id"] for r in rows}) == len(rows), "Duplicate bootstrap IDs")
    y = np.array([LABELS.index(r["label"]) for r in rows])
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        require(bool(row["duplicate_group"]), "Missing duplicate group")
        groups[row["duplicate_group"]].append(i)
    names = sorted(groups)
    require(all(len(set(y[groups[g]])) == 1 for g in names), "Conflicting group labels")
    strata = [
        [i for i, g in enumerate(names) if y[groups[g][0]] == k] for k in range(4)
    ]
    guess = z.argmax(axis=-1)
    group_cm = np.zeros((len(names), 3, 55, 16), dtype=np.int64)
    for g, name in enumerate(names):
        for s in range(3):
            for c in range(55):
                group_cm[g, s, c] = np.bincount(
                    4 * y[groups[name]] + guess[s, c, groups[name]], minlength=16
                )
    rng = np.random.Generator(np.random.PCG64(20261016))
    draws = np.zeros((replicates, len(names)), dtype=np.int64)
    stats = np.full((replicates, 4, 73, 2, 3), np.nan)
    for b in range(replicates):
        for stratum in strata:
            if stratum:
                chosen = rng.integers(0, len(stratum), size=len(stratum))
                draws[b, stratum] = np.bincount(chosen, minlength=len(stratum))
        cm = np.tensordot(draws[b], group_cm, axes=(0, 0)).reshape(3, 55, 4, 4)
        values = primary(cm)
        aggregated = np.stack(
            [values[:, ix].mean(axis=1) for ix in grouped_indices()], axis=1
        )
        all_seeds = np.concatenate(
            (aggregated, aggregated.mean(axis=0, keepdims=True)), axis=0
        )
        stats[b, :, :, 0] = all_seeds
        stats[b, :, :, 1] = all_seeds - all_seeds[:, :1]
    required = [0, 1, 2] if np.any(y == 1) else [0, 2]
    require(
        np.isfinite(stats[..., required]).all(),
        "Bootstrap numerical failure; no replacement draws",
    )
    names_conditions = [c[0] for c in conditions()] + [
        f"{conditions()[ix[0]][1]}-snr{conditions()[ix[0]][2]}-mean-repetitions"
        for ix in grouped_indices()[55:]
    ]
    intervals = {}
    for s, seed in enumerate((*map(str, SEEDS), "equal_seed_mean")):
        intervals[seed] = {
            name: {
                kind: {
                    metric: dict(
                        interval=np.quantile(
                            stats[:, s, c, j, k], [0.025, 0.975], method="linear"
                        ).tolist()
                        if k in required
                        else None,
                        undefined_reason=None if k in required else "no_AF_labels",
                    )
                    for k, metric in enumerate(PRIMARY)
                }
                for j, kind in enumerate(("absolute", "paired_delta_from_clean"))
            }
            for c, name in enumerate(names_conditions)
        }
    return (
        draws,
        stats,
        dict(
            seed=20261016,
            replicates=replicates,
            generator="PCG64",
            groups={g: [rows[i]["record_id"] for i in groups[g]] for g in names},
            draw_columns=names,
            condition_axis=names_conditions,
            statistic_axes=[
                "replicate",
                "seed42_seed43_seed44_equal_mean",
                "condition_or_repetition_mean",
                "absolute_or_delta",
                list(PRIMARY),
            ],
            intervals=intervals,
            limitations="Reused validation class mix, fixed checkpoints and plans; unknown patients, noise dependence and training sampling unaccounted for; no clinical or final-test claim",
        ),
    )


def score(rows, predictions):
    require(
        predictions.shape == (3, 55, len(rows), 4) and np.isfinite(predictions).all(),
        "Incomplete scoring tensor",
    )
    per_seed, cms = {}, []
    y = np.array([LABELS.index(r["label"]) for r in rows])
    for s, seed in enumerate(SEEDS):
        per_seed[str(seed)] = {}
        matrices = []
        for c, (name, _, _, _) in enumerate(conditions()):
            _, result = score_condition(rows, predictions[s, c])
            per_seed[str(seed)][name] = result
            matrices.append(result["confusion_matrix"])
        cms.append(matrices)
    values = primary(np.asarray(cms))
    delta = values - values[:, :1]
    means = {}
    for ix in grouped_indices()[55:]:
        name, noise, snr, _ = conditions()[ix[0]]
        key = f"{noise}-snr{snr}"
        guesses = predictions[:, ix].argmax(axis=-1)
        means[key] = dict(
            per_seed={
                str(seed): {
                    metric: float(values[s, ix, k].mean())
                    if np.isfinite(values[s, ix, k]).all()
                    else None
                    for k, metric in enumerate(PRIMARY)
                }
                for s, seed in enumerate(SEEDS)
            },
            paired_delta_per_seed={
                str(seed): {
                    metric: float(delta[s, ix, k].mean())
                    if np.isfinite(delta[s, ix, k]).all()
                    else None
                    for k, metric in enumerate(PRIMARY)
                }
                for s, seed in enumerate(SEEDS)
            },
            within_record_mean_error=(guesses != y).mean(axis=1).tolist(),
            within_record_mean_af_detection=[
                [float(v) if y[i] == 1 else None for i, v in enumerate(a)]
                for a in (guesses == 1).mean(axis=1)
            ],
        )
        means[key]["per_seed_all_metric_repetition_summary"] = {
            str(seed): summarize([per_seed[str(seed)][conditions()[i][0]] for i in ix])
            for seed in SEEDS
        }
        means[key]["equal_seed_summary"] = summarize(
            list(means[key]["per_seed"].values())
        )
        means[key]["paired_delta_equal_seed_summary"] = summarize(
            list(means[key]["paired_delta_per_seed"].values())
        )
    return dict(
        per_seed=per_seed,
        equal_seed_summary={
            c[0]: summarize([per_seed[str(s)][c[0]] for s in SEEDS])
            for c in conditions()
        },
        paired_delta_per_seed={
            str(seed): {
                name: {
                    metric: float(delta[s, c, k])
                    if np.isfinite(delta[s, c, k])
                    else None
                    for k, metric in enumerate(PRIMARY)
                }
                for c, (name, *_) in enumerate(conditions())
            }
            for s, seed in enumerate(SEEDS)
        },
        paired_delta_equal_seed_summary={
            name: summarize(
                [
                    {
                        metric: float(delta[s, c, k])
                        if np.isfinite(delta[s, c, k])
                        else None
                        for k, metric in enumerate(PRIMARY)
                    }
                    for s in range(3)
                ]
            )
            for c, (name, *_) in enumerate(conditions())
        },
        repetition_means=means,
        policy="raw_softmax_T1_no_deferral_no_quality_gate",
        real_final_evaluation=False,
    )
