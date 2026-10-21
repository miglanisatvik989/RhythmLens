"""Six-candidate paired scoring. Inputs must have passed the composite seal."""

from collections import defaultdict

import numpy as np

from .calibration import LABELS, require
from .noise_phase_a_analysis import (
    PRIMARY,
    SEEDS,
    conditions,
    grouped_indices,
    primary,
    score,
    summarize,
)


def comparison(rows, predictions):
    z = np.asarray(predictions)
    require(
        z.shape == (2, 3, 55, len(rows), 4) and np.isfinite(z).all(),
        "All six candidates required before comparison",
    )
    arms = {arm: score(rows, z[i]) for i, arm in enumerate(("original", "augmented"))}
    per_seed = {}
    for s, seed in enumerate(SEEDS):
        values = []
        for c in range(55):
            matrices = [
                arms[a]["per_seed"][str(seed)][conditions()[c][0]]["confusion_matrix"]
                for a in arms
            ]
            values.append(primary(np.asarray(matrices)))
        values = np.asarray(values)
        delta = values[:, 1] - values[:, 0]
        per_seed[str(seed)] = {
            (
                conditions()[ix[0]][0]
                if len(ix) == 1
                else f"{conditions()[ix[0]][1]}-snr{conditions()[ix[0]][2]}-mean-repetitions"
            ): {
                k: float(delta[ix, m].mean())
                if np.isfinite(delta[ix, m]).all()
                else None
                for m, k in enumerate(PRIMARY)
            }
            for ix in grouped_indices()
        }
    return dict(
        arms=arms,
        augmented_minus_original_per_seed=per_seed,
        augmented_minus_original_equal_seed_summary={
            name: summarize([per_seed[str(s)][name] for s in SEEDS])
            for name in per_seed["42"]
        },
        interpretation="Reused validation development; no clinical/final-test claim",
    )


def paired_bootstrap(rows, predictions, *, replicates=10000):
    """One draw matrix jointly retains every arm, seed, condition and repetition."""
    z = np.asarray(predictions)
    require(
        z.shape == (2, 3, 55, len(rows), 4) and np.isfinite(z).all(),
        "Incomplete six-candidate bootstrap",
    )
    require(type(replicates) is int and replicates > 0, "Invalid replicates")
    require(len({r["record_id"] for r in rows}) == len(rows), "Duplicate IDs")
    y = np.array([LABELS.index(r["label"]) for r in rows])
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        require(bool(r["duplicate_group"]), "Missing duplicate group")
        groups[r["duplicate_group"]].append(i)
    names = sorted(groups)
    require(all(len(set(y[groups[g]])) == 1 for g in names), "Conflicting group labels")
    strata = [
        [i for i, g in enumerate(names) if y[groups[g][0]] == k] for k in range(4)
    ]
    guess = z.argmax(axis=-1)
    group_cm = np.zeros((len(names), 2, 3, 55, 16), dtype=np.int64)
    for g, name in enumerate(names):
        for a in range(2):
            for s in range(3):
                for c in range(55):
                    group_cm[g, a, s, c] = np.bincount(
                        4 * y[groups[name]] + guess[a, s, c, groups[name]], minlength=16
                    )
    rng = np.random.Generator(np.random.PCG64(20261016))
    draws = np.zeros((replicates, len(names)), dtype=np.int64)
    # arm original/augmented/paired difference; seed42/43/44/equal; condition; estimand; metric
    stats = np.full((replicates, 3, 4, 73, 2, 3), np.nan)
    for b in range(replicates):
        for stratum in strata:
            if stratum:
                ix = rng.integers(0, len(stratum), size=len(stratum))
                draws[b, stratum] = np.bincount(ix, minlength=len(stratum))
        cm = np.tensordot(draws[b], group_cm, axes=(0, 0)).reshape(2, 3, 55, 4, 4)
        values = primary(cm)
        agg = np.stack(
            [values[:, :, ix].mean(axis=2) for ix in grouped_indices()], axis=2
        )
        seeds = np.concatenate((agg, agg.mean(axis=1, keepdims=True)), axis=1)
        stats[b, :2, :, :, 0] = seeds
        stats[b, :2, :, :, 1] = seeds - seeds[:, :, :1]
        stats[b, 2] = stats[b, 1] - stats[b, 0]
    required = [0, 1, 2] if np.any(y == 1) else [0, 2]
    require(np.isfinite(stats[..., required]).all(), "Invalid draw; no replacements")
    condition_names = [c[0] for c in conditions()] + [
        f"{conditions()[ix[0]][1]}-snr{conditions()[ix[0]][2]}-mean-repetitions"
        for ix in grouped_indices()[55:]
    ]
    intervals = {}
    for a, arm in enumerate(("original", "augmented", "augmented_minus_original")):
        intervals[arm] = {
            seed: {
                name: {
                    kind: {
                        metric: dict(
                            interval=np.quantile(
                                stats[:, a, s, c, j, k], [0.025, 0.975], method="linear"
                            ).tolist()
                            if k in required
                            else None,
                            undefined_reason=None if k in required else "no_AF_labels",
                        )
                        for k, metric in enumerate(PRIMARY)
                    }
                    for j, kind in enumerate(
                        (
                            "absolute_or_arm_difference",
                            "stress_minus_clean_or_difference_of_deltas",
                        )
                    )
                }
                for c, name in enumerate(condition_names)
            }
            for s, seed in enumerate((*map(str, SEEDS), "equal_seed_mean"))
        }
    return (
        draws,
        stats,
        dict(
            seed=20261016,
            generator="PCG64",
            replicates=replicates,
            groups={g: [rows[i]["record_id"] for i in groups[g]] for g in names},
            draw_columns=names,
            condition_axis=condition_names,
            intervals=intervals,
            axes=[
                "draw",
                "original_augmented_paired_difference",
                "seed42_43_44_equal",
                "condition_or_repeat_mean",
                "absolute_or_stress_minus_clean",
                list(PRIMARY),
            ],
            quantiles=[0.025, 0.975],
            method="linear",
            shared_draws=True,
        ),
    )
