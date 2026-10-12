# RhythmLens multiseed follow-up v1

This protocol is frozen before additional training. Seed42 results informed this
follow-up; the complete comparison was not prospectively planned before seed42.
Report every seed (42, 43, 44) for both architectures, without best-seed selection,
tuning between runs, or changing the bounded smoke target after a failure.

The reference is `rhythm-cnn-v1` from `config.yaml` (stem 32/kernel 15,
channels 32/64/128, kernel 7). The candidate is `rhythm-dilated-v1` from
`configs/temporal-cuda128.yaml` (stem 16/kernel 15, channels 16/24/32,
kernel 7; version-fixed dilations). This comparison changes multiple architectural
properties, including capacity and temporal context; it does not isolate temporal
context alone or establish a causal architectural mechanism.

New configurations are `configs/{reference,temporal}-cuda128-seed{43,44}.yaml`.
Only experiment.seed differs from the authoritative architecture template.
Existing experiment names and seed-qualified output paths avoid collisions.
Seed42 configurations and artifacts must remain unchanged.

Use existing cuda128 environment, CUDA device 0 and two CPU threads, sequentially.
Reuse frozen `splits/challenge2017-v3-seed42-v1` assignments: 5,969 training,
1,280 validation and 1,279 untouched holdout recordings. Training seeds must never
regenerate splits. No downloads or holdout signal access are permitted.
Patient identifiers remain unverified; recording-level splits keep verified
duplicates together but do not establish patient independence.

Preprocessing remains `median-fixed-training-rms-v1`, fitted only on training:
native 300 Hz physical mV, median centering, fixed training RMS scale. Reuse is null
as in both templates. Full training uses batch size 16, AdamW learning rate 0.001,
weight decay 0.0001, dropout 0.1, maximum 50 epochs, patience 8, caching enabled,
and validation four-class macro F1 for best-checkpoint selection.

For each new configuration run launcher doctor, bounded training-only smoke,
full train, then evaluate saved best checkpoint with configured CPU evaluation.
Smoke retains one training record per class, batch size 4, dropout 0.0,
200 steps / 1,200 seconds maximum, accuracy target 1.0 and loss target 0.1.
Verify target and checkpoint reload before full training. A failed smoke stops
that affected run without tuning. Preserve every failure and its artifacts;
only explicitly compatible epoch-boundary resume is allowed. Never overwrite runs.
Canonical full training requires clean committed source; no development override.

Report checkpoint-selected validation accuracy, four-class macro F1, each class
F1, AF precision/recall and correct/all-AF counts, coverage and deferrals; best
epoch, stop reason, duration, parameters, configuration/source/lock identities.
Verify split/preprocessing identities, train/validation counts, environment,
checkpoint reload, independent evaluation agreement and no holdout signal access.
Aggregate observed runs with arithmetic mean and sample standard deviation
(denominator n-1); disclose actual n and every missing/failed run. Report temporal
minus reference paired differences by seed for macro F1 and AF recall, without
claiming statistical significance from three seeds.

Three seeds assess training variability, not patient independence. Calibration
and quality-based deferral remain pending. No deferral policy is implemented;
automatic AF sensitivity uses all AF-labeled recordings, with deferrals and errors
excluded from detections (undefined if there are no AF labels). Holdout remains
untouched. This comparison cannot establish clinical readiness. Do not initiate
quality gating, calibration, searches or final holdout evaluation.

Compact observed metrics and provenance belong in `reports/rhythm-multiseed-v1.md`
and accompanying compact JSON if useful. Checkpoints, predictions, ECG and local
run outputs remain ignored. Commit only these experiment inputs and reports;
do not push or modify implementation, dependencies, locks or learning settings.
