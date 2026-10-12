# RhythmLens

RhythmLens is a quality-aware ECG research project exploring how signal quality
affects rhythm analysis, including atrial fibrillation (AF) assessment and explicit
deferral when an ECG cannot be assessed reliably.

This is a retrospective research prototype; clinical diagnostic use is not
validated.

## Status

- Implementation: **pending**.
- Experiments and results: **pending**. No performance claims are available.

## Research principles

- Make experiments reproducible with versioned configurations, split manifests,
  seeds, dependencies, and documented evaluation procedures.
- Prevent leakage across training, validation, and test sets: require
  patient/subject grouping when verified identifiers are available. Otherwise,
  use recording-level splits, keep verified duplicates together, and document
  the limitation. Never invent patient identities. Fit preprocessing only on
  training data.
- Reserve an untouched final holdout for evaluation after all development choices
  are frozen.
- Report AF deferrals explicitly, including their counts, rates, reasons, and
  impact on coverage and performance; do not silently exclude deferred cases.
- Define automatic AF sensitivity as correct automatic AF predictions divided
  by all AF-labeled recordings. Deferred AF recordings and incorrect predictions
  do not count as detections. Report this alongside coverage and accepted-case
  metrics; if there are no AF-labeled recordings, report it as undefined.
- Report only observed results, with limitations and uncertainty stated clearly.

## Repository policy

Keep source code, documentation, experiment configurations, and split manifests
under version control. Split manifests should use de-identified identifiers and
contain no ECG samples or secrets.

Keep secrets, raw and processed ECG data, model weights, checkpoints, and local
run outputs out of Git. Use the ignored directories defined in `.gitignore` for
these artifacts. Track configuration and split manifests separately in `configs/`
and `splits/`, rather than inside ignored artifact directories. Keep
`data/manifests/` and `reports/` trackable as well.

Only explicitly designated synthetic test fixtures may be exempted from data
ignore rules when needed. Document their synthetic provenance and allowlist each
exact fixture path in `.gitignore`; do not broadly exempt a fixture directory.

Data acquisition, model implementation, and evaluation are pending.
