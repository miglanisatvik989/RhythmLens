# Research requirements

- Preserve reproducibility: record seeds, dependency versions, data provenance,
  preprocessing, split manifests, configurations, and evaluation procedures.
- Prevent leakage: require patient/subject grouping when verified identifiers
  are available (and source/time grouping where appropriate). Otherwise, use
  recording-level splits, keep verified duplicates together, and document the
  limitation. Never invent patient identities. Fit preprocessing only on
  training data. Use validation data for development choices.
- Keep the final holdout untouched during development. Freeze all model,
  threshold, quality-gating, and analysis choices before final evaluation; never
  tune against holdout outcomes.
- Report AF deferrals explicitly: counts, rates, reasons, coverage, and metrics
  for assessed cases with clear denominators. Account for every eligible case
  and distinguish deferral from a negative AF prediction.
- Define automatic AF sensitivity as correct automatic AF predictions divided
  by all AF-labeled recordings. Deferred AF recordings and incorrect predictions
  do not count as detections. Report this alongside coverage and accepted-case
  metrics; if there are no AF-labeled recordings, report it as undefined.
- Never fabricate results, metrics, experiments, or citations. Mark unimplemented
  work and unavailable results as pending.
- Track configurations and de-identified split manifests. Keep secrets, ECG
  samples, weights, checkpoints, and local run outputs out of Git.
- Keep `configs/`, `splits/`, `data/manifests/`, and `reports/` trackable. Exempt
  data files only for explicitly designated synthetic test fixtures when needed:
  document synthetic provenance and allowlist exact paths in `.gitignore`.
