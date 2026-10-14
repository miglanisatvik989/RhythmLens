# Temporal calibration and confidence deferral v1 — draft for review

Status: implemented for synthetic verification; real execution is not authorized. No real calibration,
thresholds, new assessments, or derived manifests have been computed. All such
results are pending. The companion [YAML](../../configs/temporal-calibration-deferral-v1.yaml)
uses an analysis-only schema supported by the documented `scripts/manage.py`
analysis actions. Training schema-v2 behavior is unchanged; never pass this YAML
to training actions.

## Scope and evidence limits

Use the three existing `rhythm-dilated-v1` best checkpoints, seeds 42, 43, 44,
with equal status. Fit one calibrator and select one confidence threshold per
seed using exactly the same partitions and procedure. Report every seed; do not
select a best seed, replace a checkpoint, ensemble predictions, or retrain.
The [completed multiseed report](../../reports/rhythm-multiseed-v1.md) and its
[JSON provenance](../../reports/rhythm-multiseed-v1.json) are the source evidence.

Validation has already informed checkpoint selection (four-class macro F1,
early stopping) and architecture selection/prioritization. Seed42 also informed
the multiseed follow-up. Subdividing that validation set cannot undo this reuse.
All further results here are exploratory development evidence, including the
separate development assessment; it is not an independent test or final holdout.
Three seeds measure training variability, not three independent datasets.

Preserve implementation, original YAMLs, locks, checkpoints, reports, predictions,
and frozen train/validation/holdout splits byte for byte. No holdout signal access,
new inference, training, calibration fitting, threshold search, experiments,
commit, or push is authorized by this draft. Future execution needs review, protocol/implementation commitment, a clean
working tree, and an explicit instruction to proceed. Final holdout evaluation
is a separate future decision after all choices and artifacts are frozen.

## Fixed inputs and eligibility

Original split: `splits/challenge2017-v3-seed42-v1`; 5,969 train, 1,280 validation,
1,279 holdout. Validation labels: N=762, A=114, O=362, ~=42, ordered N,A,O,~.
Every validation record, including noisy and flat recordings, is eligible.
No eligibility filtering based on confidence, label, correctness, or quality.

Use the existing saved best-checkpoint CPU validation prediction CSV for each
seed, containing `record_id`, `true_label`, `predicted_label`, four probabilities
and `logit_N`, `logit_A`, `logit_O`, `logit_~`. Resolve each file from its existing
evaluation report, never from latest-file discovery. Exact local prediction
paths and hashes were resolved by read-only preflight and are now pinned in the
companion YAML; see the [verification evidence](../../reports/temporal-calibration-deferral-v1-verification.md).
Do not regenerate missing artifacts:
stop and report missing inputs. No signal decoding or checkpoint loading is
needed for the proposed logits-only analysis.

| Seed | Best epoch | Checkpoint SHA256 from completed report |
| --- | ---: | --- |
| 42 | 31 | `217110e48c995fcb8eac9a7eb3885356170fd461dc50b333cd62fd1022d2031d` |
| 43 | 42 | `c420c2a96ecf1e9f25ad4a7417a321a96b9eb47c217db613d9502c26e2bcde74` |
| 44 | 39 | `e6965190d29b0214a732f6bf5ef4172daaae3ede30b2e549c8d7c9760a8791c5` |

Reuse `median-fixed-training-rms-v1`, native 300 Hz, physical mV, per-record
median centering and fixed training-fitted RMS scale 0.18008390945457742 mV
(5,969 fitting records). Never refit preprocessing on validation. Preprocessing
SHA256 `a4719420f5b867e3870386b488c954fb85f1b695f2264f57235cf0e0b80783b6`;
original assignment SHA256
`fb27f37a8f9dcd1d8931116e252290a68be29844b5b81badc3223954184054c7`.

Before future analysis, verify hashes against the completed provenance, join
each CSV one-to-one to frozen validation IDs and labels, and require exactly
1,280 rows, finite four-class logits and consistent raw softmax/argmax. Use the
first class in N,A,O,~ order for exact argmax ties. Missing/repeated IDs, label
disagreement, nonfinite values or provenance mismatch block the affected run;
never silently drop cases or substitute another seed. Report all failed inputs.

## Derived validation roles and grouping

Create a new derived manifest only after execution is authorized. Original split
files stay unchanged. Use calibration-fit / threshold-select / dev-assess target
fractions 0.40 / 0.30 / 0.30 of validation, approximately 512 / 384 / 384 records;
these are targets, not observed counts. Approximate dev-assess supports are
34 AF and 13 noisy cases, not the full 114 AF cases. These are planning estimates;
actual supports must come from the frozen role manifest. Use one assignment for all seeds.

Use frozen `duplicate_group` as the indivisible unit, including its transitive
related-record closure if additional verified relationships are documented.
Never split a related group between roles. Never infer patient identity from
record names or waveform similarity. Patient IDs are currently unknown; patient
and source/time independence remain unverified. If verified subject/source/time
relations become available, stop for a reviewed grouping amendment. A newly
verified relationship crossing an original train/validation/holdout boundary
blocks this analysis; do not repair or regenerate frozen splits in this work.

Define assignment algorithm `validation-group-greedy-sha256-v1`:

1. Read only frozen metadata. Require unique record IDs, nonempty group IDs,
   homogeneous labels within each connected group, and at least three groups per
   class. Conflicting labels or insufficient groups block assignment.
2. For labels in N,A,O,~ order, sort groups by decreasing number of records,
   then ascending hex SHA256 of UTF-8 `20261013:<group_id>`, then group ID.
   The seed is a partition seed, independent of training seed.
3. Set fractional per-class targets to 0.40/0.30/0.30 times class support.
   Assign each whole group to the role with largest target minus assigned record
   count, with role order calibration-fit, threshold-select, dev-assess breaking
   ties. When remaining groups equal empty roles, restrict assignment to empty
   roles to guarantee class presence. Do not search seeds for nicer results.
4. Sort output by record ID. Record original split, role, label, duplicate group,
   verified grouping provenance, and subject_id=`unknown` where unavailable.
   Assert disjoint IDs/groups, exact union of 1,280 validation IDs, all 114 AF
   records assigned once, and all four classes in every role. Freeze counts and
   manifest SHA256 before reading logits by role.

Group integrity takes precedence over exact target sizes. Fit and select only
on their respective roles. Do not inspect dev-assess outcomes until all three
calibrators and thresholds are frozen. Any later changes require a new protocol
version and acknowledgement that this assessment has become development data.

## Calibration fitting

For each seed independently, fit a single positive scalar temperature on
calibration-fit only: `p_k = softmax(z / T)_k`. Minimize mean four-class negative
log likelihood, one equal weight per recording, using float64 stable logsumexp.
No class weighting, per-class calibrators, label correction, feature fitting,
or method search. A positive temperature preserves argmax.

Use inverse temperature beta=1/T bounded to [0.05,20], minimizing the convex
NLL in beta by deterministic golden-section search (ratio (sqrt(5)-1)/2), until
bracket width <=1e-8 or 256 iterations. Evaluate both original bounds, beta=1,
and the final midpoint too; choose lowest float64 NLL, with exact ties resolved
by closest beta to 1 then smaller beta. Save beta, T, objective, bracket,
iterations, convergence and boundary flags. Nonconvergence/nonfinite objective
blocks the affected calibrated arm; no silent fallback or retuning. Boundary
solutions are reported. T=1 is the prespecified uncalibrated comparator.

## Threshold selection and frozen comparisons

Confidence is `max_k p_k`; accept if confidence >= tau, otherwise defer for
confidence. The automatic class of accepted cases remains four-class argmax.
Select tau per seed on threshold-select only, with calibrated probabilities.
Candidate grid: exact fractions j/100 for integers j=0,...,100. At each tau,
compute accepted error fraction, coverage, AF coverage and automatic AF sensitivity.

Feasible thresholds must retain >=90% of all threshold-select records AND >=90%
of its AF-labeled records. These are proposed development coverage constraints,
not clinical performance targets. Among feasible thresholds minimize accepted
error fraction; ties prefer higher coverage, then smaller tau. Compare integer
ratios exactly for coverage and objective ties. Tau=0 is feasible for valid
complete inputs. Zero AF support or no feasible candidate blocks selection.
Do not optimize AF precision by discarding AF cases. AF coverage is retention,
not detection, and cannot guarantee sensitivity. Report sensitivity explicitly.

Freeze T, tau, hashes and the full selection table before dev-assess. Report
three prespecified arms per seed on the identical dev-assess cases:

| Arm | Probabilities | Decisions |
| --- | --- | --- |
| Raw baseline | T=1 | All cases assessed; no deferral |
| Calibrated baseline | Fitted T | All cases assessed; no deferral |
| Calibrated confidence policy | Same fitted T | Frozen selected tau |

Raw and calibrated baseline hard-label metrics should be identical. Calibration
is judged by probability metrics; confidence deferral by coverage/error/AF tradeoffs.
No additional threshold curves on dev-assess and no threshold updates after
assessment. All three seeds must pass input verification and have frozen calibrators and
thresholds before **any** dev-assess outcomes are inspected. A failed or missing
seed blocks assessment for every seed; do not proceed with a subset.
Report fitting and selection diagnostics separately from assessment,
never pool the three roles into a claimed out-of-sample result.

## Deferral semantics and complete accounting

Confidence deferral means insufficient model confidence. It is neither a negative
AF prediction nor evidence of poor signal quality. Predicted `~` is a four-class
noisy-class prediction, retained as an assessed outcome when accepted; it is
not a deferral and is not a validated signal-quality gate. True `~` has no known
underlying rhythm: do not invent AF labels or include it in the labeled-AF count.

Signal-quality deferral would require a separately specified signal-derived
quality measure and reviewed cutoffs. It is disabled in every arm here: count 0,
policy unavailable, quality metrics pending. Never call confidence or `p(~)` a
signal-quality measurement. No noisy-class routing to deferral is proposed.
Future quality gating needs a protocol amendment, training-only fitting where
applicable, selection-only tuning, and separate accounting before assessment.

For each role/seed/arm let E be all eligible recordings, C accepted recordings,
Dconf confidence deferrals, Dquality quality deferrals, and F unresolved technical
failures. Require E=C+Dconf+Dquality+F; here Dquality=0. Technical failures block
a complete assessment, remain explicit in an incomplete ledger, and never count
as predictions or clinical deferrals. Do not calculate completed-run headline
metrics after silently removing failures. E stays fixed even in failed runs.
Save reason flags and one primary disposition per case; quality precedes
confidence only if a future amendment enables it, avoiding double counting.

For A=all true-A records in that role, require:

`A = accepted_pred_A + accepted_pred_N + accepted_pred_O + accepted_pred_~
     + AF_confidence_deferred + AF_quality_deferred + AF_technical_failure`.

Report all seven counts. Accepted N/O are automatic non-AF rhythm decisions;
accepted `~` is a noisy-class outcome, not a normal/non-AF rhythm determination.
All three non-A predictions fail to detect labeled AF. Also report rhythm-output
coverage `accepted_pred_N/A/O / E` separately from four-class assessed coverage
`C/E`, so accepted noisy outcomes cannot inflate claims of rhythm assessment.

## Metrics and denominators

Report raw counts and numerator/denominator with every rate, by role/seed/arm.
For successful assessments:

| Metric | Definition |
| --- | --- |
| Assessed coverage / deferral rate | C/E; (Dconf+Dquality)/E |
| Reasons | Each primary-reason count/E and count/(Dconf+Dquality) |
| AF coverage / AF deferral rate | C_A/A; (Dconf_A+Dquality_A)/A |
| AF reason rates | Each AF reason count/A and count/(Dconf_A+Dquality_A) |
| Automatic AF sensitivity | accepted true-A predicted-A / **all A** |
| Accepted AF recall | accepted true-A predicted-A / C_A |
| Accepted AF precision | accepted true-A predicted-A / accepted predicted-A |
| Accepted accuracy / selective risk | correct accepted/C; incorrect accepted/C |
| Per-class precision, recall, F1 | TP_k/predicted_k; TP_k/accepted true_k; 2TP_k/(accepted true_k+predicted_k) |
| Four-class macro F1 | Mean of N,A,O,~ accepted F1, fixed four-class order |

Report accepted 4x4 confusion (rows true, columns predicted) with true and
predicted supports, plus an all-eligible 4x7 outcome matrix: predicted N,A,O,~,
confidence-deferred, quality-deferred, technical-failure. Row sums equal original
class supports. This exposes AF misses hidden by accepted-case recall.

Zero denominator means undefined (`null` plus reason), including A=0, no accepted
cases, no accepted AF, no AF predictions, or no deferrals. For compatibility with
existing reports also supply clearly named `legacy_zero_fill_macro_f1`, using
zero for undefined per-class F1; primary macro F1 is undefined if a component is.
Never display an empty denominator as zero performance or perfect performance.

Probability metrics on **all eligible role records before deferral**: mean
four-class NLL and unnormalized multiclass Brier score
`sum_k (p_k-1[y=k])^2 / E` summed over cases, and top-label ECE with 15 fixed
equal-width bins on [0,1]. Bins are left-closed/right-open except last closed;
ECE=sum_b n_b/E * abs(accuracy_b-mean_confidence_b), empty bins contribute zero.
Save bin edges, counts, successes and mean confidence. Also report binary AF
Brier `mean((p_A-1[y=A])^2)` and AF reliability only on true N/A/O records.
Use the existing four-class p_A without renormalization. Report the eligible
N/A/O count and excluded true-~ count explicitly, including when N/A/O support
is zero. True ~ has unknown underlying rhythm and must not become AF-negative.
Use the same bin edges on p_A with mean observed AF fraction. Retain noisy-label
cases in every four-class metric, coverage calculation and complete ledger. Accepted-only versions, if supplied,
are secondary and explicitly conditional. Use stable logits for NLL.

Report each seed and equal-weight mean/sample SD (n-1) of defined seed metrics,
with actual n, all missing/failed seeds and reasons. Never pool repeated cases
across seeds as independent samples. Report paired per-seed differences between
arms on the same dev-assess cases. No significance claims or seed ranking;
no bootstrap confidence intervals are prespecified. Small AF/noisy supports and
unverified patient independence limit interpretation.

## Provenance, artifacts and future execution boundary

Before future execution, freeze the reviewed protocol/YAML and implementation
revision, resolved configuration, grouping rule and derived manifest. Record UTC
timestamps, invocation, code revision plus dirty-state/diff hash, source bundle,
Python/platform and exact installed package versions, CPU float64 arithmetic,
two threads, algorithm versions, partition seed 20261013, and all model seeds.
Reuse existing dependency locks without modification. Record both active lock
hashes and distinguish the historical data dependency-lock identity from current
environment locks. Completed provenance supplies exact training versions and
the separate training/evaluation environment identities; preserve that distinction.

For each seed retain links/hashes of original YAML and resolved config, checkpoint,
training report, CPU evaluation report, source revision/bundle, preprocessing,
saved prediction CSV, frozen validation manifest, source audit/manifests and data
provenance. Verify published hashes before analysis; a copied hash is not a new
verification. Hash the new protocol/config and every derived artifact; record
parent hashes so each result traces to exact inputs and analysis choices.

Future trackable de-identified role manifest:
`data/manifests/temporal-calibration-deferral-v1/roles.csv`, with compact provenance
JSON alongside it. Keep protocol/YAML and compact aggregate report in `docs/`,
`configs/`, `reports/`. Keep local prediction paths out of public manifests when
they expose machine/user details. Proposed ignored, new, non-overwriting run root:
`outputs/calibration/temporal-calibration-deferral-v1/<execution-id>/seed{seed}/`.
Store input registry, calibration fit, selection table, frozen policy, per-case
probabilities/dispositions, reliability bins, confusion/outcome matrices, metrics
and verification logs there. Do not put ECG, checkpoints, predictions or full
local run outputs in Git. Existing artifacts are read-only; no output reuse.

Future verification must independently recompute metrics from the per-case ledger,
check every union/count identity above, verify all 114 validation AF records are
accounted for across roles, compare all three seed ID sets, confirm no group
overlap and no holdout signal access, and preserve hashes of protected inputs.
Full validation denominators are 1,280/114 only for combined role accounting;
dev-assess metrics use its own frozen E/A, whose exact counts are pending.
All fitted values, selected thresholds, derived hashes and outcomes remain pending.


## Numerical verification contract

Saved inference probabilities are float32 exports; analysis recomputes from
saved logits in float64. Require finite probabilities within [0,1]. Probability
sums match 1 with absolute tolerance 1e-6 and relative tolerance 0. Saved versus
recomputed softmax uses absolute tolerance 1e-7 and relative tolerance 1e-5
(`abs(saved-recomputed) <= atol + rtol*abs(recomputed)`). Baseline NLL and other
recomputed probability metrics, where a saved counterpart exists, use absolute
and relative tolerance 1e-5. Existing evaluation reports provide NLL, not saved
Brier/ECE values; absent historical probability metrics have no fabricated
comparison. All tolerances are fixed before real analysis. IDs, labels, hashes,
predicted classes, supports, confusion counts and accounting identities match
exactly; numerical tolerance never permits an argmax change.

## Supported commands and execution boundary

Use the existing CPU environment, without installing or changing dependencies:

```bash
.venv/bin/python scripts/manage.py analysis-preflight --config configs/temporal-calibration-deferral-v1.yaml
.venv/bin/python -m pytest -q tests/test_calibration.py
```

Preflight resolves the original training YAML through its recorded hash, derives
the launcher evaluation directory, authenticates training/evaluation report
hashes against the completed report, checks the evaluation's recorded YAML,
checkpoint/preprocessing/source/data identities and CPU provenance, then hashes
its adjacent `validation-predictions.csv`. Prediction exports had no historical
hash in that report: preflight records their actual hash now and verifies their
IDs, labels, logits, probabilities, confusion and baseline NLL. It does not claim
an independently recorded historical prediction hash. The frozen input registry
pins this new hash for later fitting and assessment. It does not load weights,
read signals, assign roles, write artifacts or print case-level predictions.

Later, **only after explicit user approval**, set `execution_authorized` to the
literal YAML boolean `true`, review and commit the protocol, YAML and implementation
with a clean source tree, and use a new execution ID. The current real YAML remains
`false`, so the commands below must reject execution until those prerequisites
are satisfied:

```bash
.venv/bin/python scripts/manage.py analysis-fit --config configs/temporal-calibration-deferral-v1.yaml --execution-id reviewed-v1 --authorize-analysis
.venv/bin/python scripts/manage.py analysis-assess --config configs/temporal-calibration-deferral-v1.yaml --execution-id reviewed-v1 --authorize-analysis
```

These commands are documented only; they have not been run on real data.
Both literal YAML `execution_authorized: true` and explicit `--authorize-analysis`
are required. The CLI flag cannot override false, missing, null, numeric, string
or other invalid YAML values. Authorization is checked before loading analysis
inputs, assigning roles, fitting, threshold selection or output writes. Read-only
preflight remains available with YAML false and without the CLI flag. The flag
is recorded in the invocation and does not modify YAML. Fit creates a new run directory, a de-identified role
manifest (or verifies identical existing bytes), role-local diagnostics and all
three policies. It writes `frozen-policy.json` last, only after all seeds succeed.
Partial runs remain preserved and cannot be overwritten or assessed.
Assessment permits only the generated role manifest and its provenance file to
be uncommitted; it verifies their frozen hashes. All source/configuration changes
still block assessment. This avoids a source revision change between fitting
and assessment merely to commit generated metadata.

Assessment reruns read-only input verification, validates identical config,
protocol, manifest, implementation revision and frozen artifact hashes, and
checks the three-seed gate before computing dev-assess outcomes. It creates a
new assessment directory; repeated assessment fails rather than overwriting.
Reports are recomputed from persisted per-case ledgers. Each stage records
package versions and source/input/output identities. The current implementation
uses only frozen exact duplicate groups. If additional verified relationships
are supplied, block for review rather than silently ignoring them; transitive
closure support needs a reviewed metadata source before use.

Synthetic tests create their own logits, probabilities, labels and group metadata
in temporary directories. They do not use ECG data or derive a real role manifest.
Real fitted values, thresholds, assessment outcomes and their hashes stay pending.
