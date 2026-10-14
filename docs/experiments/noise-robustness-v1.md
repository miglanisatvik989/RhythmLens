# Noise robustness v1 — frozen development proposal

Status: **proposal only, unsupported declarative configuration, execution
unauthorized, implementation/results pending**. This PR freezes reviewable method
choices; it does not implement a runner or authorize acquisition, training or
inference. `configs/noise-robustness-v1.yaml` has literal
`execution_authorized: false`. Do not pass it to any launcher action.

Implementation follow-up: [NSTDB acquisition/audit and adapter](../nstdb-adapter.md)
now supplies the separate noise-only CLI and library primitives. The proposal
YAMLs remain unsupported and false; Phase A/B runner, real acquisition/audit
and results remain pending. Original method choices below are unchanged.

Goal: test whether noise augmentation, calibrated uncertainty and separate quality
assessment can reduce incorrect automatic rhythm predictions while retaining AF
detections and manageable review workload. Future confidence-only versus quality
policies must be compared at matched assessed coverage **and** reported rhythm-output
coverage. Deferred AF cases are not detections. Quality-model training, confidence
retuning, policy integration and UI implementation are outside this proposal.

The original Challenge holdout is **consumed and excluded** from all new
augmentation, stress testing, checkpoint selection and policy development. Existing
results cannot be final validation of future quality-aware triage. All Phase A/B
outcomes here are development evidence on reused validation recordings, not a new
independent test. No new cohort, holdout exclusions or split repairs are proposed.

## Candidate and inputs

Keep all three temporal `rhythm-dilated-v1` checkpoints equally, seeds 42/43/44,
epochs 31/42/39, original weights/preprocessing and all baseline provenance from
[the multiseed report](../../reports/rhythm-multiseed-v1.json). Opaque hashes were
checked during proposal preparation; no checkpoint was loaded. Phase A requires
all three models. No seed ranking, ensemble, weight averaging or replacement.

Rhythm input remains Challenge 2017 1.0.0 / REFERENCE-v3, native physical mV at
300 Hz. Train: 5969; validation: 1280, N=762/A=114/O=362/~=42. Use original verified
record/duplicate groups and frozen train/validation membership. Patient IDs are
unknown; no subject inference is permitted. Newly verified cross-boundary linkage
blocks work pending a new reviewed development plan; never rewrite frozen files.

Noise input is NSTDB 1.0.0 noise-only `bw`, `ma`, `em`, both channels (0 and 1),
not its already-corrupted 118/119 ECG examples. No BUT records enter rhythm training
or stress metrics. BUT quality labels and native synthetic corruption parameters
have different meanings; neither supplies Challenge rhythm labels. See
[dataset readiness](../../reports/dataset-readiness-noise-quality-v1.md) for
versioned URLs, header evidence, annotation semantics and unresolved audits.

## Noise pools, resampling and reproducible case plans

These are **proposed development choices**, fixed before acquisition/outcomes:

| Role | Original noise time interval, seconds | Usage |
| --- | --- | --- |
| augmentation-train | [0,600) | Phase B training corruption only |
| development-reserve | [660,1140) | Reserved; not used for selection or policy fitting in v1 |
| stress | [1200,1800) | Phase A and Phase B validation stress, with identical maps |

Apply these boundaries to both channels of each record; unused gaps and tail stay
unused. Audit actual lengths, checksums and channel inventory before freezing a
noise-role manifest. No pooling, repetition, wrapping or substitution if a record
is too short or an expected file/channel is unavailable. Temporal pools share
source recordings; 60-second gaps do not prove subject/source independence or
remove long-range correlation. Noise volunteer identities are unavailable.

Convert noise digital samples to float64 shape values, subtract ADC zero from
metadata, then slice each role independently **before** resampling. Resample each
pool from 360 to 300 Hz using the existing SciPy 1.15.3 contract:
`resample_poly(up=5, down=6, window=('kaiser',5.0), padtype='constant', cval=0)`.
Trim 300 output samples per end. Never let filter support borrow another role's
samples; log pool input bounds, filter parameters and trim. This specifies a new
adapter; it does not change the Challenge decoder or preprocessing contract.
[Official resampling documentation](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.signal.resample_poly.html).

For a recording with L native samples, select a contiguous L-sample output slice
from a trimmed eligible noise pool. An offset is an integer relative to that
trimmed pool in [0,pool_length−L], inclusive. No resampling of Challenge ECG,
length change, cycling, concatenation or cropped rhythm labels. For each selected
slice, subtract its arithmetic mean and compute float64 RMS. Zero/nonfinite noise
RMS blocks that case/run; do not search for a more favorable offset.

Freeze case-plan metadata before model access: record ID, original duplicate group,
original split/label, condition, repetition, noise release/hash/record/channel,
role input bounds, resampled-pool hash, output offset/length, requested SNR,
key/seed, and eventual amplitude/RMS validity fields. Do not put signal samples or
predictions in trackable manifests. Repetitions and seeds use the same records,
not new independent people. Reused source intervals remain explicitly identifiable.

Stress plan: repeats 0/1/2; SHA256 of compact UTF-8 JSON list
`[20261015,"stress",record_id,noise_record,repeat]`, with no whitespace and no
Unicode escaping changes. Interpret first eight digest bytes little-endian as a
uint64 seed for a dedicated NumPy Generator(PCG64). Draw channel with
`integers(0,2)`, then offset with `integers(0,pool_length−L+1)`. Do not include
model seed, training arm or severity in this key: all checkpoints/arms/severities
see the same underlying noise shape within a paired case/type/repetition.
No global NumPy/Torch RNG is consumed by corruption planning.

## Physical scaling, flat cases and preprocessing order

Let x be the original 300 Hz ECG in mV; c=x−median(x),
`r_x = sqrt(mean(c**2))`. For the selected mean-centered noise slice v,
`r_v = sqrt(mean(v**2))`; define unitless `u=v/r_v`.
For requested SNR s in dB, add
`n = u * r_x * 10**(-s/20)` mV, then use `x_corrupt=x+n`.
Define achieved SNR as `20*log10(r_x / RMS(n))` **before** model preprocessing.
Require finite values and absolute achieved-minus-requested difference <=1e-6 dB.
Record actual added RMS, peak amplitude and achieved SNR. No clipping, denoising,
per-record amplitude normalization, extra bandpass or adaptive intensity change.

This is centered whole-record RMS SNR, a **project development definition**.
The official NST tool uses a QRS-based signal measure and weighted noise power;
our values are not a replication of its calibrated benchmark SNR. No clean-QRS
annotations or clinical noise thresholds are inferred.
[Official NST SNR definition](https://physionet.org/physiotools/wag/nst-1.htm).
The source noise headers' zero gain means native voltage calibration is unknown;
unit-RMS normalization makes absolute source gain unnecessary, not verified.

If `r_x < 1e-8 mV` (the existing preprocessing epsilon), retain the case and its
label in every condition; use added noise zero, record requested SNR and achieved
SNR `null` with reason `flat_or_near_flat_reference`, and flag `corruption_applied:
false`. This is a prespecified inapplicable-SNR case, not quality deferral or a
negative AF prediction. Report its count and unchanged outcomes separately; do
not claim the requested SNR was achieved or silently remove it. Nonfinite ECG,
overflow or invalid length is a technical failure, not this flat policy.

Add noise in physical mV **before** the unchanged per-record median centering and
fixed training RMS division. Pass corrupted or unperturbed physical arrays to the
original transform; record pre- and post-transform digests locally. Reuse each
checkpoint's saved training-only scale 0.18008390945457742 mV, preprocessing SHA256
`a4719420f5b867e3870386b488c954fb85f1b695f2264f57235cf0e0b80783b6`.
Never fit a scale on validation, corrupted variants or noise, and do not replace
this state in Phase B. Scalar signal RMS used solely for casewise corruption is
not a learned normalization state and uses no labels or model outcomes.

## Phase A — frozen-checkpoint validation stress

One unperturbed control and 3 noise types × 6 levels × 3 repeated offsets per
validation record: **55 conditions**, each with all 1280 eligible records and all
three fixed models. Proposed levels: **24,18,12,6,0,−6 dB**, drawn from the official
[NSTDB example grid](https://physionet.org/files/nstdb/1.0.0/nstdb.txt).
Applying that grid to this RMS definition, noise types and recordings is our
proposed development stress design, not a source claim of clinical severity.
Report every level, including extreme corruption; never select nicer levels or
noise channels after outcome inspection. No result is available yet.

CPU float32 inference, float64 corruption/scoring, two threads, batch size 16,
original validation length-bucket/no-shuffle batching and padded-prefix masks,
evaluation/inference mode, first-in-N,A,O,~ argmax tie order, raw softmax T=1.
Verify unperturbed outputs against saved CPU validation exports (exact IDs/labels/
argmax/confusion; logits/probability agreement atol 1e-7, rtol 1e-5; probability
sums atol 1e-6). Missing exports or mismatches block completion, not regenerate
historical outputs or loosen tolerances. Validate this numerical contract
synthetically before implementation is considered ready.

## Phase B — baseline versus augmentation

Report existing original temporal training runs as the unaugmented reference;
no baseline retraining or historical artifact replacement is required. Train
**three new augmented temporal runs**, seeds 42/43/44, on the same frozen train
records; validate on the same **clean** validation records using the original
macro-F1 early stopping rule. Original runs/configurations remain immutable.
Each original training YAML's architecture, optimizer, LR, weight decay, dropout,
epoch cap, patience, batch/sampler behavior, device profile and training seed stay
fixed. Reuse the verified clean-train preprocessing state in both arms. Name new
runs `temporal-noise-v1` in a distinct ignored output root; never reuse `temporal-v1`.

At each epoch/record encounter, independently choose corruption with probability
0.5. If selected, choose uniform noise type bw/ma/em, uniform channel 0/1, and
uniform SNR from **24,18,12,6 dB**, then a uniform contiguous offset from the
training pool. Extreme 0/−6 levels are stress-only in this proposal. No mixtures
or severity curriculum. These probability/grid choices are **proposed development
choices**, not clinical or optimal settings. No tuning from Phase A/B stress.

Dedicated RNG seed = first eight SHA256 bytes, little-endian, of compact JSON
`[20261014,"train",model_seed,epoch,record_id]`; epochs start at 1. Draw PCG64
`random()` first for the p<0.5 decision, then (if selected) integer type index in
[bw,ma,em], channel, severity index in [24,18,12,6], offset, in that order. This
must not alter Torch dropout/sampler RNG streams. Persist per-epoch corruption
plans/digests for reproducibility and resume. Cache clean physical inputs only
for augmentation; do not reuse one corrupted/transformed sample across epochs
or add noise to already normalized cached tensors. Original seed does not guarantee
identical optimizer trajectories once training inputs change.

After all three augmented checkpoints freeze on **clean validation macro F1**,
evaluate both original and augmented checkpoints on the same Phase A clean/stress
plans. Reports pair seed, source recording, type, level and repetition. Training
noise cannot overlap stress support; development-reserve is unused. No selecting
checkpoints by stress outcomes or swapping the best of three seeds. This remains
validation-reused development evidence, including checkpoint selection bias and
historical runtime/source differences between original and new runs. Record them;
do not imply a new randomized independent trial or promise identical baseline
reruns. Any implementation/lock/numerical change needs disclosure before execution.

## Label preservation and estimands

Adding artifact preserves the **source rhythm label as an experimental assumption**,
not guaranteed recognizability. Severe corruption may obscure AF or QRS completely;
label remains source truth, while visible diagnostic adequacy is unverified.
Do not relabel to ~ based on SNR or invent AF/non-AF truth for naturally noisy
recordings. Keep original ~ cases in four-class accounting; binary AF probability
metrics exclude them. Report original-label ~ strata separately from synthetic
corruption metadata, and flat-SNR-inapplicable cases explicitly.

For every seed/arm/condition report E=1280, A=114, accepted, reasoned deferrals,
technical failures and a complete eligible ledger plus 4×4 confusion/4×7 outcome
matrix. Every ID occurs exactly once. Assessed four-class coverage C/E is distinct
from automatic N/A/O rhythm-output coverage. Accepted predicted ~ is assessed,
not a normal/non-AF rhythm determination, and not a deferral. Calibration,
confidence deferral and quality gating are disabled in **both phases**: expected
policy counts/rates zero; technical errors block completion and never reduce E/A.

Prespecified metrics: four-class accuracy and incorrect automatic predictions/E;
per-class TP/FP/FN, support, predicted count, precision, recall and F1; four-class
macro F1 (zero F1 if true+predicted support zero, flag the class); false AF =
true N/O predicted A (count/(true N+O), plus count/predicted A where defined).
Also show true ~ predicted A separately; do not pretend its underlying rhythm is
non-AF. Automatic AF sensitivity = accepted correct A / **all labeled A**;
report misses by predicted N/O/~ and any deferred/failed AF terms. Undefined
precision/recall/rates use null plus reason; A=0 means undefined sensitivity.

Probability suite on all eligible complete cases: stable-logit NLL, multiclass
Brier sum across classes without dividing by four, 15-bin equal-width top-label
ECE/reliability. Binary AF Brier/reliability use original p_A, true N/A/O only,
exclude true ~ without renormalization, and show denominators. Coverage, mean/max
review workload (policy-deferred/E, currently zero), noisy outcomes and runtime
are reported separately. No fitted temperatures, threshold curves or confidence
policy retuning occur in v1.

## Pairing and uncertainty

Show each model seed equally and equally weighted mean/sample SD (n−1), never
pooled seed copies or a best-seed headline. For each type/level, show repetitions
separately plus a within-record equal repetition mean of case-level errors and
AF-detection indicators; compute mean metric/delta across repetition-specific
confusions for macro F1 (do not invent a consensus/ensemble label).
Report corruption-minus-control paired deltas and Phase B augmented-minus-original
paired deltas on identical cases/maps. Reference control once, not repeated
independent clean observations. All noise draws/severities remain linked to the
same underlying source recording.

Prespecify 10,000 class-stratified frozen duplicate-group bootstrap draws,
PCG64 seed **20261016**, class order N,A,O,~, sorted group IDs. Sample each class's
original group count with replacement; retain every member and **all** associated
arms/seeds/conditions/repetitions together. Require homogeneous group labels.
Use the same multiplicities for all comparisons, recompute each repetition's
metric then average repetitions/seeds within each draw; 95% percentile intervals
use `np.quantile(method='linear')`. Report intervals for macro F1, automatic AF
sensitivity, automatic error and their paired deltas; no silent failed-draw
replacement. Zero AF yields null/no AF interval. Save draws/group metadata and
statistics locally. No p-values, significance claims or multiplicity-adjusted
conclusions across this exploratory grid.

Intervals condition on reused validation class mix, fixed seeds and fixed noise
plans. They do not account for unknown patients, noise-source dependence, new
noise realizations, training-data sampling or a full seed distribution. Do not
bootstrap variants, windows, repetitions, or seed copies as independent recordings.
Shared duplicate groups are the minimal verified unit, not invented patients.

## Provenance, lifecycle and preservation

Implementation is pending; existing launchers support none of this new schema.
Before any later execution, implement and synthetically verify the actual parser,
noise adapter, pairing, unit/resampling/scaling/flat behavior, role boundaries,
training-RNG isolation, case accounting and failure rules. Freeze local source
hashes, original candidate input hashes, approved new dataset/audit/role manifests,
resolved configs, case plans and exact environment versions. Never install or
upgrade under evaluation authorization. Missing input hashes are pending today,
not inferred from official filenames or silently filled during a real run.

Proposed distinct ignored roots are
`outputs/noise-robustness/noise-robustness-v1/phase-a/` and `phase-b/`.
Require absent root and exclusive lifecycle creation; no overwrite/resume of a
completed run, automatic retry, changed ID to bypass collision or selective
completed-seed report. Future training resume, if implemented, must be explicit,
validate identical config/RNG/epoch corruption-plan hashes and preserve every
partial artifact; no resume implementation is claimed here.

Save original/resolved config, protocol/source/input hashes, seeds, environment,
invocation/time, case plans and audit log before work. Verify all inputs before
model access. Missing/corrupt files, linkage/roles, nonfinite values, decoder/unit
mismatch, changed configs/locks, incompatible checkpoints, output collisions or
unauthorized execution stop the stage. After signal access, retain every eligible
case with explicit unresolved/failed reasons; stop completed-run metrics on any
technical failure. Phase A requires all three seed/condition predictions sealed
before scoring; Phase B stress requires all six original/augmented checkpoint
outputs sealed before comparisons. Preserve partial lifecycle and report blocked
status rather than resampling noise or dropping cases.

Compact configs, de-identified role/case-plan metadata and aggregate reports may
be tracked. ECG, noise arrays, resampled caches, augmented samples, weights,
per-case logits/ledgers and bulk bootstrap outputs stay ignored. Include final
artifact SHA256 indexes/completion records locally. Durable preservation is
**unverified and owner destination pending**; no upload/archive is performed by
this proposal. Git documentation and hashes are not restorable output backups.

## Scope boundaries and owner decisions

This PR contains documentation/configuration drafts only. Results, source hashes
for a future runner and acquisition manifests are pending. Existing authorizations,
source numerical configs, decoder contracts, splits and local artifacts remain
unchanged. Configuration false plus unsupported schema is not runnable authority.

Owner decisions before future work: approved archival destination and retention/
restore checks; acquisition scope; compute/time budget; AF sensitivity versus
review-workload targets; BUT window/consensus usability task; quality transfer
assessment and matched-coverage selection protocol; reviewer interface scope;
and independently sealed final cohort/custodian. The numerical development choices
above are reviewable defaults, not claims that owner budgets or clinical targets
have been supplied. Changes require a versioned proposal before the affected
experiment, not retrospective tuning against stress or consumed holdout outcomes.
