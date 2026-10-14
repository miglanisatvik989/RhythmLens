# Temporal rhythm-only internal evaluation v1

Status: **dedicated executor implemented; synthetic verification recorded separately; real execution unauthorized**. The executable configuration is [temporal-final-evaluation-v1.yaml](../../configs/temporal-final-evaluation-v1.yaml), with literal `execution_authorized: false`. The existing calibration YAML remains unchanged; its validation-analysis authorization does not authorize this evaluation. No holdout signals, predictions or performance were loaded/generated during this implementation task.

This is a **qualified internal evaluation of the frozen rhythm-only baseline**, using all three temporal checkpoints equally. It is **not final validation of the eventual quality-aware system**. The historical filename and `final-*` command names identify the dedicated one-time executor, not a stronger scientific claim. **Inspecting these outcomes consumes this holdout for future development.** Subsequent quality-gating changes need separate evaluation evidence; these rhythm-only results cannot validate those changes.

## Integrity gate and scope

Read the [integrity audit](../../reports/final-evaluation-integrity-audit-v1.md) before approval. The existing recording holdout has no recorded model-performance evaluation, but all its signals were decoded before split freeze and two of its records had pre-freeze previews generated. Holdout labels/supports were available. Patient/source/time independence is unverified. Thus this cannot be called a strictly never-seen, patient-independent test.

The present implementation targets the qualified first recorded internal recording-holdout evaluation. A strict untouched final test instead requires a newly sealed independent cohort under an amended protocol. No new cohort, split, exclusions or patient identities are created by this document. Discovery of prior holdout-performance tuning or verified cross-boundary subjects/duplicates blocks this internal evaluation, pending a reviewed disposition.

## Fixed rhythm-only candidate

User-directed candidate: `rhythm-dilated-v1` (TemporalCNN), using all three existing best checkpoints equally. No seed selection based on development-assessment or final performance; seeds 42/43/44 are mandatory. No ensemble, weight averaging, checkpoint replacement, refitting or retraining. Epoch choices already came from validation macro F1 and early stopping and are disclosed as development selection.

Architecture: single physical-mV channel at native 300 Hz; stem 16 channels/kernel 15; eight two-convolution masked residual blocks with kernel 7, stride 1 and dilations 1,2,4,8,16,32,64,128. Block output widths are 16,24,32,32,32,32,32,32. ChannelNorm/ReLU and masking follow the existing implementation; masked global averaging and a four-class linear head retain N,A,O,~ order. Training dropout was 0.1; model evaluation mode disables it. 97,548 parameters; architectural receptive field 3,075 samples. No Monte Carlo dropout.

Preprocessing: `median-fixed-training-rms-v1`, native 300 Hz physical mV, per-record median centering followed by the fixed training RMS scale **0.18008390945457742 mV**; median RMS fitted on all 5,969 train records only. No resampling, per-record amplitude scaling, holdout fitting, normalization changes or signal-quality exclusions. Retain flat/noisy eligible signals. Reuse each checkpoint’s matching saved preprocessing state.

Checkpoint paths below are under `outputs/rhythm/temporal-v1/cuda128/`; all actual checkpoint-file hashes were checked against the completed multiseed report during metadata/provenance preparation, without model loading.

| Seed | Selected epoch | Checkpoint relative path | Checkpoint SHA256 |
| --- | --- | --- | --- |
| 42 | 31 | `seed42/train/best.pt` | `217110e48c995fcb8eac9a7eb3885356170fd461dc50b333cd62fd1022d2031d` |
| 43 | 42 | `seed43/train/best.pt` | `c420c2a96ecf1e9f25ad4a7417a321a96b9eb47c217db613d9502c26e2bcde74` |
| 44 | 39 | `seed44/train/best.pt` | `e6965190d29b0214a732f6bf5ef4172daaae3ede30b2e549c8d7c9760a8791c5` |

| Seed | Original configuration | YAML SHA256 | Training source revision |
| --- | --- | --- | --- |
| 42 | `configs/temporal-cuda128.yaml` | `df2bc25e8fc7fd62654cde86efd2fe4946ecab43cdd698f2f6cff364e590503b` | `cc0317e8389895ce094b5521b6f40986ac6ae5ed` |
| 43 | `configs/temporal-cuda128-seed43.yaml` | `00f07878fc338420eb7e2d7708aa3e8addea73874d5436148f0da7e989e573ac` | `05d8cebf4af570aaf2ab40d80c7c715b573be534` |
| 44 | `configs/temporal-cuda128-seed44.yaml` | `a931d40ab2ea9fdae0e3508b4b4c555f0c9744eb2c0e14a8493f5032bee1f3bb` | `05d8cebf4af570aaf2ab40d80c7c715b573be534` |

Preprocessing JSON SHA256 (all seeds): `a4719420f5b867e3870386b488c954fb85f1b695f2264f57235cf0e0b80783b6`. Training-manifest SHA256: `3f249df84acbbd253ca8fdc73c29fc74c5d84f04aa08e3e4aa3103ebbd51e580`. Fitted-training-ID SHA256: `1012e38e8d2cbf3b323eb73ffc60ebf101ed71d7f5dfdbc2577ed512c756d803`.

Protected numerical-source identities at drafting (the dedicated executor adds code, but must preserve these numerical modules and approved decoder semantics):

| File | SHA256 |
| --- | --- |
| `src/rhythmlens/rhythm_model.py` | `0ff8a9e4616c9c278a26bd0fb9a7442e7d4787a7bf0ca1fb2c8f173812c27160` |
| `src/rhythmlens/preprocessing.py` | `49886756d95ef91728ab903a60218bee52cf4776dbdfff13acb3d7100e3dd963` |
| `src/rhythmlens/rhythm_inference.py` | `b82da939a8b811fd5a4371f514b004cc82c24d6f97785b0e4b51abf8107fc514` |
| `src/rhythmlens/data.py` | `241da71767049ef6eec915939afc6105a6e5bd79c413ab103a6d85febbd505fc` |
| `src/rhythmlens/audit.py` | `c252876c201f18258b21762f6c0f3e558d52916c1eb9eb0fde078255dd29a6f5` |
| `src/rhythmlens/data_compatibility.json` | `d9af1932fe734e375f70713d4ee88b53b985864dc9a63a880470fe05b47dc2ce` |
| `uv.lock` | `71231d200792985f74f8a2c9ebff633621ad86d6efcc59b2b0063b5a2301c86d` |
| `environments/cuda128/uv.lock` | `2c5ed7cb262d94c1e2d334af9955de796ddf43e8d8be73f0c32976cdc75474a1` |

The [completed multiseed provenance](../../reports/rhythm-multiseed-v1.json) remains authoritative for original resolved configurations, source bundles, training/evaluation reports and exact training dependencies. It must match actual inputs at preflight; absent checkpoints must not be regenerated.

## Decision and eligibility rules

The dedicated executor uses the original holdout, subject to the existing execution authorization: `splits/challenge2017-v3-seed42-v1/holdout.csv`, 1,279 eligible records, N=761, A=114, O=362, ~=42. These are existing published metadata counts, not performance results. Holdout manifest SHA256: `d7aad3212e0052f33ad86666a3844dee146f99cc584c972854caf91b331b0717`; assignment SHA256: `fb27f37a8f9dcd1d8931116e252290a68be29844b5b81badc3223954184054c7`. Keep the original frozen holdout unchanged, including the two preview-exposed IDs; do not silently produce a cleaner-looking subset.

Inference is ordinary frozen four-class logits/softmax at T=1, argmax with first-in-N,A,O,~ tie breaking. **No calibration, confidence threshold, confidence deferral, signal-quality gate, noisy-class routing, AF-specific threshold or postprocessing tuning.** Predicted ~ remains an assessed four-class outcome; its underlying rhythm is unknown and it is not an automatic non-AF rhythm determination. True ~ remains included in four-class metrics and eligible accounting. No human override of predictions.

Fixed execution runtime: existing locked CPU environment, Python 3.11.16, Torch 2.8.0+cpu, NumPy 2.2.6, two CPU threads, saved numerical inference dtype float32, model.eval/inference mode, no shuffle, canonical recording-ID order and existing padded-prefix masks. Batch size 16. Do not convert weights to float64, quantize or change batching after outcome inspection. Score probability metrics in float64 from saved logits; record actual runtime/platform/installed package versions. Exact cross-hardware identity is not promised. Missing or mismatched dependencies stop execution rather than installing/upgrading them under final-evaluation authorization.

## Primary estimands, denominators and complete accounting

Co-primary outcomes are four-class macro F1 and automatic AF sensitivity. Report each of the three seeds and their equally weighted arithmetic mean; mean is across seeds, never across pooled repeated case copies. Report sample SD with n−1 denominator as descriptive training-seed variability. No winner, ensemble or primary metric switch after evaluation.

| Metric | Fixed definition |
| --- | --- |
| Four-class macro F1 | Mean of N,A,O,~ F1; F1_k = 2 TP_k/(true support_k + predicted count_k), over every eligible case in a complete run. If both supports are zero, assign F1=0 for baseline compatibility and flag that class. |
| Automatic AF sensitivity | Correct automatic A predictions / all A-labeled recordings (114 for this original holdout); all incorrect predictions and any unresolved/failed AF cases are not detections. Undefined if no AF labels. |
| Accuracy | Correct four-class predictions / all eligible cases (1,279 for the original holdout). |
| Per-class precision / recall | TP_k/predicted count_k; TP_k/true support_k. Report numerator/denominator, support, F1 and null with reason for zero precision/recall denominator; separately name any legacy zero-filled output. |
| Coverage / deferrals | Accepted/E; confidence, signal-quality and other policy-deferral counts/rates must all be zero for the completed raw policy. Distinguish technical failures from deferral. |
| Rhythm-output coverage | Predicted N/A/O count / all eligible cases; accepted predicted ~ counts remain separate. |

Save a 4x4 confusion matrix per seed, rows true and columns predicted, with fixed N,A,O,~ order, plus a complete eligible ledger/outcome matrix exposing technical failures and zero deferral columns. Every ID appears exactly once per seed. Require `E = accepted + confidence_deferred + quality_deferred + technical_failure`, with zero deferrals and accepted=E for a completed run. For AF require `A = predicted_N_on_A + predicted_A_on_A + predicted_O_on_A + predicted_~_on_A + AF_deferred + AF_technical_failure`. Report all terms; ~ on true A is a miss. Technical errors never become negative AF predictions or silently reduce E/A.

Secondary fixed probability metrics: mean four-class NLL, multiclass Brier (sum across classes, no class-count division), and top-label ECE/reliability with 15 fixed equal-width bins on [0,1], left-closed/right-open except final bin closed. Mean confidence/accuracy/count per bin; empty bins contribute zero ECE. Binary AF Brier/reliability use only true N/A/O, original p_A without renormalization, and explicit eligible/excluded-~ counts (expected 1,237/42 for this original holdout). No new calibration is fitted. Primary and secondary metrics remain pending until separately authorized evaluation.

## Uncertainty and repeated-seed interpretation

Prespecify **10,000 class-stratified duplicate-group bootstrap replicates**, NumPy Generator(PCG64(20261014)), canonical label/group order N,A,O,~ and ascending group ID. For each replicate and each true class, sample its existing duplicate groups with replacement using the original number of groups in that class and retain every member of a sampled group. Require homogeneous labels within groups. In the current holdout all 1,279 groups are singletons, so this is class-stratified recording resampling with fixed observed class supports. Use exactly the same sampled groups/multiplicities for all three seeds. Do not resample seeds or treat their predictions as independent patients.

For each replicate compute the two primary metrics per seed and their equally weighted three-seed means. Report per-seed and mean-estimand 95% percentile intervals using np.quantile with method="linear" at 0.025/0.975. Preserve bootstrap seed, version, group manifest and replicate statistics locally. Also report the descriptive seed mean/sample SD. These intervals condition on the observed class mix and the three fixed checkpoints; they do not account for unknown patient clusters, source shifts, training-data sampling or the full training-seed distribution. They can be overconfident if recordings share subjects. No patient-level confidence interval, significance test, multiplicity-adjusted claim or CI-based seed selection is justified.

Zero-A strata imply undefined AF sensitivity; report null and no AF interval. Do not synthesize identities/strata, replace failed bootstrap draws or change resampling method after inspecting results. Any numerical/bootstrap failure blocks that uncertainty report and is disclosed; no silent fallbacks. For a newly verified subject-grouped external cohort, revise and freeze the resampling unit/strata *before* access rather than reusing these recording-only intervals as a patient-independent claim.

## Execution gate, failures and one-time reporting

The existing `evaluate` launcher remains validation-only and `FrozenRecords` still rejects holdout. The dedicated [executor](../../src/rhythmlens/final_evaluation.py) uses a separate authorized path, reusing split/source integrity checks and the original decoder, preprocessing, model and padded-mask implementation. It does not route holdout through calibration commands. [Synthetic verification](../../reports/temporal-final-evaluation-v1-verification.md) exercises fresh synthetic checkpoints and generated ECG only.

Before real execution, review the exposure/independence audit and the restricted internal claim; freeze and commit the protocol, evaluator, tests, documentation and configuration with literal `execution_authorized: false`; then separately obtain approval for the existing narrowly scoped true-authorization change and explicit runtime authorization. A clean committed tree and both authorization signals are required. Existing calibration authorization grants none of this. The executor verifies original input hashes/checkpoints/preprocessing/locks and records the execution revision plus hashes of every committed file. No additional approval mechanism is introduced.

The commands use the existing installed CPU environment and never set up or install dependencies:

```bash
# Read-only metadata/opaque checkpoint hash and offline runtime preflight; no signal loading.
.venv/bin/python scripts/manage.py final-preflight --config configs/temporal-final-evaluation-v1.yaml
# Only after the separately authorized, committed one-line change below:
.venv/bin/python scripts/manage.py final-evaluate --config configs/temporal-final-evaluation-v1.yaml --authorize-evaluation
```

Exact future authorization diff, confined to `configs/temporal-final-evaluation-v1.yaml`:

```diff
-execution_authorized: false
+execution_authorized: true
```

Keep the present working configuration false. The executor rejects changes to any other frozen configuration choice. Metadata preflight is available with false authorization; it does not create output directories or deserialize checkpoints. It is not a signal-file readiness check. Authorized execution verifies all audited signal-file hashes before prediction and decodes each case only during inference. Missing dependencies, inputs or checkpoints block execution; no installation or regeneration follows.

Fixed ignored output root (not created during implementation): `outputs/final-evaluation/temporal-raw-v1/internal-holdout-v1/`, with seed42/seed43/seed44 subdirectories. Before prediction require the root absent; create it exclusively with a lifecycle record before signal-file access. No execution-ID override or resume exists. Save original/resolved final config, protocol/audit/candidate hashes, immutable input registry, case IDs, environment/versions/invocation and audit log. Generate and seal predictions for **all three seeds before opening any model-performance summary**, then score/report the complete frozen suite once. A prediction-stage failure blocks scoring for all seeds. Labels already exist in metadata, so this staged scoring is an operational safeguard, not retrospective blinding.

Expected artifacts: per-seed saved logits/probabilities/predictions and dispositions, confusion/outcome matrices and full metric reports; common accounting and bootstrap manifests/statistics; aggregate per-seed/equal-weight summaries; immutable completion record and full artifact SHA256 index. Per-case signals, outputs and weights stay ignored; only compact de-identified provenance and final aggregate reports are trackable. There is no repository output-archival mechanism; an approved durable-preservation destination/workflow remains a prerequisite for claiming complete external archival. Do not invent a destination or add bulk outputs to Git.

Stop before prediction on missing/corrupt input, checkpoint/config/preprocessing/lock mismatch, changed source/protocol, unexpected manifest/group/subject linkage, output collision, invalid authorization, dirty source, unresolved integrity concerns or changed eligibility. Do not download, refit, substitute a seed or drop records. After execution starts, any decode/nonfinite/inference/missing-prediction/ID mismatch is a technical failure: retain the original E/A and failed-case reasons, mark run incomplete and stop all scoring/headline reporting. Preserve the partial directory and lifecycle state. No automatic retry, overwrite, execution-ID change to bypass collisions, checkpoint replacement or selective completed-seed report. Investigating a failure uses a separately reviewed incident decision and labels any further run as a repeat.

Inspecting these outcomes consumes this holdout for future development. After metrics are revealed, do not tune any architecture, checkpoint/seed, preprocessing, threshold, gate, label mapping, exclusions, uncertainty method or primary estimand against them. Any future changes are a new development cycle needing new independently sealed evaluation data. The original frozen splits and every failed/completed artifact stay intact. This protocol names one internal holdout root; a fresh-cohort route needs an explicitly reviewed separate cohort identity/root and cannot overwrite this plan silently.

## Generalization limits and next review

The fixed temporal choice follows development evidence and the user’s instruction to use raw predictions; no seed is chosen from reviewed-v1 performance. Architecture and checkpoints were already selected using validation, and reviewed-v1 outcomes have been observed. This cannot be marketed as prospectively independent model selection. Existing holdout labels and signal-derived integrity information were available before evaluation, with two pre-freeze preview exposures. No available access log proves complete blindness.

If the existing cohort is approved, the result will be a first recorded internal evaluation on disjoint recording IDs and exact-duplicate groups from a single public source. It cannot establish patient-independent, temporal, cross-source, external or clinical generalization, safety, or deployment readiness. A genuinely stronger untouched test requires independently sealed new records with verified linkage and compatible labels/units, not a re-split or removal of just two known exposed IDs.

**Next milestone:** develop noise robustness on train/validation data only, using reproducible synthetic perturbations with fixed seeds, physical units, noise types and severity levels. Assess a separate signal-quality task against explicit quality labels and failure definitions; the existing ~ rhythm class is not a signal-quality reference standard. Choose gates and thresholds on development data, freeze their case accounting and AF deferral criteria, then obtain separate independently sealed evaluation evidence for the quality-aware system. Do not tune these changes against the internal holdout outcomes.

Actual evaluation, training, dependency installation, commit and push remain unauthorized in the present task. Full outputs remain local and ignored; durable external archival remains pending, so no complete external-archival claim is supported.
