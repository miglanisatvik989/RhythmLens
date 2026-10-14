# Results and verification evidence

## Recorded seed42 development runs

The linked reports are authoritative observed results for the first comparison,
using the same frozen split and training preprocessing.

| Metric | Reference CNN | Temporal candidate |
| --- | --- | --- |
| Validation records | 1,280 | 1,280 |
| Accuracy | 0.6734375 | 0.859375 |
| Four-class macro F1 | 0.5660961 | 0.8142109 |
| AF correct / all AF | 37 / 114 | 88 / 114 |
| AF recall | 0.3245614 | 0.7719298 |
| Best epoch | 23 | 31 |

- [Reference report](baselines/reference-seed42.json)
- [Temporal report](baselines/temporal-seed42.json)

These are selected-checkpoint validation results from one seed, not independent
holdout evidence or clinical validation. Architectures differ in multiple
properties. Holdout signals were not accessed. Quality gating, calibration,
and AF deferral remain pending.

The [completed multiseed report](rhythm-multiseed-v1.md) and
[full-precision summary](rhythm-multiseed-v1.json) record all six reference/temporal
seed42/43/44 runs under the [fixed protocol](../docs/experiments/rhythm-multiseed-v1.md).
The table above remains the original seed42 evidence. All reported performance
is checkpoint-selected validation evidence; the holdout has no recorded model-performance evaluation; see the exposure audit below.

## Calibration and confidence deferral

The [completed reviewed-v1 report](temporal-calibration-deferral-v1.md) records all
three seeds and arms, role counts, frozen policies, paired comparisons, artifact
identities and preservation status. The [frozen development protocol](../docs/experiments/temporal-calibration-deferral-v1.md)
and [historical implementation verification](temporal-calibration-deferral-v1-verification.md)
describe the logits-only workflow and its synthetic checks. Execution was explicitly
authorized at revision `47d198e15f0fdcfbd2210ea9cf6700af7a1b3889`.

Calibration improvements were mixed. Deferral increased accepted-case F1 on a
different subset but reduced automatic AF sensitivity; this does not establish
overall improvement. Results are exploratory because validation already informed
checkpoint and architecture selection. No policy retuning or deployment
recommendation follows from these outcomes. Signal-quality gating remains
unavailable and holdout evaluation remains excluded.

Role metadata are trackable at the protocol-specified paths. Complete outputs
remain ignored and local: no repository archival mechanism was found, and a Git
commit of the report/role metadata would not preserve those output bytes.

## Final-evaluation preparation

The [integrity audit](final-evaluation-integrity-audit-v1.md) verifies recording
and exact-duplicate separation but documents pre-freeze access to eventual
holdout signals, including two previewed holdout IDs. No holdout model-performance
evaluation is recorded; strict never-seen status and patient/source/time
independence cannot be established. The [internal rhythm-only protocol](../docs/experiments/temporal-final-evaluation-v1.md)
uses all three raw temporal checkpoints equally, without calibration or deferral,
and documents the implemented dedicated evaluator and synthetic verification.
Real execution remains unauthorized. This is a qualified internal evaluation of
the frozen rhythm-only baseline, not final validation of the eventual quality-aware
system. Inspecting its outcomes consumes this holdout for future development;
subsequent quality-gating changes need separate evaluation evidence. Existing
frozen artifacts and all exposure/independence limitations are preserved.

## Dated verification milestones

| Report | Evidence |
| --- | --- |
| [Internal evaluator verification](temporal-final-evaluation-v1-verification.md) | Dedicated executor; synthetic checks; real evaluation pending |
| [Environment setup](environment-setup.md) | Initial installation and dependencies |
| [Data audit](challenge2017-audit.md) | Integrity, labels, duplicates, limitations |
| [Audit verification](challenge2017-verification.md) | Acquisition and frozen splits |
| [Baseline implementation](rhythm-baseline-implementation.md) | Initial implementation checks |
| [Configuration workflow](configuration-workflow.md) | Earlier launcher/configuration delivery |
| [Pretraining readiness](pretraining-readiness.md) | Hardening, packaging, synthetic CPU/CUDA checks |
| [Repository simplification](repository-simplification.md) | Workflow fixes and behavior-preservation evidence |

Dated reports describe their milestone, not the current quickstart. Their contents
and paths are preserved. Follow [current documentation](../docs/README.md).
