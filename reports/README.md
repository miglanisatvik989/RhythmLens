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
is checkpoint-selected validation evidence; the holdout remains untouched.

## Calibration and confidence deferral

The [development protocol](../docs/experiments/temporal-calibration-deferral-v1.md)
and [verification evidence](temporal-calibration-deferral-v1-verification.md)
describe the implemented logits-only workflow and synthetic checks. Real role
assignment, calibration, threshold selection and assessment remain pending.
The real YAML has `execution_authorized: false`; fitting and assessment require
literal YAML true plus explicit CLI authorization, reviewed committed inputs
and a clean source tree. CLI flags cannot override YAML false. Signal-quality
gating remains unavailable. Further results will be exploratory because validation
already informed checkpoint and architecture selection.

## Dated verification milestones

| Report | Evidence |
| --- | --- |
| [Environment setup](environment-setup.md) | Initial installation and dependencies |
| [Data audit](challenge2017-audit.md) | Integrity, labels, duplicates, limitations |
| [Audit verification](challenge2017-verification.md) | Acquisition and frozen splits |
| [Baseline implementation](rhythm-baseline-implementation.md) | Initial implementation checks |
| [Configuration workflow](configuration-workflow.md) | Earlier launcher/configuration delivery |
| [Pretraining readiness](pretraining-readiness.md) | Hardening, packaging, synthetic CPU/CUDA checks |
| [Repository simplification](repository-simplification.md) | Workflow fixes and behavior-preservation evidence |

Dated reports describe their milestone, not the current quickstart. Their contents
and paths are preserved. Follow [current documentation](../docs/README.md).
