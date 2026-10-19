# Documentation

Start with the [quickstart](../README.md). New experiments use YAML and
`scripts/manage.py`; Makefile commands delegate to that launcher.

| Document | Purpose |
| --- | --- |
| [Reproduction](reproduction.md) | Locked setup, data restoration, artifacts, historical replay |
| [Configuration](configuration.md) | Existing configurations, settings, GPU selection, resume |
| [Methods](methods.md) | Preprocessing, model versions, evaluation definitions, limitations |
| [Multi-seed protocol](experiments/rhythm-multiseed-v1.md) | Fixed seed42/43/44 comparison |
| [Calibration and deferral protocol](experiments/temporal-calibration-deferral-v1.md) | Frozen logits-only analysis methods and execution gates |
| [Completed exploratory calibration/deferral results](../reports/temporal-calibration-deferral-v1.md) | reviewed-v1 three-seed outcomes, provenance and local-preservation limitation |
| [Final-evaluation integrity audit](../reports/final-evaluation-integrity-audit-v1.md) | Metadata-only split/exposure audit and limits of the untouched-holdout claim |
| [Rhythm-only internal evaluation](experiments/temporal-final-evaluation-v1.md) | Frozen three-seed methods and execution history; see completed results below |
| [Completed internal rhythm-only results](../reports/temporal-internal-rhythm-evaluation-v1.md) | All three seeds, accounting, uncertainty, artifact hashes and consumed-holdout limitations |
| [Internal evaluator verification](../reports/temporal-final-evaluation-v1-verification.md) | Synthetic inference, accounting, bootstrap and failure checks |
| [Noise/quality dataset readiness](../reports/dataset-readiness-noise-quality-v1.md) | Official source identities, label separation, pending acquisition/audits and backup gap |
| [NSTDB acquisition/audit and adapter](nstdb-adapter.md) | Explicit noise-only acquisition, audit, synthetic-tested adapter; real data and Phase A/B pending |
| [NSTDB publisher-index repair](../reports/nstdb-publisher-index-repair-v1.md) | Preserved failed acquisition, explicit index grammar, metadata-first checks and synthetic verification; real retry/audit pending |
| [Noise robustness proposal](experiments/noise-robustness-v1.md) | Frozen Phase A/B development design; declarative configs only; execution false |
| [Results and evidence](../reports/README.md) | Observed runs and dated verification milestones |
| [Contributing](../CONTRIBUTING.md) | CPU checks, packaging, and change requirements |
| [Compatibility](compatibility.md) | Retained legacy entry points |

`docs/archive/` contains earlier instructions, not the current workflow.
Dated reports describe checks performed at that milestone. Their original paths
and contents are retained for historical references and provenance.
