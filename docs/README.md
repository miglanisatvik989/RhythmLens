# Documentation

Start with the [quickstart](../README.md). New experiments use YAML and
`scripts/manage.py`; Makefile commands delegate to that launcher.

| Document | Purpose |
| --- | --- |
| [Reproduction](reproduction.md) | Locked setup, data restoration, artifacts, historical replay |
| [Configuration](configuration.md) | Existing configurations, settings, GPU selection, resume |
| [Methods](methods.md) | Preprocessing, model versions, evaluation definitions, limitations |
| [Multi-seed protocol](experiments/rhythm-multiseed-v1.md) | Fixed seed42/43/44 comparison |
| [Calibration and deferral](experiments/temporal-calibration-deferral-v1.md) | Logits-only analysis commands, synthetic verification; real analysis pending |
| [Results and evidence](../reports/README.md) | Observed runs and dated verification milestones |
| [Contributing](../CONTRIBUTING.md) | CPU checks, packaging, and change requirements |
| [Compatibility](compatibility.md) | Retained legacy entry points |

`docs/archive/` contains earlier instructions, not the current workflow.
Dated reports describe checks performed at that milestone. Their original paths
and contents are retained for historical references and provenance.
