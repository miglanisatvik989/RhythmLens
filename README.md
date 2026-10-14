# RhythmLens

Quality-aware ECG research with reproducible rhythm-classification experiments.
Reference and temporal models are implemented, with [recorded development results](reports/README.md).
The [completed exploratory calibration/deferral report](reports/temporal-calibration-deferral-v1.md)
records all three temporal seeds using the [frozen protocol](docs/experiments/temporal-calibration-deferral-v1.md).
Calibration effects were mixed; confidence deferral increased accepted-case F1
but reduced automatic AF sensitivity. The [completed internal rhythm-only evaluation](reports/temporal-internal-rhythm-evaluation-v1.md)
reports mean automatic AF sensitivity 80.70%, automatic four-class error 14.33%,
and 100% assessed coverage. Signal-quality gating remains pending.

## Run an experiment

Use **Python 3.11.16**, **uv 0.12.5**, and Git. Run from the repository root.
[Choose a configuration](docs/configuration.md#available-configurations):
CPU is the starting point; CUDA 12.8 uses a separate locked environment.

```sh
CONFIG=configs/reference.yaml
python3 scripts/manage.py setup --config "$CONFIG"
python3 scripts/manage.py setup-evaluation --config "$CONFIG"
# Restore the recorded dataset once: see docs/reproduction.md#restore-data.
python3 scripts/manage.py verify-data --config "$CONFIG"
python3 scripts/manage.py doctor --config "$CONFIG"
python3 scripts/manage.py smoke --config "$CONFIG"
# Inspect the smoke report before continuing to full training.
python3 scripts/manage.py train --config "$CONFIG"
python3 scripts/manage.py evaluate --config "$CONFIG"
```

For GPU training choose `config.yaml` (reference) or
`configs/temporal-cuda128.yaml` (temporal). `setup-evaluation` prepares the
configured evaluator, including CPU evaluation after CUDA training; it is
idempotent when both profiles are the same.

Preprocessing fits automatically on the **5,969 training records**. The
**1,280 validation records** select checkpoints. The **1,279 holdout records**
now have completed qualified internal rhythm-only evaluation and are consumed
for future development.
The [integrity audit](reports/final-evaluation-integrity-audit-v1.md) documents
pre-freeze signal decoding and two holdout preview exposures; patient, source and
time independence remain unverified. Canonical training requires clean committed source.
For a new run, choose and commit a new experiment configuration with a distinct
name/seed. To reproduce historical inputs, use a fresh checkout at their recorded revision.

Outputs: `outputs/rhythm/<experiment>/<profile>/seed<seed>/<action>/`.
Training writes `best.pt`, `last.pt`, `history.json`, and `run-report.json`;
evaluation writes `evaluation-report.json` and validation predictions/plots.
Existing outputs cannot be silently overwritten.

`make <action> CONFIG=<file>` is an optional alias to the same launcher.
Use `make help` or `python3 scripts/manage.py --help` to see the actions.

## Read more

[Reproduction](docs/reproduction.md) · [Configuration and resume](docs/configuration.md) ·
[Methods](docs/methods.md) · [Documentation index](docs/README.md) ·
[Results and evidence](reports/README.md) · [Contributing](CONTRIBUTING.md)

Patient identifiers are unavailable; recording splits preserve verified duplicate
groups but do not establish patient independence. This research system has not
been clinically validated.

The [dedicated rhythm-only internal evaluation](docs/experiments/temporal-final-evaluation-v1.md)
keeps all three temporal checkpoints fixed and equally reported, with calibration,
deferral and quality gating disabled. The [completed report](reports/temporal-internal-rhythm-evaluation-v1.md)
records all three seeds and the execution revision. This does not validate
quality-aware triage. The holdout is consumed; subsequent quality-gating changes
need separate evaluation evidence. Ignored outputs still need external archival;
a Git report is not an artifact backup.
