# RhythmLens

Reproducible ECG rhythm research on the frozen PhysioNet Challenge 2017 v3
recordings. Reference and temporal-candidate implementations are available;
research results, quality gating, calibration and AF deferral remain pending.
This is not a clinically validated system.

Use Python pinned in `.python-version` and uv 0.12.5. From a fresh clone:

```sh
python3 scripts/manage.py setup --config configs/reference.yaml
# Restore the recorded raw files; see docs/reproduction.md. Do not regenerate splits.
python3 scripts/manage.py verify-data --config configs/reference.yaml
python3 scripts/manage.py doctor --config configs/reference.yaml
python3 scripts/manage.py test --config configs/reference.yaml
python3 scripts/manage.py fit-preprocessing --config configs/reference.yaml
# Set preprocessing.reuse to the fit-preprocessing/preprocessing.json just produced.
# Review and commit that YAML change before canonical training.
python3 scripts/manage.py smoke --config configs/reference.yaml
python3 scripts/manage.py train --config configs/reference.yaml
```

Training is an explicit user action; preflight requires clean committed source.
The existing 5,969 / 1,280 / 1,279 assignments are reused throughout. Patient
identifiers are unavailable: these are recording splits with verified duplicates
kept together, not evidence of patient independence. Holdout evaluation is locked.

[Reproduction and data restoration](docs/reproduction.md) ·
[YAML settings and resume](docs/configuration.md) ·
[Contributor checks](CONTRIBUTING.md) ·
[Readiness report](reports/pretraining-readiness.md)

The separate CUDA 12.8 environment uses `config.yaml` (CUDA by default). Change
`runtime.gpu_index` and `evaluation` settings in YAML, then run the same launcher.
Use `configs/temporal.yaml` to opt into the longer-context candidate. Read the
reproduction guide before running either experiment.
