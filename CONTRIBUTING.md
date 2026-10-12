# Contributing

Read [AGENTS.md](AGENTS.md). Preserve frozen manifests, compatibility, experiment
inputs, and historical results. Use YAML and the [launcher](docs/configuration.md);
do not regenerate frozen splits for a fresh clone.

## CPU checks

Tests generate signals in temporary directories; no dataset or GPU is required.

```sh
python3 scripts/manage.py setup --config configs/reference.yaml
.venv/bin/ruff check src tests scripts environments/cuda128/verify_gpu.py
.venv/bin/ruff format --check src tests scripts environments/cuda128/verify_gpu.py
.venv/bin/python -m pytest -q tests
uv pip check --python .venv/bin/python
uv lock --check --offline
uv lock --check --offline --project environments/cuda128
python3 scripts/check_repository_contracts.py
python3 scripts/verify_frozen.py
.venv/bin/python -m build --no-isolation
git diff --check
```

CI also installs the wheel in a clean locked environment and checks CLI/imports.
Use the project interpreter for additional scripts instead of assuming a system
Python version.

Decoder changes require a reviewed compatibility contract; model changes require
versioned artifacts and regression evidence. Never tune against holdout, fabricate
results, or claim clinical performance from synthetic checks.

Keep ECG, environments, checkpoints, predictions, and local outputs ignored.
Track compact observed reports, configurations, and protocols.
Canonical training needs clean committed source; `doctor --smoke` permits development
checks. Do not change source/configuration during an active experiment.
Preserve original revisions for historical replay and resume.

Verify Git author/committer identity before committing. License, authorship,
citation, DOI, and publication decisions belong to the project owner.
