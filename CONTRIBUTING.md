# Contributing

Read AGENTS.md and preserve frozen manifests, assignments and historical provenance.
Use YAML plus the launcher; never regenerate frozen splits for a fresh clone.
Keep ECG, environments, checkpoints and outputs out of Git. Synthetic tests generate
signals in temporary directories; no raw dataset or GPU is required.

```sh
python3 scripts/manage.py setup --config configs/reference.yaml
.venv/bin/ruff check src tests scripts environments/cuda128/verify_gpu.py
.venv/bin/ruff format --check src tests scripts environments/cuda128/verify_gpu.py
.venv/bin/python -m pytest -q tests
.venv/bin/python -m build --no-isolation
uv pip check --python .venv/bin/python
uv lock --check --offline
uv lock --check --offline --project environments/cuda128
python3 scripts/manage.py show --config configs/reference.yaml
git diff --check
```

Changes to decoder semantics require a reviewed versioned compatibility contract;
changes to models require versioned artifacts. Do not tune against holdout. Use
`doctor --smoke` during development; canonical full training requires clean committed
source. Report actual checks and failed checks honestly. Do not claim clinical results
from synthetic tests. CI performs CPU-only synthetic and packaging checks. License,
CITATION.cff authorship, DOI and publication decisions belong to the project owner.
