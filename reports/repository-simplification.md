# Repository simplification

Reviewed input: the uploaded snapshot at
`05d8cebf4af570aaf2ab40d80c7c715b573be534`.
Changes were prepared in an isolated copy. The user's WSL checkout, active
experiments, branch history, environments, and output directories were not modified.

## Changes

- Keep YAML and `scripts/manage.py` as the public workflow; retain Makefile aliases.
- Add `setup-evaluation` to prepare the evaluation profile independently, using
  the existing exact locked setup commands. Existing setup behavior is unchanged.
- Make `verify-data` a repeatable train/validation raw-checksum and frozen-metadata
  check. It does not decode ECG, run models, query CUDA, fit preprocessing, create
  output, or require empty run directories. Doctor/train preflight remains unchanged.
- Point missing evaluation environments to the correct setup command.
- Document automatic training-only preprocessing as the ordinary workflow;
  separate fitting/reuse is optional.
- Shorten the current workflow, index configurations and results, separate
  numerical methods, and distinguish compatibility/history from current instructions.
- Retain all legacy paths and supported loaders rather than breaking provenance
  to reduce the file count.

Changed existing files: README.md, CONTRIBUTING.md, Makefile,
docs/configuration.md, docs/reproduction.md, environments/cuda128/README.md,
scripts/manage.py, scripts/doctor_environment.py.
Added: docs/README.md, docs/methods.md, docs/compatibility.md, reports/README.md,
tests/test_workflow.py, and this report.

## Preservation evidence

**57 protected files are byte-for-byte unchanged**: every numerical source file,
all existing configurations, interpreter pin, both project descriptors and locks,
frozen manifests/splits/contracts, historical documents, and existing result reports.
All 17 frozen/historical snapshot hashes pass, with unchanged
5,969 / 1,280 / 1,279 assignments. No raw holdout signals were available or accessed.

All eight existing YAML documents, resolved configurations, and output paths
match the pre-change baseline. Original setup/train/smoke/evaluate/doctor/test/fit
command vectors match exactly in the same checkout. The intentional routing
change is verify-data; setup-evaluation is a new action.

Both model versions' pre-change synthetic checkpoints load under the new workflow.
On fixed generated physical inputs with the same CPU environment, logits,
predictions, and preprocessing match exactly: maximum absolute difference 0,
using zero tolerance. These are engineering checks with synthetic weights,
not new research results.

The provided seed42 reports remain unchanged:

| Validation metric | Reference | Temporal |
| --- | --- | --- |
| Four-class macro F1 | 0.5660961198335968 | 0.8142108625947617 |
| Accuracy | 0.6734375 | 0.859375 |
| AF correct / all AF | 37 / 114 | 88 / 114 |

Actual trained weights, new seed43/44 results, and raw ECG were not included in
the uploaded source archive. Real-checkpoint and full-dataset verification were
therefore not performed here. The update does not contain or modify those artifacts.

## Verification

Executed with Python 3.11.16, uv 0.12.23, Torch 2.8.0+cpu, NumPy 2.2.6,
SciPy 1.15.3, WFDB 4.3.1, and the unchanged exact CPU lock.
The documented project uv pin remains 0.12.5; it was not upgraded.

- Exact CPU installation from the existing local cache: 58 compatible distributions.
- **177 synthetic tests passed** (162 existing and 15 new).
- Ruff lint and formatting passed.
- Both locks passed offline consistency checks.
- Ignore/source/credential contracts and frozen snapshot verification passed.
- Configuration/command parity and synthetic checkpoint checks passed.
- Current documentation file links passed; whitespace checks passed.
- Installer checks passed: compatible application and backup, additional reports
  preserved, committed idempotent recheck, dirty/conflicting checkout rejection,
  lifecycle/process rejection, corrupted payload rejection, and write-failure rollback.
  Process-table cases used synthetic records because live /proc is unavailable in
  this verification sandbox; the shipped installer uses Linux/WSL /proc and fails
  closed when it cannot inspect that interface.
- Built sdist and wheel; clean lock-populated wheel environment passed dependency,
  CLI help/version, import, and packaged decoder-contract checks.

Representative commands, from the repository root:

```sh
uv sync --locked --all-extras --python 3.11.16 --default-index https://pypi.org/simple
.venv/bin/ruff check src tests scripts environments/cuda128/verify_gpu.py
.venv/bin/ruff format --check src tests scripts environments/cuda128/verify_gpu.py
.venv/bin/python -m pytest -q tests --tb=short
uv pip check --python .venv/bin/python
uv lock --check --offline --python .venv/bin/python --default-index https://pypi.org/simple
uv lock --check --offline --project environments/cuda128 --python .venv/bin/python
.venv/bin/python scripts/check_repository_contracts.py
.venv/bin/python scripts/verify_frozen.py
.venv/bin/python -m build --no-isolation
git diff --check
```

Installation used offline mode and the task-local cache; a clean temporary
environment installed the built wheel without an editable source package.
Verification used a task-local managed interpreter, `UV_PYTHON`/
`UV_PYTHON_INSTALL_DIR` for offline discovery, and a writable `TMPDIR`.
These locations are verification-host details, not reproduction requirements.

Initial baseline failures were environmental: two tests could not discover Python
3.11 in offline uv subprocesses. Pointing discovery at the already installed
interpreter fixed both; no numerical code changed. A first package build included
sandbox-generated pytest/PyTorch temporary directories because the default system
temporary directory was unavailable. Removing those isolated test artifacts and
using a writable temporary directory outside the source checkout fixed the build.
The container also printed CPU-info/NNPACK initialization warnings; numerical
checks and tests passed. No GPU execution or full research training was performed.

## Application and historical replay

Apply the guarded update only after the experiment manager and training jobs finish.
The installer verifies reviewed file hashes, requires a clean checkout, refuses
active processes/running lifecycle records, backs up changed files, and rejects
conflicting edits. Additional completed experiment reports are preserved.

Launcher changes alter source-provenance hashes. Historical reproduction/resume
uses the original recorded Git revision and environment. Resume checks are not
weakened. Preserve existing outputs; do not rerun completed experiments in their
original output directories. No claim is made of hardware-independent or
bit-identical future retraining.

Model/training refactors, dependency changes, configuration inheritance, removal
of historical compatibility files, performance tuning, and new research stages
are outside this cleanup.
