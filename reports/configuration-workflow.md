# Configuration workflow verification

Date: 2026-10-12. This report separates local implementation checks from the
user-supplied WSL training diagnostics. Full real-data baseline training and
generalization results remain pending.

## Implemented

- Root `config.yaml` schema 2 controls the environment profile, execution device,
  visible GPU index, seed, model dimensions/dropout, AdamW settings, training
  limits, smoke settings, preprocessing reuse, and ignored output root.
- `Makefile` and `scripts/manage.py` provide show, setup, doctor, test,
  fit-preprocessing, smoke, train and development-validation evaluation commands.
  Setup selects the existing CPU or CUDA lock and its interpreter, with no
  activation step. CUDA setup uses the locked editable build dependency added
  after the earlier missing-editables failure.
- Configuration validation rejects unknown/missing/duplicate keys, wrong types,
  unsupported algorithm choices, invalid GPU indices, unsafe paths, unsupported
  smoke budgets and explicit CUDA execution with a CPU profile.
- Reports/checkpoints contain the original YAML snapshot/hash, resolved values,
  active environment lock hashes and exact installed versions. GPU index is
  honored, and a mismatched Torch/Python profile cannot be reported as the selected
  environment. Valid legacy JSON configurations remain supported.
- Frozen splits, decoder checks, training-only preprocessing, holdout restrictions,
  checkpoint reload checks, AF denominators and output overwrite protection remain
  enforced. Quality gating, calibration and exact interrupted-run resume remain
  unimplemented.

## Local synthetic and installation verification

Local runtime: Linux x86_64, Python 3.11.16, uv 0.12.23, Torch 2.8.0+cpu,
NumPy 2.2.6, SciPy 1.15.3, WFDB 4.3.1, PyYAML 6.0.3, Ruff 0.11.13,
pytest 8.4.2. No GPU is available in this verification workspace.

Checks passed:

- **121 tests**: original 91 plus 30 configuration/routing/provenance cases.
  Final run: 12.33 seconds. All ECG-like signals in tests are generated inside
  temporary directories; none are research patient recordings.
- Ruff check and format validation for package source, launch scripts and tests.
- Actual CPU setup from an alternate temporary YAML: 57 installed distributions;
  dependency consistency passed. This used the same root lock without modification.
- CPU doctor ran the configured default CNN forward/backward on synthetic masked
  arrays with finite loss/gradients. No ECG records were read. Torch emitted a
  cpuinfo processor-information advisory in this container; the check exited zero.
- Isolated Python environment had no PyYAML. The launcher successfully bootstrapped
  pinned PyYAML 6.0.3, validated a CPU configuration, and showed a dry-run training
  command without installing the research environment or training.
- CPU/CUDA setup and training command routing, actual GPU index API selection
  using stubs, and no-execution dry runs were tested. Stubbed CUDA tests do not
  establish GPU execution.
- Synthetic YAML-driven CLI smoke run verified checkpoint reload, unchanged
  holdout manifest bytes and no validation/holdout signal access. A separate
  synthetic library run reused saved full-training preprocessing while a guard
  made any new scale fitting fail; reuse succeeded.
- Both root and CUDA locks passed offline stale-lock checks. Root CPU lock SHA256:
  `4ff3ef044d9e70d26b8a5c0494478e30cbc3f73ac3ecd94c109e1dd1adf44c8d`.
  Neither dependency recipe/lock is modified by this source update. The CUDA
  environment already includes the user's `editables==0.5` repair.
- Nineteen original data/config/decoder/policy/dependency files were checked byte
  unchanged against the uploaded snapshot. Frozen training metadata still validates
  all 5,969 records. This is metadata integrity checking, not new rhythm evaluation.
- Built the installable distributions, confirmed the new configuration module is
  in the wheel, and verified wheel CLI help/module import in a separate environment
  with no optional dependencies.

Representative commands, run from the repository root:

```sh
python3 scripts/manage.py setup --config outputs/configuration-verification/cpu.yaml
make doctor CONFIG=outputs/configuration-verification/cpu.yaml
python3 scripts/manage.py train --config outputs/configuration-verification/cpu.yaml --dry-run
.venv/bin/pytest -q tests
.venv/bin/ruff check src scripts tests
.venv/bin/ruff format --check src scripts tests
uv lock --check --offline --default-index https://pypi.org/simple
uv lock --project environments/cuda128 --check --offline --default-index https://pypi.org/simple
.venv/bin/python -m build --no-isolation
```

Writable runtime cache/interpreter/temp locations were used in this workspace.
They are not requirements for another checkout. The temporary CPU verification
YAML and generated artifacts are ignored and not included in the update.

## Previously supplied WSL evidence

The user supplied successful CUDA verification on an NVIDIA GeForce RTX 5050
Laptop GPU, compute capability 12.0, Python 3.11.16 and Torch 2.8.0+cu128 with
CUDA runtime 12.8. Real model forward/backward was verified there.

The four-record training-only overfit diagnostic, one record from each class,
was reported as follows:

| Execution | Steps | Training accuracy | Cross-entropy | Checkpoint reload |
| --- | ---: | ---: | ---: | --- |
| CPU | 70 | 1.0 | 0.093455 | Passed |
| CUDA | 66 | 1.0 | 0.099305 | Passed |

Both reported training-only diagnostics accessed no validation or holdout signals.
The earlier 16-record CPU diagnostic stopped at 200 steps with accuracy 0.625 and
loss 1.004647; it did not meet its overfit target. These are memorization checks,
not generalization, calibrated AF detection, clinical validation or speed guarantees.
These WSL runs preceded this YAML workflow update. The new GPU launcher/runtime
changes should be checked using `make doctor` and `make test` on the user's machine
before launching full training.

## Delivery boundaries

The guarded installer verifies existing source/policy hashes, refuses conflicting
local files, preserves data/dependency files, and creates rollback backups under
ignored outputs. It installs source/documentation only. No real ECG download,
full real-data training, Git commit or push is performed by this update.
