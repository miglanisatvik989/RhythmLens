> Historical workflow snapshot. Use ../reproduction.md for current instructions.

# Configuration and reproduction

Edit the root `config.yaml`. Its schema is strict: every documented setting must
be present, and unknown or duplicate keys are rejected before setup or execution.
The existing JSON configuration files and low-level CLI commands remain supported.

## Select the environment

For the verified RTX 5050 / WSL route:

```yaml
environment:
  profile: cuda128
runtime:
  device: cuda
  gpu_index: 0
  cpu_threads: 2
```

For CPU execution, change `environment.profile` to `cpu` and `runtime.device` to
`cpu`. Run `make setup` after changing the environment profile. The launcher picks
the interpreter itself, so environment activation is unnecessary. With
`device: auto`, GPU fallback to CPU is explicitly requested; with `device: cuda`,
missing CUDA or an unavailable GPU index causes an error.

| Profile | Exact dependency lock | Environment directory | Torch build |
| --- | --- | --- | --- |
| `cpu` | root `uv.lock` | `.venv/` | 2.8.0+cpu on Linux/Windows; native 2.8.0 on macOS |
| `cuda128` | `environments/cuda128/uv.lock` | `environments/cuda128/.venv/` | 2.8.0+cu128 |

Python is pinned to 3.11.16. CUDA 12.8 is currently locked for Linux x86_64,
including WSL. Changing a GPU index does not require another environment setup.
GPU indices refer to the devices visible to PyTorch. Changing CUDA or Torch
versions requires a matching environment recipe and lock;
these software versions are not arbitrary runtime YAML overrides. Current profiles
are CPU and CUDA 12.8. No alternate accelerator profile is implemented.

The launcher requires `uv` for setup. When the Python used to launch it does not
have PyYAML 6.0.3, it bootstraps that pinned version using `uv run --no-project`. That first
invocation may need network access and a Python download. The selected research
environment is always installed from its existing lock with `--locked`; bootstrap
does not change the root lock. CUDA setup also installs the editable source package
with dependency resolution and build isolation disabled, using its locked
Hatchling and `editables==0.5` build dependencies.

## Commands

Run from the repository root:

```sh
make show
make setup
make doctor
make test
make smoke
```

`show` prints the resolved configuration, environment, and output paths. `doctor`
checks the installed Python/Torch build and runs the configured CNN forward and
backward on synthetic masked arrays; it reads no ECG records. `test` runs the
synthetic suite using the selected environment. `smoke` is a bounded training-only
memorization check, using `smoke.dropout` instead of `model.dropout`.

Once checks pass, launch full training:

```sh
make train
```

`train` starts fresh weights, uses `model.dropout`, and selects the checkpoint by
four-class macro F1 on development validation. There is no holdout evaluation
command. `make evaluate` repeats development validation of this experiment's
best training checkpoint. It cannot choose a holdout partition.

If Make is unavailable, use the equivalent Python commands:

```sh
python3 scripts/manage.py setup
python3 scripts/manage.py doctor
python3 scripts/manage.py train
```

Inspect a command without installing or executing it:

```sh
python3 scripts/manage.py train --dry-run
```

On a first invocation, even a dry run may need the small pinned YAML bootstrap;
it does not install the research environment or load any ECG records.

Use another tracked configuration with `make train CONFIG=configs/my-run.yaml`,
or `python3 scripts/manage.py train --config configs/my-run.yaml`. Configurations
and their paths must stay inside the repository. Missing environments, invalid
choices, unsupported algorithms, and unsafe paths produce errors.

## Choose a run and its settings

| Section | Choices |
| --- | --- |
| `experiment` | Name and seed; a new name creates a separate output tree |
| `environment` | Locked CPU or CUDA 12.8 profile |
| `runtime` | Execution device, GPU index, CPU threads |
| `data` | Path to an already frozen split; no re-splitting during training |
| `preprocessing` | Existing training-scale method; optional saved full-training state |
| `model` | Stage widths, odd kernel sizes, and full-training dropout |
| `optimizer` | AdamW learning rate and weight decay |
| `training` | Batch size, maximum epochs, validation patience, signal cache |
| `smoke` | Training subset size, batch size, bounded budget, target, diagnostic dropout |
| `paths` | An ignored output root |

Only `rhythm-cnn-v1`, AdamW, the fixed training-derived amplitude scale, and
validation four-class macro F1 selection are implemented. Validation and holdout
policy are enforced by the pipeline, rather than switchable convenience options.
No quality model, probability calibration, AF deferral policy, clinical claims,
exact interrupted-run resume, or additional algorithm is implemented by this update.

Run outputs follow:

```text
outputs/rhythm/<experiment-name>/<profile>/seed<seed>/<action>/
```

For the supplied YAML, training is under
`outputs/rhythm/baseline-v1/cuda128/seed42/train/`. Existing action directories are
never overwritten. To repeat an action, copy the config and use a new experiment
name. Ctrl+C stops execution; preserved checkpoints do not support exact resume.

`preprocessing.reuse: null` fits the scale on all frozen training records. To reuse
your already verified scale, set it to:

```yaml
preprocessing:
  method: median-fixed-training-rms-v1
  reuse: outputs/rhythm/smoke-seed42-v1/preprocessing.json
```

Saved preprocessing must match the full frozen training manifest and identifiers.
Setting another path does not bypass this verification. `make fit-preprocessing`
fits a new training state explicitly. Raw data paths still come from the audited
manifest; acquisition and audit use the existing data CLI and tracked audit config.

## What each run records

Training/smoke runs retain `experiment.yaml`, the resolved `config.json`, fitted
`preprocessing.json`, history, checkpoints, and the final report in ignored output
directories. Reports and checkpoints record the original YAML hash and document,
the frozen CPU lock and selected environment lock hashes, exact installed package
versions, seed, model configuration, preprocessing identity, data hashes, source
hashes, Git revision/dirty state, and actual device. This fixes the earlier GPU
report's lack of an explicit active CUDA lock hash. Hashes describe observed inputs;
synthetic checks and tiny training accuracy are not generalization results.

Fit and repeated-validation reports also record reproduction metadata. Legacy
JSON runs retain their existing behavior and have no original YAML snapshot.
Checkpoints remain loadable through the existing safe tensor/primitive route.
CPU and GPU runs can differ numerically even with deterministic runtime settings.

Commit source, documentation, YAML, environment recipes/locks, de-identified
manifests and frozen split files when you decide to publish. Keep local environments,
ECG samples, checkpoints and run outputs ignored. This update performs no data
download, full real-data training, Git commit or push.
