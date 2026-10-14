# Configuration and commands

Use strict schema-2 YAML for new experiments. Duplicate, unknown, missing, or
invalid settings fail before execution. Paths are repository-relative; keep
secrets out of YAML. Original YAML and resolved JSON are saved with each run.

## Available configurations

| Configuration | Model | Training environment | Seed |
| --- | --- | --- | --- |
| [reference.yaml](../configs/reference.yaml) | Reference | CPU | 42 |
| [config.yaml](../config.yaml) | Reference | CUDA 12.8 | 42 |
| [temporal.yaml](../configs/temporal.yaml) | Temporal | CPU | 42 |
| [temporal-cuda128.yaml](../configs/temporal-cuda128.yaml) | Temporal | CUDA 12.8 | 42 |
| [reference-cuda128-seed43.yaml](../configs/reference-cuda128-seed43.yaml) | Reference | CUDA 12.8 | 43 |
| [reference-cuda128-seed44.yaml](../configs/reference-cuda128-seed44.yaml) | Reference | CUDA 12.8 | 44 |
| [temporal-cuda128-seed43.yaml](../configs/temporal-cuda128-seed43.yaml) | Temporal | CUDA 12.8 | 43 |
| [temporal-cuda128-seed44.yaml](../configs/temporal-cuda128-seed44.yaml) | Temporal | CUDA 12.8 | 44 |

Seed43/44 files belong to the [fixed follow-up](experiments/rhythm-multiseed-v1.md).
Do not edit inputs used by running or completed experiments.

For a new experiment, copy a YAML to a new file, change the name/seed to obtain
a distinct output path, inspect it, and commit it before training. The example
below refers to a file you create:

```sh
python3 scripts/manage.py show --config configs/my-experiment.yaml
python3 scripts/manage.py train --config configs/my-experiment.yaml --dry-run
```

Changing the training seed does not change the frozen split directory or assignments.

## Settings

| Section | Purpose |
| --- | --- |
| `experiment` | Distinct name and 32-bit training seed |
| `environment.profile` | `cpu` or `cuda128`; selects the locked training environment |
| `runtime` | `device` (cpu/cuda/auto), visible `gpu_index`, CPU threads |
| `data` | Existing frozen split directory |
| `preprocessing` | Fixed method; `reuse: null` fits automatically on training |
| `model` | Supported version, widths, kernels, training dropout |
| `optimizer`, `training` | AdamW settings, batch size, epochs, patience, caching, validation macro F1 |
| `smoke` | Training-only subset, diagnostic dropout, bounded steps/time and target |
| `paths.output_root` | Ignored output location |
| `execution` | Explicit resume and noncanonical-development flags |
| `evaluation` | Independent profile, device, GPU index, CPU threads |

CUDA 12.8 is supported on Linux x86_64, including WSL. Choose `cuda128` and
`device: cuda`, then select `runtime.gpu_index` in a new committed configuration.
The index respects caller-provided CUDA visibility restrictions.
Candidate dilations are fixed by the model version.

## Actions

| Action | Behavior |
| --- | --- |
| `show` | Resolve settings, paths, and setup commands |
| `setup` | Install the selected locked training environment |
| `setup-evaluation` | Install the independently selected locked evaluator |
| `verify-data` | Repeatable metadata and train/validation raw checksums; no decoding, GPU execution, or run output |
| `doctor` | Check readiness for a new run, including environment/device execution and output availability |
| `test` | Run synthetic tests |
| `smoke` | Bounded training-only wiring check |
| `train` | Full development training |
| `evaluate` | Evaluate the saved best checkpoint on validation |
| `fit-preprocessing` | Optional separate training-only fit for explicit reuse |

CUDA training and CPU evaluation need both environments:
`setup` prepares training; `setup-evaluation` prepares the evaluator.
Evaluation keeps checkpoint training provenance intact, records its actual runtime,
and derives the checkpoint/output path from the training profile. It is validation-only
and rejects smoke checkpoints.

`verify-data` works after completed runs. `doctor` intentionally checks readiness
for a new train output; a completed output is unavailable for a new run.
`--dry-run` prints commands without executing them.
`doctor --smoke` permits development source and checks training inputs.
Canonical full training requires clean committed source; the explicit development
override is recorded and does not establish canonical results.

## Calibration analysis actions

The [analysis protocol](experiments/temporal-calibration-deferral-v1.md) documents
the separate analysis-only YAML schema and commands. Training schema-2 behavior
is unchanged. `analysis-preflight` reads saved CPU predictions and provenance
without fitting or writing outputs, even when `execution_authorized: false`.
`analysis-fit` and `analysis-assess` require literal YAML `execution_authorized: true`
plus `--authorize-analysis`, reviewed committed inputs and a clean source tree.
The CLI cannot override false, missing or invalid YAML authorization values.
The completed exploratory run was explicitly authorized at
`47d198e15f0fdcfbd2210ea9cf6700af7a1b3889`; its configuration remains true as
execution provenance. See the [completed report](../reports/temporal-calibration-deferral-v1.md).
Existing outputs must not be overwritten or silently retried. Authorization does
not permit policy changes or holdout access through calibration commands.

## Completed internal rhythm-only evaluation

The separate `final-preflight` and `final-evaluate` actions use the
[frozen internal protocol](experiments/temporal-final-evaluation-v1.md), not the
training or calibration schemas. `final-evaluate` requires committed literal
YAML true, explicit runtime authorization and a clean tree. All three checkpoints
and the one-time output path are fixed; calibration, deferral and quality gating
are disabled. Execution at `bbdf121a81557bb4e614a452a1ffb42ba6a9b27d` is
[complete](../reports/temporal-internal-rhythm-evaluation-v1.md). The YAML remains
true as provenance; the occupied output root rejects further evaluation. Do not
rerun, rename outputs to bypass the collision, or treat this authorization as
permission for a new experiment. The holdout is consumed for future development;
quality-aware triage needs separate evaluation evidence.

## Declarative noise/dataset proposals

`configs/noise-quality-datasets-v1.yaml` and `configs/noise-robustness-v1.yaml`
are **unsupported declarative drafts**, not training schema-2 or implemented
analysis/final-evaluation schemas. Both have `execution_authorized: false`.
Do not pass them to `scripts/manage.py`; there is no runnable acquisition,
augmentation or stress-test command for them. See the
[readiness report](../reports/dataset-readiness-noise-quality-v1.md) and
[frozen proposal](experiments/noise-robustness-v1.md). Dataset/input/case-plan
hashes are pending until separately authorized acquisition/audit/implementation;
no new numerical policy is enabled by documenting these drafts.

## Resume

Keep the original configuration and request continuation with an action flag:

```sh
python3 scripts/manage.py train --config configs/my-experiment.yaml --resume
```

A usable `last.pt` restores model, optimizer, selection/history, and applicable RNG
states at the completed epoch boundary. A partial epoch is repeated. Completed runs
cannot resume or be overwritten. There is no exact mid-epoch continuation claim.

Settings, data, preprocessing, model, source bundle, packages, locks, and runtime
must agree. The YAML resume flag is retained, but `--resume` avoids editing an
experiment merely to continue it. For historical resume after source changes,
use the [recorded revision](reproduction.md#historical-reproduction).
