# Rhythm baseline implementation verification

This report describes implementation checks, not ECG generalization results.
The uploaded source archive contained metadata and code, but no raw ECG samples.
No actual Challenge 2017 signal was decoded or used for training in this workspace.
The real-data training-only smoke run, baseline training and validation remain pending.

## Implemented

- Training-only median-of-record-RMS amplitude fitting; immutable, versioned state.
- One shared physical-signal transform for dataset and inference.
- Frozen metadata verification, recorded decoder identity checks and per-file
  checksums before materializing ECGs. No final-holdout signal access.
- Whole-record datasets, deterministic length buckets, padded tensors and masks.
- 242,244-parameter residual 1D CNN with channel-only temporal-position LayerNorm,
  masking throughout convolution/residual branches, masked pooling and four logits.
- Bounded training-only smoke command, configuration-driven baseline training,
  validation-selected checkpoints, validation metrics/predictions/confusion plots.
- Tensor/primitive-only checkpoint loading and independent reload verification.
- Validation-only evaluation; smoke checkpoints are refused. No final-holdout option.

The CLI has fit-preprocessing, smoke-rhythm, train-rhythm and evaluate-rhythm
commands. Existing acquire/audit/split behavior is retained. Top-level training
imports remain lazy so package --help does not require training dependencies.

## Verification in this workspace

The original uv.lock was installed with uv 0.12.23 (the repository documents
uv 0.12.5). Locked package versions were not changed. The actual runtime was
CPython 3.11.16, PyTorch 2.8.0+cpu, NumPy 2.2.6, SciPy 1.15.3 and WFDB 4.3.1,
on Linux x86_64. GPU execution was not tested.

Commands and outcomes:

```sh
uv sync --locked --all-extras --default-index https://pypi.org/simple
MPLCONFIGDIR=<workspace-cache> .venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
uv pip check --python .venv/bin/python
.venv/bin/rhythmlens --help
.venv/bin/python -m build --no-isolation
```

- Pytest: **91 passed** in the final run (40 existing audit checks, 51 new
  parameterized baseline checks). Final observed runtime: 5.91 seconds.
- Ruff: all checks passed; all 13 source/test files were formatted.
- Dependency consistency: 57 compatible installed distributions.
- CLI help exposes the new commands without running training.
- The sdist and wheel build passed. Wheel CLI execution is checked separately.
- Training metadata from the uploaded frozen snapshot verifies for all 5,969
  training records, without loading any ECG waveform.
- All 15 checked split/source snapshots, uv.lock, pyproject.toml and original
  audit/data/split modules were preserved byte-for-byte.

Synthetic tests cover training-only fitting and the exact scale statistic,
flat signal retention, original-signal preservation, invalid channels/units/rates,
float32 overflow, bounded fitting, adversarial padding (including NaN padding and
nonzero affine/convolution biases), short/odd valid lengths, downsample masks,
zero padding gradients, finite valid gradients, complete seeded bucket coverage,
frozen-data/raw-file tampering, shared training/inference transform parity,
checkpoint/logit reload equivalence, label/data identity checks, AF denominators,
output overwrite refusal, configuration guards, smoke partition access guards,
and a synthetic two-epoch development train/evaluate round trip.

An additional default-architecture CPU forward/backward check used random
synthetic arrays of 2,714, 9,000, 15,000 and 18,287 samples. Loss and every parameter
gradient were finite. It took approximately 0.667 seconds with two Torch threads
in this workspace. This is not ECG performance, a general throughput benchmark,
or a runtime estimate for the user's WSL machine. No synthetic weights are shipped.

## Preservation and scope

Frozen split assignments and source snapshots were not regenerated. Their
implementation hashes remain freeze-time provenance, while new checkpoints
record current source hashes and revision information. The verified decoding
modules must still match their original frozen hashes. All waveform samples,
run outputs, environments and weights remain excluded from the delivery.

The smoke command fits on all frozen training signals and optimizes only its
16-record training subset. Validation/holdout metadata can be read to verify
assignment integrity; those partitions' waveforms are not decoded during smoke.
The time budget is checked between fitting records and batches. A current decode,
optimization batch and final evaluation/checkpoint writes can overrun the deadline.
Failed runs retain diagnostic output and should be retried in a new run directory.

The current model uses unweighted cross-entropy. Class-weight experiments,
augmentation, quality modeling, calibration, deferral, holdout evaluation,
multiple training seeds and a reviewer application are not implemented here.
Optimizer and Torch/Python RNG states are saved; exact training resume is not
implemented. Determinism across different hardware/software is not promised.

## Remaining local verification

Run the tests in the existing WSL environment, then run the README's bounded
smoke-rhythm command using the already audited local ECG files. Review its
run-report.json and history.json before choosing a longer training budget.
Never report this workspace's synthetic tests as actual ECG training outcomes.

Implementation references:

- https://docs.pytorch.org/docs/2.8/generated/torch.nn.LayerNorm.html
- https://docs.pytorch.org/docs/2.8/generated/torch.load.html
- https://docs.pytorch.org/docs/2.8/notes/randomness.html

## Delivery verification

The update helper was tested against a fresh copy of the uploaded source.
Checksum preflight accepted all 10 intended changes; a local README edit was
refused before mutation. An injected write failure restored previously changed
files and removed a newly added file. Successful application preserved all 20
protected files. Reapplying the same payload was idempotent.

The built wheel was installed without dependencies into a separate Python 3.11.16
environment; CLI help passed there, confirming lazy dependency imports.
`uv lock --check --offline` passed; the original lock was not changed.

No Git commits or pushes were made in this workspace.
