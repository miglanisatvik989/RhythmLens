# Reproducing the frozen experiment

## Locked setup

Use uv 0.12.5 and the interpreter version in `.python-version`. Obtain uv from
its official release distribution and verify `uv --version`. The launcher installs
the pinned YAML bootstrap before setup if needed. CPU:

```sh
python3 scripts/manage.py setup --config configs/reference.yaml
```

This syncs `uv.lock` with all extras. `train` includes decoding and plotting
requirements; `dev` includes pinned Hatchling and editables for reproducible builds.
Build dependencies are pinned in `pyproject.toml` too. Both locks specify exact
versions and artifact hashes. No package versions are silently upgraded by setup.

CUDA 12.8 on Linux x86_64 uses a separate lock and environment:

```sh
python3 scripts/manage.py setup --config config.yaml
python3 scripts/manage.py doctor --config config.yaml
```

The second environment syncs `environments/cuda128/uv.lock`; root editable
installation uses `--no-deps --no-sources --no-build-isolation`, preserving CUDA
Torch. Never run root CPU sync against the CUDA environment. Driver/hardware
compatibility is checked by real synthetic forward/backward execution in doctor
and immediately before CUDA training. Doctor reports architecture, compute
capability, memory, CUDA runtime, selected GPU and compiled architectures. The old
`verify_gpu.py` delegates to doctor and requires `--config`. CPU and CUDA environments
can coexist. Do not infer working hardware from the existence of a CUDA lock.

## Restore recorded data, reuse frozen metadata

Acquire the official Challenge 2017 `training2017.zip` and `REFERENCE-v3.csv`
identified by `data/manifests/challenge2017-v3/provenance.json` and `audit.json`.
Follow the recorded source URLs, versions, archive checksums and extraction
layout in those files. Files belong under the existing ignored `data/raw/` tree;
`records.csv` records each required `.hea` and `.mat` path. Local acquisition:

```sh
.venv/bin/rhythmlens acquire --local-archive /path/to/training2017.zip --local-reference /path/to/REFERENCE-v3.csv
python3 scripts/manage.py verify-data --config configs/reference.yaml
```

Acquisition is a separate explicit action; doctor never downloads. Acquisition
refuses inconsistent existing raw files. Do not run `audit` or `split` to reproduce
this release. Do not overwrite tracked manifests. Development integrity verification
checks the frozen source snapshots, all partition metadata and accounting, plus
training raw checksums without decoding ECG. Canonical doctor additionally checks
validation raw checksums. No reproduction command decodes holdout signals.
A missing raw file causes an actionable preflight failure.

Then run doctor and synthetic tests, fit preprocessing, set `preprocessing.reuse`
to its resulting state, run a bounded smoke, and finally explicitly launch training
as in README. Preprocessing fits only training records, checks all fitting IDs and
preserves native 300 Hz physical mV samples, median centering and a fixed training
RMS scale. There is no resampling, per-record amplitude normalization or new loss
or weighting policy. Float32 remains the default.

## Frozen-data compatibility

The audit-time lock and decoder hashes in frozen metadata are immutable historical
provenance. They are not the active-software compatibility gate. The packaged
`src/rhythmlens/data_compatibility.json` contract, version
`challenge2017-physical-v1`, separately approves exact decoder hashes and physical
semantics. Frozen snapshot checksums, label order, assignments, duplicate separation,
complete accounting and per-access raw checksums remain enforced. The active lock,
installed versions, interpreter and contract identity are recorded separately.
A benign lock change does not alter the frozen data identity.

To change compatibility intentionally: review decoder semantics and dependency
changes, run synthetic calibration/unit/shape/checksum/duplicate regression tests,
create a new contract version with reviewed decoder hashes, update the supported
version validator, and document evidence in a new tracked report. Re-resolve the
appropriate lock and verify clean installations. Commit the reviewed software and
contract together; never rewrite audit-time metadata to make a change pass. An
unsupported decoder or contract fails closed. Compatibility is not a claim that
new numerical software produces identical research results.

## Models and evidence

`rhythm-cnn-v1` preserves the local-morphology reference. Default stem kernel 15,
three two-convolution residual stages with kernel 7 and strides 1,2,2 give an
81-sample receptive field (0.27 seconds at 300 Hz). Masked global averaging aggregates
local features but loses their order and cannot explicitly compare widely spaced
beats. Whole-record input alone does not make this a long-context rhythm model.

`rhythm-dilated-v1` is an opt-in temporal candidate (`configs/temporal.yaml`). Eight
masked two-convolution blocks at native rate use dilations 1,2,4,8,16,32,64,128;
its default receptive field is 15 + 2*6*255 = 3,075 samples (10.25 seconds).
This is the architectural/theoretical receptive field; learned effective dependence
may be smaller. Both versions validate prefix masks and mask after every convolution.
Artifacts record the model version, configuration and receptive field. The controlled
pulse-spacing regression is engineering evidence only. No validation comparison
or clinical superiority is claimed; a winner has not been selected.

## Canonical source and artifacts

Full training runs require a Git revision and a clean working tree. An explicit
`execution.noncanonical_development: true` override permits development runs and
is recorded in resolved configuration and preflight provenance. Smoke and
`doctor --smoke` permit dirty source without claiming canonical results. The gate
runs before training or output creation. Use `show` and `--dry-run` to review commands.

Output layout is `<output_root>/<experiment>/<profile>/seed<seed>/<action>/`.
Training saves original `experiment.yaml`, resolved `config.json`, `preflight.json`,
`preprocessing.json`, `status.json`, `history.json`, atomic `best.pt`/`last.pt`,
`run-report.json`, and development-validation metrics/predictions/plot. Checkpoints
record source hashes, Git state, locks, installed versions, data identity, decoder
contract, model capacity, preprocessing and runtime. Weights, environments, ECG
and local outputs stay ignored. Every eligible validation recording is assessed;
AF recall uses all AF labels, with undefined sensitivity when none exist. No deferral
policy is implemented and a deferral must never be interpreted as a negative.

## Interruption and evaluation

See [configuration](configuration.md) for explicit epoch-boundary resume and
cross-device evaluation. Resume restores selection state, optimizer, metric history,
Python/NumPy/Torch CPU/CUDA RNG, sampler epoch and both loader generators. Partial
epochs are repeated; exact mid-epoch continuation is not supported. Hard termination
may leave `running`; `last.pt` still identifies the last atomic completed boundary.
A failure before the first epoch completes has no usable checkpoint: preserve that
run directory and choose a new experiment name. Completed runs cannot be resumed
or silently overwritten. Atomic replacement preserves the previous checkpoint if
serialization fails; lifecycle failure details distinguish exceptions and Ctrl+C.

## Maintainer-only new audit versions

A new audited release is a separate research decision. Use a new manifest location,
split version/config and explicit provenance; audit decoding, labels and duplicate
identity, choose grouping from verified identifiers, and freeze once. The original
`audit`/`split` commands are historical maintainer tools for the original version;
they refuse overwrites and must not be used to regenerate the frozen release.
Changes to these tools/versions require review. Never invent patient identities.
The current dataset lacks verified patient IDs; recording-level splitting cannot
establish patient independence. Keep final holdout untouched while developing all
models, thresholds and analysis choices.

Quality gating, calibration, AF deferral, final holdout evaluation and research
results remain pending. AMP, broader performance tuning and winner selection are
deferred until measurements support them. Licensing, authorship and citation metadata
require owner decisions; no license, DOI or publication is implied by this repository.
Historical reports and `docs/archive/` are evidence of earlier workflows, not current
reproduction instructions. The four baseline/smoke JSON examples remain deprecated
legacy loading fixtures; YAML plus `scripts/manage.py` is canonical.
