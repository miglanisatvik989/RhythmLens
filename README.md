# RhythmLens

RhythmLens is a quality-aware ECG research project exploring how signal quality
affects rhythm analysis, including atrial fibrillation (AF) assessment and explicit
deferral when an ECG cannot be assessed reliably.

This is a retrospective research prototype; clinical diagnostic use is not
validated.

## Status

- Data acquisition, auditing, and frozen split CLI: **implemented**.
- Model implementation: **pending**.
- Experiments and results: **pending**. No performance claims are available.

## Research principles

- Make experiments reproducible with versioned configurations, split manifests,
  seeds, dependencies, and documented evaluation procedures.
- Prevent leakage across training, validation, and test sets: require
  patient/subject grouping when verified identifiers are available. Otherwise,
  use recording-level splits, keep verified duplicates together, and document
  the limitation. Never invent patient identities. Fit preprocessing only on
  training data.
- Reserve an untouched final holdout for evaluation after all development choices
  are frozen.
- Report AF deferrals explicitly, including their counts, rates, reasons, and
  impact on coverage and performance; do not silently exclude deferred cases.
- Define automatic AF sensitivity as correct automatic AF predictions divided
  by all AF-labeled recordings. Deferred AF recordings and incorrect predictions
  do not count as detections. Report this alongside coverage and accepted-case
  metrics; if there are no AF-labeled recordings, report it as undefined.
- Report only observed results, with limitations and uncertainty stated clearly.

## Repository policy

Keep source code, documentation, experiment configurations, and split manifests
under version control. Split manifests should use de-identified identifiers and
contain no ECG samples or secrets.

Keep secrets, raw and processed ECG data, model weights, checkpoints, and local
run outputs out of Git. Use the ignored directories defined in `.gitignore` for
these artifacts. Track configuration and split manifests separately in `configs/`
and `splits/`, rather than inside ignored artifact directories. Keep
`data/manifests/` and `reports/` trackable as well.

Only explicitly designated synthetic test fixtures may be exempted from data
ignore rules when needed. Document their synthetic provenance and allowlist each
exact fixture path in `.gitignore`; do not broadly exempt a fixture directory.

Model implementation and evaluation are pending. See the data pipeline commands below.

## Python environment and installation

Use CPython 3.11.16 and uv 0.12.5. `.python-version` selects 3.11.16; package metadata
currently restricts support to 3.11 until other versions are tested. The setup
host had only Python 3.14.4, so Python 3.11.16 was downloaded without changing
the system Python. No change to the preferred Python version was necessary.

Install the tested uv version using the version-specific standalone installer
from [uv's official installation documentation](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).
On Linux or macOS:

```sh
curl -LsSf https://astral.sh/uv/0.12.5/install.sh | sh
```

Restart your terminal to pick up the installer's PATH changes, or, with the
default installation location in a POSIX shell, run:

```sh
export PATH="$HOME/.local/bin:$PATH"
hash -r
uv --version
```

On Windows, run the version-specific installer in PowerShell, then close and
reopen the terminal before verifying the version:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.12.5/install.ps1 | iex"
# Run in the reopened terminal:
uv --version
```

Verify that the output starts with `uv 0.12.5` (build/platform details may follow).
If another version appears, check `command -v uv` on POSIX or `Get-Command uv`
in PowerShell and ensure the installer's binary directory comes first on PATH.
For a customized installation location, follow the installer's PATH instructions.

From the repository root, reproduce the tested Python patch version and extras:

```sh
uv --version
uv python install 3.11.16
uv sync --python 3.11.16 --locked --all-extras --default-index https://pypi.org/simple
uv pip check --python .venv/bin/python
.venv/bin/rhythmlens --help
.venv/bin/rhythmlens --version
.venv/bin/python -m rhythmlens --version
```

On Windows use `.venv\Scripts\python.exe` and `.venv\Scripts\rhythmlens.exe`.
uv installs managed Python in its default persistent storage (on this Linux
host, `~/.local/share/uv/python`); do not set `UV_PYTHON_INSTALL_DIR` to `/tmp`.
Check `.venv/bin/python -c "import sys; print(sys.base_prefix)"` on POSIX to
verify the underlying interpreter location.
Activation is optional (`source .venv/bin/activate` on POSIX).
`uv.lock` records exact transitive versions, public artifact URLs, hashes, and
platform markers. Keep it under version control; `--locked` refuses stale
metadata rather than silently resolving new versions. The project itself is
installed editable using a relative `.` source. `.venv/` is ignored.

The CLI exposes help/version plus acquisition, audit, and split commands. Its base
installation has no runtime dependencies; audit and split commands require the `audit` extra. Select optional extras as needed:

```sh
uv sync --python 3.11.16 --locked --extra audit --default-index https://pypi.org/simple
uv sync --python 3.11.16 --locked --extra train --extra dev --default-index https://pypi.org/simple
```

Sync installs exactly the selected environment and can remove unselected extras.
The `audit` extra provides WFDB for ECG records and annotations, NumPy/SciPy for
arrays and signal processing, pandas for manifests, Matplotlib for inspection,
and PyYAML for configurations. `train` provides PyTorch, scikit-learn for
splits/baselines/metrics, numerical libraries and configuration parsing. `dev`
provides pytest, Ruff, and pinned build tooling. These are dependencies for
the data pipeline and planned research; models and evaluation are not implemented.

For a conventional pip installation with Python 3.11:

```sh
python3.11 --version  # Verify Python 3.11.16 before continuing
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[audit,dev]'
```

This pip command resolves metadata ranges and does **not** reproduce the exact
lock; use uv sync for reproducible research. After the locked sync above, build distributions with
`.venv/bin/python -m build --no-isolation`.

### CPU PyTorch route and compatibility

The uv configuration routes only torch to the explicit official CPU index on
Linux and Windows. Other dependencies use PyPI; macOS uses its native PyPI torch
wheel. The lock uses torch 2.8.0 (with `+cpu` on Linux/Windows). No torchvision or
torchaudio is needed for this ECG scaffold. For a separate pip-based CPU install
on Linux/Windows, run these commands inside a Python 3.11 environment:

```sh
python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[audit,train,dev]'
python -m pip check
```

The second command keeps the already compatible CPU torch installation. pip does
not interpret `[tool.uv.sources]`; installing the training extra directly from
PyPI can select a different accelerator build. The pip route is not lock-exact.
GPU environments require a separately reviewed configuration and lock.

Compatibility sources checked during setup:

- [SciPy 1.15 release notes](https://docs.scipy.org/doc/scipy-1.15.2/release/1.15.0-notes.html)
  specify Python 3.10–3.13 and NumPy >=1.23.5; the selected Python 3.11,
  NumPy 2.2, and SciPy 1.15 series satisfy these requirements.
- [WFDB installation documentation](https://wfdb.readthedocs.io/en/stable/installation.html)
  and [WFDB 4.3.1 metadata](https://github.com/MIT-LCP/wfdb-python/blob/v4.3.1/pyproject.toml)
  describe installation and dependency constraints; the resolver checks their
  intersection with the other extras.
- [Official PyTorch 2.8 installation commands](https://pytorch.org/get-started/previous-versions/#v2-8-0)
  document the CPU index. Platform wheel availability is resolved in the lock;
  execution is verified on Linux x86_64 only, not every locked platform.

See [the environment setup report](reports/environment-setup.md) for actual
versions, verification commands, and setup failures. No datasets are required
or downloaded for environment verification. Future runs must record seeds,
provenance, preprocessing, splits, configuration, and evaluation procedures as
required by AGENTS.md; this scaffold does not prescribe experimental choices.

## Challenge 2017 acquisition, audit, and frozen splits

Install the locked environment with the `audit` extra (plus `dev` for tests),
then run from this repository's root:

```sh
uv sync --python 3.11.16 --locked --extra audit --extra dev --default-index https://pypi.org/simple
.venv/bin/rhythmlens acquire
.venv/bin/rhythmlens audit
.venv/bin/rhythmlens split
```

`acquire` downloads only `training2017.zip` and the separately published
`REFERENCE-v3.csv` from the [official version 1.0.0 source](https://physionet.org/content/challenge-2017/1.0.0/).
If the primary host is slow, use `.venv/bin/rhythmlens acquire --mirror s3`
for the official `physionet-open` bucket listed on the same dataset page.
Provenance records both the actual download URL and the canonical PhysioNet URL.
Existing cached files are reused and checked against recorded SHA-256 hashes.
To reuse downloads elsewhere:

```sh
.venv/bin/rhythmlens acquire --local-archive /path/to/training2017.zip --local-reference /path/to/REFERENCE-v3.csv
```

Downloads and extracted originals remain in ignored `data/raw/challenge2017/`.
Extraction preflights all paths, rejects traversal and symlinks, and publishes
an extraction only after completion. Existing extractions are not overwritten;
audit checks extracted file contents against the archive. Partial downloads
are never published as completed files. Provenance in
`data/manifests/challenge2017-v3/provenance.json` records official URLs, dataset
and label versions, byte sizes, SHA-256 checksums, acquisition method, UTC download
time and license. For supplied local files the original download time is unknown
and explicitly null; observation time is recorded separately. Checksums describe
the observed files, not an independent publisher-signed authenticity guarantee.

`audit` joins both references by record ID with label order **N, A, O, ~**.
It recomputes counts and records every archive-to-v3 label change. The expected
format is one-channel, 300 Hz, MATLAB V4 with a WFDB header. SciPy loads `val`;
WFDB supplies header semantics and independently reads digital samples to check
payload/offset agreement. Header sample count, signal name, format, gain,
baseline, units, finite samples and integer range are validated. Optional
initial-value and checksum fields are compared and recorded. In this archive all
header checksums are zero; the pipeline flags them as unverified instead of
treating them as integrity evidence. Initial-value discrepancies are advisory
for format 16, where that field is not used to decode amplitudes. A nonzero
checksum mismatch remains a blocker. Original-byte agreement with the archive
and independent MAT/WFDB payload agreement are required. Physical values are `(ADC - baseline) / gain`, converted to mV;
an absent baseline uses the header ADC zero. Calibration and units must be
explicit. Original files are never normalized or rewritten.

Missing pairs, orphan records, invalid labels, malformed records and nonfinite
signals are reported and block split creation. Format-16 missing-value sentinels
are also flagged. Flat means all decoded samples are exactly equal: those
records are flagged and retained, as are naturally noisy `~` records. There is
no amplitude/quality cutoff. Every label or MAT/header record appears in the
inventory; errors never silently remove cases. Unknown subject IDs are written
as `unknown` rather than inferred from recording names.

Exact duplicate groups hash canonical float64 mV samples plus channel count,
sampling rate and length. This detects equivalent physical signals across MAT
containers and calibrations; a second digest records digital values with gain,
baseline and original units. Near duplicates are not inferred. Conflicting v3
labels in an exact group are reported and block freezing, requiring an explicit
reviewed policy in a future version.

Audit artifacts:

- `configs/challenge2017-audit-v1.json`: decoding and validation policy.
- `data/manifests/challenge2017-v3/records.csv`: full record inventory and flags.
- `data/manifests/challenge2017-v3/audit.json`: findings, changed labels, duplicates,
  input hashes, software versions and code hashes.
- `reports/challenge2017-audit.md`: actual counts, durations and blockers.
- `outputs/challenge2017/previews/`: ignored waveform PNGs for the first five
  valid record IDs per class (or fewer if unavailable), generated before freezing.

Use `rhythmlens audit --no-previews` to skip images before freezing. Audit exits
nonzero on validation errors, after writing its findings. Once splits exist,
audit refuses to replace their source snapshot or generate more previews.

`split` requires a passing, unchanged audit and retains all eligible records.
It groups exact duplicates, sorts groups deterministically, uses seed 42 for
size ties, and assigns each class's groups to the largest remaining target
count deficit for approximately 70/15/15 train/validation/holdout proportions.
Class coverage is enforced; fewer than three groups in a class is a blocker.
Indivisible groups can prevent exact proportions. Record and duplicate-group
overlap are checked explicitly, and actual class/overall proportions are saved.
These are **recording-level** splits. Patient and longitudinal independence
cannot be established from the available identifiers. The single data source
also does not establish external-source generalization.

Frozen files are under `splits/challenge2017-v3-seed42-v1/`: `train.csv`,
`validation.csv`, `holdout.csv`, `config.json`, `summary.json`, `README.md`, and
`frozen.json`. A matching configuration lives in `configs/`. Frozen metadata
binds manifest, audit, provenance, source code, dependency lock, configuration
and partition hashes. Existing frozen outputs are refused, with no force flag.
Future changes require an explicitly reviewed version and implementation update.
Use validation for development; keep holdout untouched until all model,
threshold, quality-gating and analysis choices are frozen. No AF predictions,
deferrals or performance metrics are produced by this data pipeline.

### Dataset attribution and decoding references

Challenge data were donated by AliveCor and published on PhysioNet, dataset
DOI [10.13026/d3hm-sf11](https://doi.org/10.13026/d3hm-sf11). The dataset page lists
[Open Data Commons Attribution License v1.0](https://physionet.org/content/challenge-2017/view-license/1.0.0/)
(ODC-By-1.0). Preserve this attribution and license notice with derived manifests
and reports. The dataset license is separate from any future RhythmLens code
license.

Cite Clifford GD et al., *AF classification from a short single lead ECG
recording*, Computing in Cardiology (2017),
[doi:10.22489/CinC.2017.065-469](https://doi.org/10.22489/CinC.2017.065-469).
The current dataset page also requests the PhysioNet citation: Pollard T et al.,
*PhysioNet as a global platform for biomedical research*, Nature Health (2026),
[doi:10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z).

Decoding follows the official [WFDB header specification](https://physionet.org/physiotools/wag/header-5.htm)
and [SciPy 1.15.3 loadmat documentation](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.io.loadmat.html).
The dated class table on the challenge page is not used as the v3 ground truth;
counts are always recomputed from the separately acquired reference.

### Verification

```sh
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```

Tests generate synthetic MAT/header signals in temporary directories using
explicit generator seeds and exercise the real acquisition, audit and split
functions. No synthetic sample files are stored in Git. Synthetic test outcomes
and actual dataset findings are reported separately in
`reports/challenge2017-verification.md` and `reports/challenge2017-audit.md`.
