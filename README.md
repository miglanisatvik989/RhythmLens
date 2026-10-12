# RhythmLens

RhythmLens is a quality-aware ECG research project exploring how signal quality
affects rhythm analysis, including atrial fibrillation (AF) assessment and explicit
deferral when an ECG cannot be assessed reliably.

This is a retrospective research prototype; clinical diagnostic use is not
validated.

## Status

- Implementation: **pending**.
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

Data acquisition, model implementation, and evaluation are pending.

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

The CLI currently exposes only package help and version information. Its base
installation has no runtime dependencies. Select optional extras as needed:

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
planned work; no pipeline, models, or evaluation are implemented yet.

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
