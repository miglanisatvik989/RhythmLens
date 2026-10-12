# Environment setup verification

Setup date: 2026-10-12. Host: Linux x86_64. Read AGENTS.md and inspected all
initial repository files (AGENTS.md, README.md, .gitignore); no package existed.
System Python: 3.14.4. Project Python: CPython 3.11.16. uv: 0.12.5.
Python 3.11 was retained as requested. Only Linux x86_64 was executed;
other platform markers in the portable lock are not runtime validation.

## Commands and outcomes

Inspection: `pwd`, `rg --files` (including hidden files excluding .git/.venv),
`cat AGENTS.md`, `cat README.md`, `cat .gitignore`, `git status --short`,
`command -v python python3 python3.11 uv`, `python3 --version`,
`ls /usr/bin/python*`, `uv --version`, `uv python list --only-installed`,
`uname -sm`.

Initial setup commands below were run with `UV_CACHE_DIR=/tmp/rhythmlens-uv-cache`
and, where interpreter discovery was needed,
`UV_PYTHON_INSTALL_DIR=/tmp/rhythmlens-python`. These are local setup locations,
not requirements for other users; the portable README commands use uv defaults.

```sh
uv python install 3.11
uv lock --default-index https://pypi.org/simple
uv sync --locked --all-extras --default-index https://pypi.org/simple
uv pip check --python .venv/bin/python
.venv/bin/rhythmlens --help
.venv/bin/rhythmlens --version
.venv/bin/python -m rhythmlens --version
.venv/bin/ruff check src
.venv/bin/python -m build --no-isolation
uv venv --python .venv/bin/python /tmp/rhythmlens-wheel-check
uv pip install --python /tmp/rhythmlens-wheel-check/bin/python --no-deps dist/rhythmlens-0.1.0-py3-none-any.whl
/tmp/rhythmlens-wheel-check/bin/rhythmlens --version
/tmp/rhythmlens-wheel-check/bin/python -m rhythmlens --help
uv lock --check --offline --default-index https://pypi.org/simple
git check-ignore .venv/pyvenv.cfg
```

All verification commands passed. Dependency check: 57 compatible installed
packages. Lock: 59 entries (including platform alternatives). CLI version:
`rhythmlens 0.1.0`. Built both sdist and wheel; installed the wheel into a separate
clean environment with no dependencies and verified CLI execution.
Ruff passed. No model tests or research evaluation were performed.

An inline Python smoke check (with `MPLCONFIGDIR=/tmp/rhythmlens-matplotlib`)
imported scipy.signal, pandas, wfdb, matplotlib, sklearn, yaml, NumPy and torch;
verified a float64 NumPy-to-CPU-tensor-to-NumPy round trip and
`torch.version.cuda is None`. It read uv.lock with tomllib and checked every
package has an exact version, registry/artifact URLs use HTTPS without userinfo
or query credentials, artifacts have hashes, the editable project source is
relative `.`, and the lock has no `/home/`, `/tmp/`, `file://`, `token=`, or
`password=` strings. These assertions passed. `.venv/pyvenv.cfg` is ignored.
The lock uses public PyPI and official PyTorch URLs. Full installed versions
are recorded below; all locked platform versions and artifact hashes are in
uv.lock. See README for official compatibility sources and CPU install commands.

## Failures and adjustments

- `uv python list --only-installed` initially failed because the default uv
  cache was read-only in this sandbox. Subsequent uv commands used the writable
  temporary cache above.
- The first sandboxed Python download failed after three retries with DNS
  resolution errors. Retried with network approval; Python download, dependency
  resolution and installation succeeded with approved network access.
- Installation warned that hardlinks across cache/environment filesystems failed;
  uv automatically copied files and installation succeeded. Use `UV_LINK_MODE=copy`
  if desired on such hosts.
- The first module invocation printed `__main__.py 0.1.0`; the CLI program name
  was made explicit and the rebuilt wheel now consistently prints `rhythmlens`.

The initial managed interpreter was under /tmp. The persistent-storage correction
below replaces that environment; the current .venv no longer depends on /tmp.
No datasets, weights, models, experiments, commits, or pushes were created.
Build outputs are in ignored dist/; the separate wheel check is under /tmp.

## Persistent interpreter correction

Pinned `.python-version` to `3.11.16` and aligned README commands, including
optional-extra sync commands, with that patch version. README retains official
version-specific uv 0.12.5 installation commands, terminal/PATH steps, and
`uv --version` verification.

Inspection of the original `.venv/pyvenv.cfg` and resolved `.venv/bin/python`
confirmed its interpreter was under `/tmp/rhythmlens-python`. Installed the same
CPython 3.11.16 into uv's default persistent location using approved access
outside the sandbox. No workspace tools directory or new ignore rule was needed.
Recreated only the disposable `.venv`; source files and research documentation
were preserved. The initial setup history above remains for provenance.

Actual correction and verification commands:

```sh
cat .python-version
cat .venv/pyvenv.cfg
readlink -f .venv/bin/python
uv --version
uv python dir
sha256sum uv.lock
UV_CACHE_DIR=/tmp/rhythmlens-uv-cache uv python install 3.11.16
UV_CACHE_DIR=/tmp/rhythmlens-uv-cache uv venv --clear --python /home/kaizen/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11 .venv
UV_CACHE_DIR=/tmp/rhythmlens-uv-cache UV_LINK_MODE=copy uv sync --python 3.11.16 --locked --all-extras --offline --default-index https://pypi.org/simple
UV_CACHE_DIR=/tmp/rhythmlens-uv-cache uv pip check --python .venv/bin/python
.venv/bin/rhythmlens --help
.venv/bin/rhythmlens --version
.venv/bin/python -m rhythmlens --version
sha256sum uv.lock
git check-ignore .venv/pyvenv.cfg
git diff --check
```

`uv --version` returned `uv 0.12.5 (x86_64-unknown-linux-gnu)`.
Python installation succeeded, with a warning that the existing
`/home/kaizen/.local/bin/python3.11` launcher was not managed by uv and could not
be replaced. It was left untouched; explicit persistent interpreter selection
avoided relying on that launcher. No correction command failed.

The recreated environment installed the same 57 packages from the existing
lock/cache. Dependency consistency passed, CLI help and both version entry
points passed (`rhythmlens 0.1.0`). An inline Python check verified Python
`3.11.16`, resolved `sys.base_prefix`, `sys._base_executable`, and
`.venv/bin/python` outside `/tmp`, and observed:

```text
sys.base_prefix = /home/kaizen/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu
base executable = /home/kaizen/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
resolved .venv/bin/python = /home/kaizen/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
```

NumPy 2.2.6 / PyTorch 2.8.0+cpu float64 array-to-tensor-to-array equality passed;
the tensor device was CPU and `torch.version.cuda` was `None`.
`.venv/pyvenv.cfg` remains ignored. The temporary uv cache is only an installation
cache; the recreated runtime does not depend on it. All installed versions below
remain unchanged. Lock SHA-256 before and after was identical:
`4ff3ef044d9e70d26b8a5c0494478e30cbc3f73ac3ecd94c109e1dd1adf44c8d`.
No datasets, models, commits, or pushes were made.

## Installed distributions

```text
aiohappyeyeballs==2.7.1
aiohttp==3.14.5
aiosignal==1.4.0
attrs==26.1.0
build==1.6.1
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.2
cloudpickle==3.1.2
contourpy==1.3.3
cycler==0.12.1
filelock==4.1.1
fonttools==4.66.1
frozenlist==1.8.0
fsspec==2026.9.0
hatchling==1.27.0
idna==3.20
iniconfig==2.3.1
Jinja2==3.1.6
joblib==1.6.0
kiwisolver==1.5.1
MarkupSafe==3.0.4
matplotlib==3.10.9
mpmath==1.3.0
multidict==7.1.0
networkx==3.6.1
numpy==2.2.6
packaging==26.3
pandas==2.2.3
pathspec==1.1.1
pillow==12.3.0
pluggy==1.6.0
propcache==0.5.4
pycparser==3.11
Pygments==2.21.0
pyparsing==3.3.3
pyproject_hooks==1.3.3
pytest==8.4.2
python-dateutil==2.9.0.post0
pytz==2026.5
PyYAML==6.0.3
requests==2.34.2
rhythmlens==0.1.0
ruff==0.11.13
scikit-learn==1.6.1
scipy==1.15.3
six==1.17.0
soundfile==0.14.0
sympy==1.14.0
threadpoolctl==3.7.0
torch==2.8.0+cpu
trove-classifiers==2026.9.21.13
typing_extensions==4.16.0
tzdata==2026.5
urllib3==2.8.0
wfdb==4.3.1
yarl==1.25.1
```
