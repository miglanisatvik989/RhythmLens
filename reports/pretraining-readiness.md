# Pretraining readiness — 2026-10-12

## Verdict

Engineering verification passed on the prepared working tree: **162 synthetic tests**,
Ruff lint/format, both lock checks, dependency checks, clean editable and wheel
installations, CLI/config routing, configured CPU preflight, actual synthetic CUDA
execution, frozen hashes and ignore rules. This is **conditional pretraining readiness,
not a fully ready canonical run**. Source remains intentionally uncommitted and dirty,
so the canonical gate rejects training. The preserved original CPU environment needs
locked setup to add editables; clean temporary CPU environments already pass. CI has
been added but has not run on GitHub; no green badge or CI success is claimed.

No full baseline training, real-data training smoke, dataset download, commit, push,
quality gate, calibration, AF deferral mechanism or final holdout evaluation was run.
Synthetic fixtures were generated in pytest temporary directories. Doctor only
checks training raw checksums and metadata and runs generated tensors; no ECG was
decoded by doctor. No real holdout signals were opened or decoded. Holdout metadata
was inspected for integrity and accounting only.

## Scope and changes

The initial working tree already contained modified README/CLI files and substantial
untracked training, configuration, CUDA and documentation work. It was preserved
and extended without reset. Historical reports were left byte-for-byte unchanged.
The optional `reports/repository-review-2026-10-12.md` was absent. AGENTS.md and the
existing README, configuration, environment, baseline and audit/split documentation
were inspected before implementation.

- **Frozen-data compatibility:** introduced packaged contract
  `challenge2017-physical-v1`, independent of the current lock. Original decoder
  files remain unchanged. Audit-time lock/source hashes remain immutable historical
  data identity; active locks, installed packages and decoder contract are separately
  recorded and checked by runtime preflight. Snapshot, label, membership, duplicate
  and record-accounting checks remain active, as do raw-file checksums before decoding.
  Meaningful verified subject IDs are checked for cross-partition leakage; the
  literal `unknown` sentinel is never treated as an identity.
- **Sampling and RNG:** randomize the remainder before length sorting during
  training, retain every record once, preserve local seed-plus-epoch behavior and
  length buckets. Each loader owns an explicitly seeded Torch generator. Validation
  iteration does not consume global training CPU RNG. Resume captures both generators.
- **Model versions:** preserve `rhythm-cnn-v1`; separately add `rhythm-dilated-v1`.
  Default architectural receptive fields are 81 samples / 0.27 seconds and 3,075
  samples / 10.25 seconds at 300 Hz, respectively. Global masked averaging limits the
  reference to aggregation of local morphology. Candidate dilations are
  1,2,4,8,16,32,64,128 with two convolutions per block. YAML experiments, artifact
  model versions/configurations and receptive fields are explicit. Learned effective
  dependence can be smaller than the architectural receptive field. No winner or AF
  performance advantage is claimed. Timing sensitivity is engineering evidence only.
- **Checkpoints:** validate schema, model version, task, labels, field types,
  model/config agreement, preprocessing semantics/identity, data identity and
  continuation state. Preserve `weights_only=True`, strict parameter loading,
  finite-parameter checks and known legacy reference inference artifacts. New
  identity checksums catch metadata corruption. Checkpoints use atomic replacement
  and fsync; a failed save preserves the previous valid file.
- **Evaluation:** current YAML selects its interpreter, profile, CPU/CUDA device
  and index independently of immutable checkpoint training provenance. CUDA-labelled
  synthetic checkpoints evaluate on CPU. GPU-index routing is tested with mocks;
  that test is distinct from actual CUDA execution below. Evaluation remains
  development-validation-only; smoke and holdout bypasses are rejected.
- **Lifecycle/resume:** atomic running/completed/failed/interrupted records contain
  timestamps, invocation, UUID, configuration hashes, completed epoch, observed step
  count, last usable checkpoint and failure details. Epoch checkpoints include model,
  optimizer, selection best/epoch/patience/history, Python/NumPy/Torch CPU/CUDA RNG,
  sampler epoch and loader RNG states. Resume validates configuration, model,
  preprocessing, data, source bundle, runtime, packages and locks. `--resume` is an
  explicit launcher/CLI action; the YAML action flag is also supported. Original
  YAML/config/provenance remain preserved. The invoked YAML is separately hashed,
  rather than conflated with source hashes; only its resume action can change.
- **Environment and workflow:** train extras now include WFDB and plots; Hatchling
  1.27.0 and editables 0.5 are pinned as build requirements and locked dev tools.
  CPU lock adds editables and updates extra metadata without upgrading existing
  packages. CUDA lock/descriptor remain unchanged; CUDA editable installation retains
  `--no-deps --no-sources --no-build-isolation`. Runtime Torch expectations derive
  from the selected lock, and interpreter setup derives from `.python-version`.
  YAML plus `scripts/manage.py` is canonical; strict settings include evaluation
  runtime and resume. Legacy JSON examples/loading remain as deprecated compatibility
  fixtures. No historical audit configuration was removed.
- **Preflight:** selected interpreter, dependency consistency, lock/descriptor
  agreement, device/index, synthetic execution, decoder compatibility, frozen
  integrity, local training/validation availability/checksums, preprocessing reuse,
  source state, writable output and collisions are checked before training. CUDA
  requests always execute synthetic forward/backward. Full canonical runs require
  clean committed source and resolved YAML. An explicit recorded development
  override is available; smoke permits development source. Existing run directories
  cannot silently restart; completed runs reject resume.
- **Documentation/CI:** concise README; detailed reproduction and configuration
  guides; prior workflows archived with historical banners. Fresh clones restore
  recorded raw files and reuse frozen manifests, separately from maintainer audit
  creation. Added CPU-only CI for locked setup, lint/format, synthetic tests, frozen
  and lock/ignore contracts, build, clean wheel install and CLI/import checks.
  Added CONTRIBUTING.md. Standalone GPU verification now delegates to configured
  doctor, retaining architecture, capability, execution and memory diagnostics.

Relevant files: `src/rhythmlens/{compatibility,lifecycle,preflight,configuration,
rhythm_data,rhythm_model,rhythm_training,preprocessing,cli}.py`, packaged
`data_compatibility.json`, `scripts/{manage,doctor_environment,verify_frozen,
check_repository_contracts}.py`, `configs/{reference,temporal}.yaml`, `config.yaml`,
`pyproject.toml`, `uv.lock`, `tests/test_pretraining_readiness.py`, existing baseline
and configuration tests, `.github/workflows/cpu.yml`, README, CONTRIBUTING,
`docs/`, CUDA verifier/README and this report. Existing inference/data/audit/split
implementations and historical configs/reports are preserved.

## Frozen evidence and compatibility policy

Before editing, 17 files under frozen manifests/splits, historical reports and
audit configs were hashed to `/tmp/rhythmlens-frozen-before.json`. The exact snapshot
is now trackable as `reports/frozen-provenance-sha256.json`. After work:

```sh
python3 scripts/verify_frozen.py
```

Observed: `unchanged_files: 17`, train **5,969**, validation **1,280**, holdout
**1,279**, `ECG_files_opened: 0`. The decoder hashes still equal the approved
contract. Original audit-time CPU lock hash remains
`4ff3ef044d9e70d26b8a5c0494478e30cbc3f73ac3ecd94c109e1dd1adf44c8d` in historical metadata.
Active CPU lock is
`71231d200792985f74f8a2c9ebff633621ad86d6efcc59b2b0063b5a2301c86d`.
Unchanged CUDA lock is
`2c5ed7cb262d94c1e2d334af9955de796ddf43e8d8be73f0c32976cdc75474a1`.

An intentional compatibility change requires review of decoder physical units,
shape, checksum and duplicate semantics, synthetic regression evidence, a new
contract version and approved hashes, an updated version validator, separate lock
review/clean installation checks and a tracked rationale. Never rewrite frozen
metadata to accommodate software changes. Contract and lock compatibility are
software gates, not claims of unchanged clinical or numerical performance.

## Exact verification commands and outcomes

Executed with uv **0.12.5**, Python **3.11.16**, CPU Torch **2.8.0+cpu**, NumPy
**2.2.6**. Temporary installations preserve both original environments.

Clean editable installation (network/cache access was approved after sandbox
network/cache failures):

```sh
UV_PROJECT_ENVIRONMENT=/tmp/rhythmlens-readiness-editable uv sync --locked --all-extras --python .venv/bin/python --default-index https://pypi.org/simple
UV_CACHE_DIR=/tmp/rhythmlens-uv-cache uv pip check --python /tmp/rhythmlens-readiness-editable/bin/python
```

Passed: editable package built and installed; 58 installed distributions including
editables 0.5; dependency consistency passed. CPU Torch remained 2.8.0+cpu.

```sh
/tmp/rhythmlens-readiness-editable/bin/ruff check src tests scripts environments/cuda128/verify_gpu.py
/tmp/rhythmlens-readiness-editable/bin/ruff format --check src tests scripts environments/cuda128/verify_gpu.py
MPLCONFIGDIR=/tmp/rhythmlens-matplotlib /tmp/rhythmlens-readiness-editable/bin/python -m pytest -q tests --tb=short
UV_CACHE_DIR=/tmp/rhythmlens-uv-cache uv lock --check --offline --default-index https://pypi.org/simple
UV_CACHE_DIR=/tmp/rhythmlens-uv-cache uv lock --check --offline --project environments/cuda128
python3 scripts/check_repository_contracts.py
python3 scripts/verify_frozen.py
```

Passed: all lint checks, 24 formatted Python files, **162 tests**, both exact locks
(60 CPU / 74 CUDA resolution entries, not installed-package counts), no credentials
or absolute machine paths in locks, separate CPU/CUDA Torch sources, expected ignore
rules, and unchanged frozen/history bytes. Tests cover 65/16 and 5,969/16 unbiased
remainders, accounting, deterministic epochs, loader RNG isolation, supported lock
changes, raw/manifest/label/membership/group corruption, both models' forward/backward,
padding and checkpoint parity, controlled pulse timing, malformed checkpoint metadata,
legacy loading, lifecycle failures/Ctrl+C, deterministic CPU resume including dropout,
partial-epoch replay, incompatible resume, atomic-save failure, cross-device evaluation
and mocked GPU-index/launcher routing.

Build and clean wheel installation:

```sh
/tmp/rhythmlens-readiness-editable/bin/python -m build --no-isolation --outdir /tmp/rhythmlens-readiness-dist
UV_CACHE_DIR=/tmp/rhythmlens-uv-cache uv venv /tmp/rhythmlens-readiness-wheel --python .venv/bin/python
UV_PROJECT_ENVIRONMENT=/tmp/rhythmlens-readiness-wheel uv sync --locked --all-extras --no-install-project --python .venv/bin/python --default-index https://pypi.org/simple
UV_CACHE_DIR=/tmp/rhythmlens-uv-cache uv pip install --python /tmp/rhythmlens-readiness-wheel/bin/python --no-deps --reinstall-package rhythmlens /tmp/rhythmlens-readiness-dist/rhythmlens-0.1.0-py3-none-any.whl
UV_CACHE_DIR=/tmp/rhythmlens-uv-cache uv pip check --python /tmp/rhythmlens-readiness-wheel/bin/python
cd /tmp
/tmp/rhythmlens-readiness-wheel/bin/rhythmlens --help
/tmp/rhythmlens-readiness-wheel/bin/rhythmlens --version
/tmp/rhythmlens-readiness-wheel/bin/python -c 'import rhythmlens; import rhythmlens.rhythm_training; from rhythmlens.compatibility import validate_decoder_contract; validate_decoder_contract(); print(rhythmlens.__file__)'
```

Passed: sdist and wheel built; clean lock-populated environment installed wheel
without an editable source package; dependency check passed; CLI version 0.1.0,
help/import/packaged-contract checks passed from `/tmp`, with import resolving to
`/tmp/rhythmlens-readiness-wheel/lib/python3.11/site-packages/rhythmlens/__init__.py`.
CI uses this direct-lock approach to preserve explicit index selection.

Configuration and preflight from repository root:

```sh
/tmp/rhythmlens-readiness-editable/bin/python scripts/manage.py show --config configs/reference.yaml
/tmp/rhythmlens-readiness-editable/bin/python scripts/manage.py show --config configs/temporal.yaml
/tmp/rhythmlens-readiness-editable/bin/python scripts/manage.py train --config configs/reference.yaml --dry-run
/tmp/rhythmlens-readiness-editable/bin/python scripts/manage.py train --config configs/reference.yaml --resume --dry-run
/tmp/rhythmlens-readiness-editable/bin/python scripts/manage.py evaluate --config config.yaml --dry-run
/tmp/rhythmlens-readiness-editable/bin/python scripts/doctor_environment.py --config configs/reference.yaml --smoke
/tmp/rhythmlens-readiness-editable/bin/python scripts/doctor_environment.py --config configs/reference.yaml
```

Resolved configs and dry-run routes passed, including CPU evaluation interpreter
for CUDA training provenance and explicit resume routing. Development CPU doctor
passed real synthetic forward/backward and checksum-verified 11,938 training files
(5,969 MAT/header pairs). Preprocessing is explicitly pending a training-only fit
when reuse is null. Canonical doctor **rejected dirty source as expected**, before
training or run creation. This rejection is an operational blocker, not a bypassed
check. Invalid config/device/output/reuse cases are covered by synthetic tests.

Actual CUDA verification, outside sandbox after sandbox reported no visible GPU:

```sh
environments/cuda128/.venv/bin/python scripts/doctor_environment.py --config config.yaml --smoke
```

Passed: actual CUDA synthetic forward/backward, finite loss and gradients, CUDA
synchronization and configured GPU 0 selection. Observed hardware/software:

| Evidence | Observed value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| Compute capability | 12.0 |
| Torch / CUDA runtime | 2.8.0+cu128 / 12.8 |
| Python / NumPy | 3.11.16 / 2.2.6 |
| Device / Torch CPU threads | cuda:0 / 2 |
| Compiled architectures | sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120 |
| Peak allocated bytes | 82,049,024 |
| Free / total device bytes at observation | 7,216,300,032 / 8,546,484,224 |
| Platform | Linux 6.18.33.1 Microsoft WSL2, x86_64 |

Memory figures are point-in-time diagnostics, not performance benchmarks. This
command checksum-checked training files and validated frozen metadata; it decoded
no ECG, performed no real-data training and accessed no holdout signals. The mocked
index-1 routing test is separate and does not claim hardware execution on GPU 1.

Initial failed attempts were resolved: offline editables metadata was absent;
sandbox package downloads failed DNS; approved cache/network access installed the
pinned helper. A plain lock export omitted explicit CPU Torch index information and
wheel dependency sync failed; direct `uv sync --no-install-project` fixed this and is
used by CI. Synthetic tests found a small-dataset sampler zero-step edge case, the
`unknown` subject sentinel, and active-YAML/source-bundle handling during resume;
all were fixed before the final passing suite. Earlier failures are not hidden or
reported as successful runs.

Final repository checks:

```sh
python3 scripts/check_repository_contracts.py
python3 scripts/verify_frozen.py
git diff --check
git status --short
```

Passed contracts, frozen comparison and whitespace checks. Status contains only
prepared source/config/docs/tests/CI/report changes and preserved initial work;
environments, ECG samples, weights and local outputs remain ignored. No changes
were staged or committed.

## Resume guarantees and limits

Deterministic CPU synthetic training with dropout matches uninterrupted training
exactly in checkpoint weights, best selection state and metric history after a
simulated exception or Ctrl+C. Optimizer and all applicable RNG state are restored
from the completed boundary. Partial-epoch updates are discarded and that epoch is
repeated. There is no exact mid-epoch continuation claim. Changed learning settings,
data, preprocessing, model, source bundle, lock/package environment or runtime are
rejected. A YAML-only resume action can have a new committed revision while retaining
the same compatible source bundle; original training provenance remains preserved.

Runs failing before the first usable checkpoint need a new output name. Hard kill
may leave status `running`; the last atomically saved checkpoint remains authoritative.
Completed runs cannot be resumed or silently overwritten. CUDA RNG state is stored,
but deterministic CUDA resume equivalence was not tested here; CPU/GPU equivalence
and hardware-independent numerical identity are not promised. Legacy known reference
artifacts support inference, not newly invented resume state.

## Remaining owner actions and deferred work

1. Review prepared changes together with the preserved initial uncommitted work,
   then commit them; canonical training must retain a clean source state.
2. Run locked CPU setup when using the preserved original `.venv` (or CPU evaluation)
   to install editables. CUDA setup remains separate and must not sync the root CPU
   environment into the CUDA environment.
3. Choose the reference/candidate YAML and runtime, fit or verify reusable training
   preprocessing, run the bounded training-only smoke explicitly, and then launch
   full training only as a separate user action. None of those real-data training
   results are claimed here. Validation evidence is still required for model selection.
4. Owner decisions remain for LICENSE and CITATION.cff, authorship and any publication
   metadata. No license, author identity, DOI or publication was invented. These are
   publication decisions, not unfinished engineering fixes.

Patient independence remains unverified: identifiers are `unknown`, so recording
splits keep verified duplicate groups together but cannot establish subject/source/time
independence. Quality gating, calibration, AF deferrals, frozen final-analysis choices,
holdout evaluation and research results remain pending. Classifier metrics explicitly
report all-assessed coverage, zero implemented deferrals, and automatic AF sensitivity
with all AF-labeled recordings as denominator (undefined if no AF labels). Float32,
loss, class weighting and valid preprocessing semantics are preserved. AMP and broader
performance tuning are deferred until measurements justify them. No Docker, DVC,
tracking service or distributed-training infrastructure was added.
