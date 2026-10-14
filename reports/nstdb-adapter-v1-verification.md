# NSTDB adapter v1 implementation verification

Date: 2026-10-14. Parent proposal revision:
`39916c1fe0698b32ce1173724737aae1af7ee787` (PR #7, open draft and unmerged).
Implementation branch: `feat/nstdb-adapter`; stacked PR base: `feat/noise-robustness`.
This report describes synthetic implementation verification, **not real acquisition,
a digital audit of the public release, or experimental robustness results**.

Implemented in [nstdb.py](../src/rhythmlens/nstdb.py): explicit local import or
explicit versioned HTTPS download of the nine noise-only files; publisher index
and required RECORDS membership checks; preserved bytes and compact acquisition
provenance; exclusive destinations; separate format-212 digital decoder;
header/channel/rate/length/initial/checksum/sentinel audit; partial failure evidence,
role descriptors, audit hash index and source-revalidating pool loader.

Implemented in [noise.py](../src/rhythmlens/noise.py): isolated role slicing before
360→300 polyphase resampling and trimming; contiguous deterministic stress/training
plans with exact proposal keys and draw orders; global NumPy/Torch RNG isolation;
unitless centered unit-RMS shape scaling into physical mV; requested/achieved SNR,
added amplitude and flat-case metadata; length/plan/pool/partition failures without
substitution, redraw or overwrite. No experiment runner, metrics engine, complete
case ledger, preprocessing fit, model loading, calibration or quality gating was
added. The future runner must establish array source/partition/units, preserve
complete eligible accounting and record post-transform hashes.

## Actual checks

Existing `.venv` only; no installation/bootstrap. Python 3.11.16, NumPy 2.2.6,
SciPy 1.15.3, WFDB 4.3.1 and Torch 2.8.0+cpu. Dedicated tests generate temporary
synthetic digital fixtures, including full expected lengths, and use independent
WFDB digital decoding as a cross-check. All acquisition transport tests are mocked;
autouse guards forbid network access. No real ECG, NSTDB, checkpoint or prediction
inputs are used.

| Check | Observed result |
| --- | --- |
| `.venv/bin/python -m pytest -q tests/test_nstdb.py tests/test_noise.py` | 47 passed |
| `.venv/bin/python -m pytest -q tests` | 320 passed |
| `.venv/bin/ruff check src tests scripts environments/cuda128/verify_gpu.py` | Passed |
| `.venv/bin/ruff format --check src tests scripts environments/cuda128/verify_gpu.py` | Passed; 33 files formatted |
| `.venv/bin/python scripts/check_repository_contracts.py` | Passed: ignore/trackability, portable credential-free CPU/CUDA locks |
| `.venv/bin/python scripts/verify_frozen.py` | Passed: 17 frozen identities unchanged; train/validation/holdout counts 5969/1280/1279; zero ECG files opened |
| `.venv/bin/python -m rhythmlens.nstdb --help` | Only implemented `acquire` and `audit` commands exposed; no acquisition |
| `git diff --check` and staged `git diff --cached --check` | Passed |
| Opaque pre/post SHA256 preservation check | All 325 existing local output files unchanged; all preexisting tracked files unchanged except the four intended documentation/index updates |
| Literal YAML authorization and byte checks | Both proposal YAMLs unchanged and literal false; historical calibration/final-evaluation authorizations unchanged |
| Git ignore probes | NSTDB raw bytes ignored; compact manifest paths trackable |

Tests cover checksum/member/path/duplicate rejection, destination conflicts,
source-byte/provenance preservation, explicit-only transport, failed-audit retention,
missing sentinels, invalid headers, independent digital decoding, role/filter-support
isolation, pool identities, exact RNG order, SNR/amplitude, flat/null-SNR accounting,
zero/nonfinite noise, invalid ECG/length, overflow, saved preprocessing compatibility
and rejection of holdout planning/application. These tests verify primitives only;
no Phase A/B case-accounting or experiment lifecycle implementation is claimed.

## Preservation and limitations

The Challenge decoder, original source, locks, configurations, splits, numerical
protocol choices, all historical artifacts and authorizations are unchanged.
Only noise primitives and supporting documentation/tests were added. Both new
proposal `execution_authorized` flags remain false; neither YAML has executable
schema support. The protocol receives only an implementation-status cross-link.
No real data acquired/imported, no signals decoded outside generated fixtures,
and no training, inference, calibration, stress evaluation or quality experiments
were performed. No raw data, arrays, weights, environments, predictions or bulk
outputs belong to the commit.

No numerical protocol deviations. Before implementation, the
[adapter documentation](../docs/nstdb-adapter.md) recorded the checksum-index
trust clarification: local imports trust an owner-supplied publisher index;
HTTPS downloads trust the versioned PhysioNet endpoint; the index's own recorded
hash is not publisher-authenticated by itself. Real endpoint payloads/checksums
and headers still require the actual acquisition/audit; synthetic checks cannot
establish real release readiness or availability.

Real NSTDB acquisition/audit and source/role manifest freezing are pending. Durable
external backup destination and restoration checks remain unspecified/unverified;
Git documentation and hashes are not backups of ignored bytes. No archive/upload
was attempted. Temporal roles share recordings and unverified volunteer identities;
they do not establish subject/source independence. Noise types are not rhythm
labels or BUT quality classes. Project global-RMS SNR is not the NST tool's SNR.
The original Challenge holdout is consumed and excluded from future development.

Next: explicitly acquire/import the approved subset, audit into a new output,
review/preserve source and role identities, then implement and synthetically verify
the Phase A runner before any separately authorized validation-only inference.
Phase B and separate BUT quality assessment remain later milestones.
