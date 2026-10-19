# NSTDB publisher-index compatibility repair v1

Date: 2026-10-19. Starting revision:
`74b518872ce74840259e3f88820266df64c1b058`, branch `feat/nstdb-adapter`, matching
its fetched upstream. PR #8 remains draft, stacked on `feat/noise-robustness`;
PR #7 remains open/draft and unmerged. This is a code/synthetic verification
milestone, **not real NSTDB acquisition, digital auditing or experiment execution**.

## Preserved failure and primary evidence

The [original failure report](nstdb-real-data-readiness-v1.md) is included unchanged
as the historical acquisition record, SHA256
`2de92d4587217d0a69c3321e75f2c66b974e9567ba5614c2b2c9c4b1ad648660`.
Its statements about being uncommitted and pending repair describe that earlier
task; this record supplies the subsequent repair state without rewriting findings.
The original signal transfers were discarded from memory on failed validation;
no signal-file identities, passing audit or role manifests resulted.

The retained diagnostic index was inspected locally and verified as 7439 bytes,
SHA256 `b76bd98c5111439fcfff2f410afd70d64e79f072049c45b5a9916a3044fdb84f`.
All 97 entries use a single space; 31 are under `old/`. This copy was retrieved
after the failed acquisition, not retained from its in-memory buffer. Raw index,
diagnostic JSON, logs and all prior evidence under ignored
`outputs/nstdb-readiness-v1/` remain byte-identical. No dataset bytes were fetched
over the network in this repair task. Git/GitHub network operations are separate.

## Root cause and exact repair

The old parser required two spaces or space-plus-asterisk and rejected every
slash-containing name. Synthetic index fixtures used two spaces and omitted the
release's unrelated nested metadata. Therefore they did not expose either
compatibility issue.

`src/rhythmlens/nstdb.py` now separates publisher-index parsing from required
metadata validation and complete bundle validation. The
[adapter documentation](../docs/nstdb-adapter.md#publisher-index-grammar-and-acquisition-ordering)
defines the exact ASCII grammar: 64 hexadecimal digits, one space / two spaces /
space-plus-asterisk, and slash-separated components drawn from letters, digits,
underscore, dot and hyphen. One leading `./` is allowed. LF/CRLF and one optional
terminal newline are supported; other whitespace, comments, escaping and non-ASCII
names are rejected. Components ending in a dot, including `.`/`..`, are rejected.

Every entry is validated, including unrelated ones. Absolute paths, traversal,
backslashes, colons, repeated/trailing slashes and ambiguous syntax fail closed.
Duplicate names after removal of leading `./`, plus ASCII case aliases, fail even
if digests agree. Required filenames retain exact case. Safe `old/...` entries
are metadata only; names never select network URLs, import paths or output paths.

Both acquisition paths retain the same fixed nine-file allowlist and required
bw/ma/em RECORDS membership. After reading the three metadata files, acquisition
validates the entire index, all eight required checksum entries, RECORDS membership
and metadata byte hashes **before any signal/header read**. All nine file hashes
are still checked before exclusive publication; a valid name alone cannot replace
a matching SHA256. Index consistency is not independent authentication of its
contents; versioned HTTPS versus owner-supplied-index trust remains explicit.

No decoder, audit, pool loader or noise numerical change was made. AST comparison
verified all preexisting NSTDB functions other than `validate_files` and `acquire`
unchanged; only the two new parsing/metadata helpers were added. Other numerical
source, locks, authorizations, frozen protocols, manifests/splits and historical
experimental results remain unchanged.

## Verification actually run

Existing environment only: Python 3.11.16, NumPy 2.2.6, SciPy 1.15.3, WFDB 4.3.1,
Torch 2.8.0+cpu, PyYAML 6.0.3, pytest 8.4.2, Ruff 0.11.13 and build 1.6.1.
No installation or dependency upgrade occurred.

| Check | Observed result |
| --- | --- |
| `.venv/bin/python -m pytest -q tests/test_nstdb.py tests/test_noise.py` | **134 passed in 3.56 s** |
| `.venv/bin/python -m pytest -q tests` | **407 passed in 21.24 s** |
| Ruff lint and format check for source/tests/scripts/CUDA verifier | Passed; 33 files already formatted |
| Repository contracts | Passed: ignored raw/output bytes; trackable evidence; credential-free portable locks |
| Frozen verification | Passed: 17 unchanged identities, counts 5969/1280/1279, zero ECG files opened |
| Both `uv lock --check --offline` profiles | Passed; writable temporary uv cache, no sync/install |
| `uv pip check --python .venv/bin/python` | Passed; 58 distributions compatible |
| Build with `--no-isolation` into a temporary output directory | sdist and wheel built successfully |
| Built wheel extracted into a temporary directory, using existing interpreter/dependencies | Imports, packaged decoder contract, index parsing, CLI help/version passed; no installation |
| Retained local diagnostic index through repaired parser | 97 entries parsed, 31 `old/` entries, all eight required names present; **metadata only** |
| Preservation snapshot | Only intended code/current documentation changed; failure report, diagnostics and all other preexisting files unchanged |

The focused suite adds 87 cases beyond the original 47 NSTDB/noise cases.
Transport tests are mocked and an autouse guard rejects unmocked NSTDB network
access. They cover all supported separators and line endings; realistic generated
indexes with safe nested entries; exact nine-file download/import without unrelated
reads; malformed unrelated entries; aliases/unsafe paths; each required checksum's
absence/mismatch; metadata failure before signal I/O in both acquisition paths;
payload mismatch after metadata success; destination collisions and publication
failure retention. Existing decoder, noise, audit and output safeguards remain
covered. Fixtures are generated in temporary directories, not copied from real
signals or the diagnostic index.

During test development one supposed malformed entry accidentally formed valid
CRLF when its final CR was followed by the fixture's newline. The fixture was
corrected to place CR inside the filename; the supported CRLF grammar was retained.
The final results above are after this correction and final path/alias cases.
The earlier 320-test CI result is historical; the 407-test result is newly observed
local verification. Full-suite model operations use synthetic temporary fixtures;
no real training, inference, calibration, stress or final evaluation ran.

## Failure-byte behavior and remaining limits

This repair changes ordering, not failure storage. Prepublication transport or
validation failure leaves no acquisition destination; buffered bytes are discarded
on exit. Metadata-first validation now avoids signal transfer for invalid metadata,
but does not save metadata or payload failures automatically. Preserve console
logs and separately authorized diagnostics. Publication failures may leave partial
original files/directories, which are preserved and block reuse. There is no
automatic retry, resume, cache or broad acquisition subsystem.

The grammar is deliberately restricted, not a general checksum utility supporting
arbitrary filenames or platform normalization. Unrelated files' names/digests are
validated syntactically but their bytes are never acquired or verified. The index
is hashed as provenance, not self-authenticated. Local parsing of the retained
index does not validate release payloads, future endpoint availability or decoder
compatibility with real samples.

**Real NSTDB readiness remains pending until a separately authorized retry and
passing strict audit**, followed by review of all nine generated role descriptors.
Those descriptors must remain distinct from frozen executable experiment inputs.
The native noise voltage calibration remains unverified; digital shapes are not
calibrated mV. Both unsupported proposal YAMLs retain literal false authorization.
No Challenge signals/per-case predictions, real checkpoints, BUT QDB or consumed
holdout signals were opened. Required frozen verification read split metadata only.
Durable archival remains pending; no approved destination or upload was introduced.

The next task is a separately authorized bounded acquisition/audit retry, preserving
these failed-attempt diagnostics and using absent destinations. Phase A runner
implementation follows only after real-data readiness is established; experiment
execution remains unauthorized.
