# NSTDB acquisition, audit and noise adapter

This milestone implements standalone acquisition/audit commands and a library
adapter for the frozen [noise proposal](experiments/noise-robustness-v1.md).
It does **not** implement the Phase A/B experiment runner. Both proposal YAMLs
remain declarative and unauthorized; the Challenge decoder is unchanged.
The [original acquisition attempt](../reports/nstdb-real-data-readiness-v1.md) failed
publisher-index parsing before publication; the
[compatibility repair](../reports/nstdb-publisher-index-repair-v1.md) preserves that
failure. The subsequent [real-data readiness record](../reports/nstdb-real-data-readiness-v2.md)
documents a timed-out network attempt followed by an explicitly authorized local
import, passed strict audit and review of all nine noise-role descriptors.
Native voltage calibration and durable archival remain unverified. No new model
experimental results exist.

Implementation clarification recorded before implementation: publisher SHA256
checksums cover required release files except the checksum index itself, which
cannot contain its own digest. The index's digest is recorded as provenance,
not represented as a publisher-authenticated self-check. Explicit download trusts
the versioned PhysioNet HTTPS endpoint. Local import trusts the owner's supplied
publisher index; consistency checks alone do not authenticate that index. This
is a provenance limitation, not a new execution approval mechanism. No numerical
or experimental proposal rules change.

Successful acquisitions preserve original bytes. Existing destination directories
(even incomplete ones) block acquisition/audit publication; failures are never
automatically retried or overwritten. Interrupted writes must be investigated and retained.
The official RECORDS/index may list excluded 118/119 examples; required membership
is checked without acquiring those examples. Only the nine selected files are
copied/downloaded. No other data or model files are read by these commands.

## Publisher-index grammar and acquisition ordering

Every entry, including unrelated metadata, must match this ASCII grammar:

```text
entry     = HEX64 separator name
separator = SP | SP SP | SP "*"
name      = ["./"] component ("/" component)*
component = one or more characters from A-Z a-z 0-9 _ . -
```

`HEX64` is exactly 64 hexadecimal characters, case-insensitive. LF and CRLF line
endings and one optional final newline are supported. Empty entries, comments,
tabs, extra spaces, escaped/percent-encoded names and non-ASCII names are not.
Components ending in a dot (including `.` and `..`) are rejected. Absolute paths,
backslashes, colons, repeated or trailing slashes and additional `./` segments are
rejected. One leading `./`
is removed for lookup; duplicate normalized names, including ASCII case aliases,
are rejected regardless of whether their digests agree. Required filenames still
use exact case. The grammar deliberately does not support arbitrary filename or
filesystem normalization conventions.

Safe nested entries such as `old/oldma.dat` are validated index metadata only.
They never select a URL, local source path or destination. Both download and
import enumerate the fixed nine-file `FILES` tuple, not the index or RECORDS.
RECORDS must contain all bw/ma/em names without duplicate lines; excluded examples
may be listed without being acquired.

Acquisition first reads only RECORDS, SHA256SUMS.txt and nstdb.txt. Before any
signal/header read, it validates every index entry, coverage of all eight required
non-index files, required RECORDS membership and the actual hashes of RECORDS and
nstdb.txt. It then reads the six fixed signal/header files and repeats complete
validation of all nine bytestrings before reserving or writing successful outputs.
Index consistency does not independently authenticate the index; the trust
limitations described above remain unchanged.

No caching or retry subsystem is implemented. Bytes remain in memory until full
validation succeeds. A transport or validation failure before publication creates
no acquisition destination and discards buffered bytes when the process exits;
operators must preserve console logs and any separately authorized diagnostic
metadata. This ordering avoids expensive signal transfers for invalid metadata,
but does not preserve failed transfers or provide resume. Publication failures
after exclusive destination creation may leave partial original files/directories;
they are retained and block subsequent acquisition, never overwritten or retried.

## Supported commands

Run from the repository with the existing CPU environment. These commands require
explicit acquisition authorization and absent destinations. The local-import and
audit commands were used in the linked readiness record; its occupied destinations
must be preserved. Choose one source, not both:

```sh
# Import the owner's nine original release files; never contacts the network.
.venv/bin/python -m rhythmlens.nstdb acquire --local /path/to/nstdb-1.0.0
# Alternatively, explicitly download only the nine versioned release files.
.venv/bin/python -m rhythmlens.nstdb acquire --download
# Verify acquisition provenance/checksums and audit digital noise, without ECG/model I/O.
.venv/bin/python -m rhythmlens.nstdb audit \
  --output data/manifests/nstdb-v1/audit-v1
```

`acquire` accepts `--raw` (default `data/raw/nstdb/1.0.0`) and `--manifest`
(default `data/manifests/nstdb-v1`). `audit` accepts `--raw`, `--provenance`
(default `<default manifest>/provenance.json`) and requires a new `--output`
directory. Overriding the manifest does not implicitly change the audit provenance
argument. Raw/noise bytes must stay in ignored raw storage; use trackable manifest
locations only for compact JSON. No acquisition happens on module import,
`--help`, audit, adapter calls, or tests. No YAML execution parser was added;
these commands do not interpret either proposal's authorization or restrictions
as runnable configuration. There is no Phase A/B command in this milestone.

## Outputs and failure meanings

| File/schema | Definition |
| --- | --- |
| `provenance.json` / `rhythmlens-nstdb-acquisition-v1` | Version/DOI, attribution/license, explicit source method, per-file official URL, UTC acquisition time, byte size and SHA256, publisher-checksum status; checksum-index trust limitation; real audit/archive initially pending |
| `audit.json` / `rhythmlens-nstdb-audit-v1` | Overall passed/failed, input hashes, canonical sorted compact-JSON provenance SHA256, per-record format/channel/header/rate/length/initial/checksum checks, per-channel missing counts, errors, role descriptors and NumPy version |
| `roles.json` / `rhythmlens-nstdb-roles-v1` | Written only after a passed audit: audit-file SHA256 and nine role descriptors with raw half-open bounds, two channels, source hash, resampling/trim/SciPy identity, pool length and pool SHA256 |
| `sha256.json` | Byte-level SHA256 index of published audit and (if passed) roles JSON; no self-hash claim |

Publisher checksum/membership/provenance failures stop before signal decoding.
Decoded missing sentinel `-2048`, invalid header/length, initial value or 16-bit
signed checksum mismatch blocks readiness. Structural record failures remain in
the three-record inventory; overall failed audits never publish a usable role
manifest. Failure reports may retain successfully examined records/roles as partial
evidence. Failure exits nonzero. Exceptions before auditing (for example missing
provenance or output collision) do not create a new output. An interrupted write
may leave partial files, which remain untouched. Successful acquisition alone is
not a passed signal audit.

Roles are sliced at raw samples `[0,216000)`, `[237600,410400)`,
`[432000,648000)` before independent resampling. After 300-sample trimming at
each end the respective lengths are 179400, 143400, 179400 per channel.
Pool digests use little-endian float64 C-order samples-by-channels bytes.
Source gain remains unknown: digital shape values are never calibrated mV.
Temporal separation provides neither subject nor source independence.

## Adapter API and boundaries

`rhythmlens.nstdb.load_pools(raw, provenance, role)` revalidates release bytes,
provenance, all three headers and digital records, then returns immutable pool
objects. `rhythmlens.noise.stress_plan` requires `partition="validation"`;
`training_plan` requires `partition="train"`. Both implement the proposal's
compact UTF-8 JSON/SHA256/little-endian PCG64 keys and exact draw orders. They
contain no model, arm or severity draw in a stress plan. `apply_plan` verifies the
exact plan and pool identity, selects one contiguous slice without wrapping,
and applies noise to a caller-supplied **physical mV, 300 Hz** ECG array. It cannot
load ECG files or establish that a caller's array units/partition are truthful.
The future runner must verify source/split/unit identities before supplying arrays.

`corrupt_physical` is the pure numerical primitive: mean-center the selected
unitless shape, normalize its RMS, scale using median-centered ECG RMS and the
requested project SNR, and add in physical mV. Metadata report requested/achieved
SNR, reference/added RMS, added peak and flat accounting. `apply_plan` also records
plan/role identity and original/corrupted physical digests. The caller then uses
unchanged `transform_physical(corrupted, saved_state)`; no preprocessing is fitted
or automatically loaded here. Test fixtures supply an explicitly synthetic state.
Post-transform digests and complete case ledgers remain future runner work.

Reference RMS below `1e-8` retains the unchanged case, adds zero, and reports null
achieved SNR with `flat_or_near_flat_reference`; it is not a deferral. Zero or
nonfinite selected-noise RMS, invalid lengths, nonfinite ECG, overflow, altered
plans/pools or incompatible SciPy raise technical failures without redrawing or
substituting. The future runner must retain eligible denominators and record these
failures; this library does not implement an experiment ledger or metrics engine.
Global-RMS SNR differs from the official NST QRS/weighted-noise definition.
Noise labels are corruption types, never rhythm or BUT quality labels.

## Verification and next step

Generated synthetic fixtures exercise packed decoding against independent WFDB
digital decoding, byte/checksum/member/collision handling, failed-audit retention,
role/filter isolation, exact RNG draws and NumPy/Torch global RNG isolation,
SNR, flat/invalid cases, length preservation and unchanged saved preprocessing.
No clinical ECG, real NSTDB, checkpoints or predictions are test inputs.

The original failed acquisition is preserved in the
[failure report](../reports/nstdb-real-data-readiness-v1.md), with its findings
unchanged. The [dated repair record](../reports/nstdb-publisher-index-repair-v1.md)
distinguishes local metadata parsing and synthetic tests from real readiness.
The [local-import readiness record](../reports/nstdb-real-data-readiness-v2.md)
records the passed real digital audit, supplemental finite/nonflat and independent
WFDB checks, and review of all nine source/role descriptors. The acquisition
provenance's `acquired_utc` fields are local import times, not observed browser
download times; its original `audit: pending` field is acquisition-time state.
The separate audit records the later outcome. Reviewed descriptors are not frozen
executable experiment inputs, and both proposal YAMLs remain unchanged.
The owner must specify a durable backup
destination and restoration checks: ignored local noise and historical run outputs
are not backed up by Git
reports or hash indexes. This gap does not prevent synthetic implementation.
The [Phase A runner](noise-phase-a.md) now implements pairing, baseline reproduction,
complete case accounting/lifecycle and sealed scoring with generated synthetic
verification. Its separate execution contract remains disabled; real validation-only
inference still requires separate authorization. Phase B training, BUT quality
assessment, deferral and reviewer interface remain later work. The consumed
Challenge holdout is excluded;
no new independently sealed final cohort is established here.

Primary format references: [WFDB format 212](https://physionet.org/physiotools/wag/signal-5.htm),
[WFDB header semantics](https://physionet.org/physiotools/wag/header-5.htm),
[WFDB missing-value definitions](https://github.com/MIT-LCP/wfdb-python/blob/v4.3.0/wfdb/io/_signal.py),
and [NSTDB release](https://www.physionet.org/content/nstdb/1.0.0/).
