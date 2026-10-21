# NSTDB 1.0.0 local-import readiness v2

Date: 2026-10-21 (Asia/Kolkata). **Local import and the single strict audit passed.**
The three original noise records and nine generated temporal role descriptors are
reviewed. This establishes bounded noise-data readiness, with the limitations below;
Phase A/B implementation and all real model experiments remain pending and unauthorized.

## Scope, Git state and preserved failures

Starting branch `feat/nstdb-adapter`, HEAD and fetched upstream
`4c8f1576d635b0afbb0da0f58130077ce68c1b2b`, clean working tree. PR #8 was open/draft
against `feat/noise-robustness`; PR #7 remained open and unmerged. Process inspection
found no running acquisition/audit. Raw, manifest and audit destinations were absent.
No concurrent acquisition, branch switch, reset, rebase or dependency installation occurred.

The [original failure](nstdb-real-data-readiness-v1.md) and
[publisher-index repair](nstdb-publisher-index-repair-v1.md) remain byte-identical.
Their earlier pending statements describe those milestones. A separately authorized
network retry on 2026-10-21 completed RECORDS, SHA256SUMS.txt and nstdb.txt, passed
metadata validation, then timed out on bw.dat. Its log's last write was
2026-10-21T03:12:42.105754Z (08:42:42 IST); it exited 1. No raw/manifest destination
was created. The three metadata buffers and any partial bw.dat response were
**discarded from memory**, not retained as original files. No automatic retry followed.

Retained timeout evidence: `outputs/nstdb-readiness-v2/acquire.log`, 705 bytes,
SHA256 `437bff84dff506f871013fd4cfc9cda5b55343a50879ed253b3ae368a2423cc9`,
and `failure-summary.json` in that directory. The owner's subsequent manual download
and explicit local-import authorization started this separate attempt. Prior ignored
logs, diagnostic index, staging copies and transfer receipt remain unchanged.
A before-snapshot covers 132 existing files; final preservation verification permits
only the three current documentation/index updates listed below.

## Actual source, import and provenance

Owner-supplied Windows folder:
`C:\Users\satvi\Downloads\mit-bih-noise-stress-test-database-1.0.0`.
The required files are inside its nested `mit-bih-noise-stress-test-database-1.0.0`
subdirectory. The actual imported source is recorded verbatim in provenance:

```text
/mnt/c/Users/satvi/Downloads/mit-bih-noise-stress-test-database-1.0.0/mit-bih-noise-stress-test-database-1.0.0
```

Only the fixed nine files below were opened/imported. Additional release members,
including 118/119 ECG examples and old/ entries, were left alone. Earlier copies
in `data/raw/nstdb-manual/1.0.0/` remain separate staging evidence; this import read
the supplied Windows originals directly. Source originals remain unchanged.

Canonical release URLs are `https://physionet.org/files/nstdb/1.0.0/` plus each exact
filename; these identify the release, not an observed browser request history.
Version 1.0.0, DOI `10.13026/C2HS3T`, license ODC-By-1.0 and source attribution/license
URL are recorded in provenance. The owner's browser download time, request method
and archive extraction history were not observed. Filesystem modification times do
not establish those events. The provenance field `acquired_utc` records **local
import reads**, 2026-10-21T03:45:23.251156Z through 03:45:23.347872Z (09:15:23 IST),
not the manual download time. No network dataset requests occurred in this attempt.

```sh
.venv/bin/python -u -B -m rhythmlens.nstdb acquire --local \
  /mnt/c/Users/satvi/Downloads/mit-bih-noise-stress-test-database-1.0.0/mit-bih-noise-stress-test-database-1.0.0
.venv/bin/python -u -B -m rhythmlens.nstdb audit \
  --output data/manifests/nstdb-v1/audit-v1
```

Each command ran once and exited 0. Logs are respectively
`outputs/nstdb-readiness-local-v1/import.log` and `audit.log`; both are empty because
the supported CLI is silent on success. Original bytes are preserved under ignored
`data/raw/nstdb/1.0.0/`, exactly nine files totaling **5,859,729 bytes**.

| Original file | Bytes | SHA256 |
| --- | --- | --- |
| `RECORDS` | 93 | `006d76f378c140ebbb1c20865a79ccb58e2f797b8ea7f1aa88968f1e319af5a4` |
| `SHA256SUMS.txt` | 7439 | `b76bd98c5111439fcfff2f410afd70d64e79f072049c45b5a9916a3044fdb84f` |
| `nstdb.txt` | 1922 | `7d31e1797fa7c67e77a4cf38a3321d0594169680fa8ad5f4c9124ad7b265fdb0` |
| `bw.dat` | 1950000 | `15f68280aba8b006f5497c14da44b558e68caaabc2afc5c4b710eea779b7512c` |
| `bw.hea` | 93 | `e3ea3b09fb8d65877cf6ae4dec213b9693243c435333a150c14f6b551f7b7d61` |
| `ma.dat` | 1950000 | `40bda613a3e65adb277df1b72f8a41d2c1c2e0131ea4ec1e03cf1e2cd68199c4` |
| `ma.hea` | 90 | `c3751bb8d327b83158f3e4a81f821518a71d1ae8566189b098baa47fb89afaa7` |
| `em.dat` | 1950000 | `4f9416a04b2bed98081ee3118c1282f9e35321bfd5da96ce48056a3fe147358d` |
| `em.hea` | 92 | `0bdfa989de0dd06486159f2c36bd80abd684d951c3930daea2ee46572fd20d49` |

All 97 index entries, including 31 safe old/ entries, were syntax-validated.
All eight required non-index files matched publisher SHA256 entries; RECORDS
contains bw/ma/em. Metadata validation precedes any signal/header read. Complete
validation precedes destination publication. The index digest matches the retained
October 19 diagnostic copy. This is publisher-index **consistency**, not independent
index authentication; local import trusts the owner's supplied index. The index
cannot self-authenticate and has `publisher_checksum_verified: false`.

## Evidence identities

All following digests hash exact on-disk file bytes:

| Evidence path | Bytes | SHA256 |
| --- | --- | --- |
| [provenance.json](../data/manifests/nstdb-v1/provenance.json) | 3484 | `964dff5fc3154eee3d1c4787edddb85ae8070b2b7f1bf882acb935db734ae1e2` |
| [audit.json](../data/manifests/nstdb-v1/audit-v1/audit.json) | 11746 | `ec04b4a146fbcfffe7223ac4722ff985dcfefed0cc9302a08c23b3eee62d9b39` |
| [roles.json](../data/manifests/nstdb-v1/audit-v1/roles.json) | 8145 | `cf6e3cb30e88811129c3f5ac82d0a78e9c92a5c7867ec7d9d91d66ebbbb3b68c` |
| [sha256.json](../data/manifests/nstdb-v1/audit-v1/sha256.json) | 171 | `7e622acc9b92dc3c4070b9cb8af2e813746c56c3943e2fe2e34431af79116002` |
| [readiness-review.json](../data/manifests/nstdb-v1/audit-v1/readiness-review.json) | 5770 | `7d7481a145700e5a041dc26ad97da25def88ead2c56d51b35602707f5a1f736f` |

The audit's `provenance_sha256` is instead the digest of sorted compact canonical
JSON: `6a87f70e444fc3e4ef26afbfadd9912dc7139cb1aba5c028d45e248a1aac173e`. It intentionally differs from the byte digest.
`roles.json` binds the audit's byte digest; `sha256.json` indexes audit/roles bytes.
The supplemental review binds these files, frozen proposal and verification script.
The original provenance retains `audit: pending` as its immutable acquisition-time
state; the separate passed audit and this report supply the later result.

## Three records, both channels

Strict audit timestamp: 2026-10-21T03:45:28.751195Z (09:15:28 IST). All three records
passed with no errors. Each .dat is 1,950,000 bytes, representing 650,000 samples
per channel at 360 Hz, two channels named noise1/noise2, packed WFDB format 212.
Headers have ADC resolution 12, ADC zero 0, gain 0 and block size 0. Strict parsing
checked the expected header inventory/rate/length/fields. Decoding checked exact
payload length, both initial values, signed 16-bit sample-sum checksums and missing
sentinel -2048 counts. Header and computed checksum/initial values agree:

| Record | Initial values, channels 0/1 | Signed checksums, channels 0/1 | Missing counts, channels 0/1 | Status |
| --- | --- | --- | --- | --- |
| bw | -29, 25 | -3562, -20847 | 0, 0 | passed |
| ma | -18, 3 | 10269, 7630 | 0, 0 | passed |
| em | 5, -21 | -5959, -26067 | 0, 0 | passed |

The existing audit checks finite resampled pool values; digital decoding yields
finite integers. It does **not** report full-source/channel flatness, independently
cross-check WFDB, or provide a separate full-source nonfinite finding. Those were
checked supplementally, without rerunning the strict audit, by the retained one-off
`outputs/nstdb-readiness-v2/review-noise.py`, SHA256
`6e63e1ba4e2d6a78e13bd493e965d442967beaa05a6930fe6b07305d84bbc403`.
All 3 digital arrays exactly agreed with WFDB 4.3.1 physical=False decoding.
Both channels of every complete source and all nine pools were finite and nonflat
(peak-to-peak greater than zero). No samples/arrays were written into Git metadata.
These are automatic numerical findings, not human waveform or clinical review.
Whole-source/pool nonflatness does not establish nonzero centered RMS for every
possible future selected slice; the future runner must retain the existing per-case
RMS failure rule, without redraws.

## Nine role descriptors reviewed

The assistant reviewed metadata against the unchanged YAML/protocol. Supplemental
automation independently sliced the original digital arrays by YAML bounds and
called SciPy directly, without `prepare_pool`, reproducing all nine pool digests.
Both channels share each role's half-open source interval:

| Role | Seconds | Raw samples | Trimmed 300 Hz samples/channel |
| --- | --- | --- | --- |
| augmentation-train | [0,600) | [0,216000) | 179400 |
| development-reserve | [660,1140) | [237600,410400) | 143400 |
| stress | [1200,1800) | [432000,648000) | 179400 |

Slicing occurs before isolated `resample_poly`, SciPy 1.15.3, up=5/down=6,
window=('kaiser',5.0), constant pad cval=0; 300 output samples are trimmed from each
end. Filter input is confined to each role. The gaps and final 2,000 raw samples
remain unused. Pool hashes encode little-endian float64, C-order samples × channels.
No pool arrays were published; they can be reconstructed from retained originals.

| Record | Role | Samples/channel | Pool SHA256 |
| --- | --- | --- | --- |
| bw | augmentation-train | 179400 | `58d2c0aed22ef5cfce4928b60b29768b9c69400dcdb9c9a7884a3da6c5fc8e23` |
| bw | development-reserve | 143400 | `a039be56a71cd1f140d36f85fd21a11c302ccd7c9fc95fb78f871a6b80631aae` |
| bw | stress | 179400 | `73374a0df35e0b7dd83420176ad915df1cfbf02d8cee7aefa81102b92e3763bf` |
| ma | augmentation-train | 179400 | `13760d61e5f6413b40bafec0b5f2d88b38a74c2ff80400f376cdc7d73d67ca36` |
| ma | development-reserve | 143400 | `7ebbb5e9039773e8ddc921b9ae01729409e16cd3ef514d8952b4c78f2f45d9b3` |
| ma | stress | 179400 | `af92fc4b236466a0809b1989358234407cfda52065f7d65804e6c178652bc8ad` |
| em | augmentation-train | 179400 | `e27e0f9eef4d03ef5e28d8a87f45dd332a53f3393ab71dc390a6f97e9e53ee0a` |
| em | development-reserve | 143400 | `cf1e1788a0f743024276bc9f40ab2919a5eb92493d0f1bc3311db1220b2f6128` |
| em | stress | 179400 | `6b8fa13e35efa052d18e648d4c2dc7051a2352798c17b390d843e035aed1e407` |

Descriptors are **generated, reviewed evidence**, not frozen executable experiment
inputs. No case plans, candidate runtime identities or resolved experiment config
were created. The proposal's null audit/role/case-plan placeholders remain unchanged;
a future implementation must bind this reviewed evidence in a separate versioned
execution input/registry without rewriting the frozen proposal.

## Verification actually run in this local-import task

Existing environment: Python 3.11.16, NumPy 2.2.6, SciPy 1.15.3, WFDB 4.3.1 and
PyYAML 6.0.3. No dependencies were installed or upgraded.

| Check | Actual result |
| --- | --- |
| Local source preflight, repaired adapter import | Nine files, metadata coverage/membership and required publisher hashes passed |
| Single strict audit | Passed, three records/six channels/nine roles, zero errors |
| Supplemental noise-only review | WFDB digital agreement, finite/nonflat channels/pools, independent protocol resampling and all nine hash reconstructions passed |
| Full repository synthetic/regression suite | **407 passed in 15.71 s** |
| Ruff lint and format | Passed; 33 files already formatted |
| Repository contracts | Passed, raw/output bytes ignored and compact evidence trackable |
| Frozen metadata verification | 17 unchanged identities, counts 5969/1280/1279, zero ECG files opened |
| Both offline lock checks | Passed, 60 and 74 packages resolved |
| Installed environment compatibility | Passed, 58 distributions |
| Existing-dependency build, --no-isolation | sdist and wheel built successfully |
| Extracted wheel using existing interpreter/dependencies, without installation | Imports, packaged decoder contract and CLI help/version passed |
| Preservation, trackability and documentation links | 132 preexisting files checked; only three intended current docs/indexes changed; Windows originals/import bytes match; both proposal flags false/unsupported; local links resolve |
| Diff whitespace check | Passed |

Full-suite model operations use generated temporary fixtures only. No real Challenge
signals/per-case predictions, real checkpoints, consumed-holdout signals, model
training/inference/calibration/stress evaluation or BUT files were accessed.
Frozen verification reads split metadata only. No source implementation, decoder,
noise numerical method, dependency lock, historical result, frozen protocol or
proposal YAML changed. Both proposals remain unsupported with literal
`execution_authorized: false` and `launcher_supported: false`.

## Preservation, limitations and next dependency

Tracked additions are only the five compact JSON files above and this report;
current docs/nstdb-adapter.md, docs/README.md and reports/README.md link the result.
Raw originals, Windows downloads, staging copies, diagnostic/progress scripts/logs,
preservation snapshots and build outputs remain ignored/local only. The Windows
copy and local copies are not a verified durable backup. No archive destination
was approved, no upload occurred and restoration remains unverified. Hashes and
Git reports cannot restore lost original bytes or historical experiment outputs.

**Native noise voltage calibration remains unverified**: gain=0 digital shapes are
not calibrated mV. Time-separated roles share source records; subject identities,
subject/source independence and long-range correlation are unresolved. No human
waveform-quality assessment or clinical adequacy is claimed. Checksum consistency
does not independently authenticate the index. No new final evaluation cohort exists.

The single next implementation dependency is a **synthetically verified Phase A
runner**, including reviewed evidence binding, pairing, lifecycle, accounting and
scoring. This task does not implement it or authorize execution. Durable archival
remains an owner action pending an approved destination, separate from implementation.
