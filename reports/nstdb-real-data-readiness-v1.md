# NSTDB 1.0.0 acquisition v1 — blocked real-data readiness

Date: 2026-10-19. **Acquisition validation failed; real-data readiness is not
established.** No strict signal audit, role generation or model experiment ran.
This report is deliberately uncommitted failure evidence, not a success milestone.

## Verified starting state and scope

Clean branch `feat/nstdb-adapter`, HEAD and fetched upstream
`74b518872ce74840259e3f88820266df64c1b058`. Live PR #8 was open/draft against
`feat/noise-robustness`; PR #7 was open/draft against main. No branch switch,
pull, rebase, reset, merge or dependency installation occurred.

No original local release files were supplied. The existing adapter requested
only RECORDS, SHA256SUMS.txt, nstdb.txt, and bw/ma/em .dat/.hea from
`https://physionet.org/files/nstdb/1.0.0/`. The selected release is ODC-By-1.0;
see the [recorded source/license research](dataset-readiness-noise-quality-v1.md).
No corrupted ECG examples or BUT QDB files were downloaded.

## Actual outcome

The initial sandbox request failed DNS before publication. A network-enabled
attempt was interrupted before publication. On the owner's explicit request to
resume, no active acquisition or occupied raw/manifest directory existed.

The resumed operation used the unchanged adapter with a temporary URL progress
wrapper: it logged request start/completion without changing transport arguments,
checksum validation, decoder or numerical rules. All nine responses completed
between 2026-10-19T06:14:54.817732+00:00 and
2026-10-19T06:36:06.709251+00:00. These are UTC evidence timestamps.

Acquisition then exited 1 with:

```text
NSTDB blocked: Malformed publisher checksum index
```

`acquire` validates the complete in-memory bundle before writing destinations.
Consequently neither `data/raw/nstdb/1.0.0/` nor
`data/manifests/nstdb-v1/` was created. The completed signal transfers were **not
retained on disk** when validation failed; they cannot be audited or assigned
verified file identities from this attempt. No per-file acquisition provenance,
audit JSON, source-role manifest, pool hashes or readiness-review JSON exists.
Do not infer checksum success, sample validity or readiness from transfer completion.

The affected acquisition stopped. It was not retried after the validation failure.
No source bytes, checksum index or parser were patched to force acceptance.

## Metadata-only diagnosis

After the failure, only the small publisher checksum index was retrieved once
into a separate ignored diagnostic directory, not as an acquisition retry. Its
source URL, observation timestamp, size, digest and parser findings are retained
in `outputs/nstdb-readiness-v1/checksum-index-diagnostic/`.

| Diagnostic observation | Result |
| --- | --- |
| Index bytes | 7439 |
| Index SHA256 | `b76bd98c5111439fcfff2f410afd70d64e79f072049c45b5a9916a3044fdb84f` |
| Entries | 97 |
| Entries with one ASCII space between hash and filename | 97 |
| Entries under `old/` | 31 |
| Duplicate names | 0 |
| All eight selected non-index filenames represented | Yes |
| Selected signal hashes actually verified | No |
| Diagnostic summary SHA256 | `815badd1857087d5f585df53347dc5fa2343386422948afff1d0c89279b016cc` |

This digest identifies the later diagnostic copy, not the unpreserved index
buffer from the failed acquisition. The source endpoint is the same; identity
of those two copies cannot be independently proved from retained bytes.

The parser in `src/rhythmlens/nstdb.py:49` requires a space followed by another
space or `*`; a single-space line is rejected. Independently, line 53 rejects
every slash-containing index name, including the publisher's `old/...` entries.
These are implementation compatibility blockers, not evidence of a signal-file
checksum mismatch or a corrupt release. The first error prevents reaching later
checks; the nested-path incompatibility is established by source inspection.

Synthetic index construction in `tests/test_nstdb.py:35` uses two spaces and does
not reproduce this real index layout. The passing synthetic suite therefore did
not establish real publisher-index compatibility. A different mirror serving
the same index bytes would encounter the same parser rules.

## Evidence retained and checks actually run

- `outputs/nstdb-readiness-v1/acquire.log`: original sandbox DNS failure.
- `outputs/nstdb-readiness-v1/acquire-network.log`: interrupted attempt's log.
- `outputs/nstdb-readiness-v1/acquire-resumed.log`: nine URL completions and the
  validation error; no signal samples are logged.
- `outputs/nstdb-readiness-v1/checksum-index-diagnostic/SHA256SUMS.txt`: unchanged
  diagnostic publisher metadata.
- `outputs/nstdb-readiness-v1/checksum-index-diagnostic/diagnostic.json` and
  `summary.json`: diagnostic provenance and compact findings.

New verification during this task, before the real acquisition outcome:

| Check | Actual result |
| --- | --- |
| Existing NSTDB/noise synthetic tests | 47 passed in 3.13 s |
| Ruff lint / formatting | Passed; 33 files already formatted |
| Repository ignore/trackability and credential-free lock contracts | Passed |
| Frozen snapshot verification | 17 unchanged identities, counts 5969/1280/1279, zero ECG files opened |
| Both offline dependency lock checks | Passed |
| Installed CPU dependency compatibility | Passed; 58 distributions |
| Preexisting tracked-file preservation snapshot | All 109 files unchanged before writing this new report |

Initial uv checks encountered a read-only default cache; rerunning those checks
with a writable temporary cache passed without installation or upgrade. No new
full regression suite, packaging build or model inference was run. The earlier
320-test CI result remains historical evidence. Required frozen verification read
split metadata only; no Challenge signals or per-case predictions were opened.

Both proposal YAMLs, including literal false execution authorization and their
unsupported status, remain byte-identical. Numerical source, dependencies,
decoder contracts, frozen protocols, historical results, manifests and splits
are unchanged. No checkpoints were loaded. No calibration, training, stress/final
evaluation, quality windows/splits or consumed-holdout signals were accessed.

## Readiness, preservation and next dependency

**Not ready for Phase A runner execution or a real-data-ready claim.** Even
successful acquisition would still require strict decoding and a nine-role
review; neither occurred here. Native noise voltage calibration remains
unverified, and digital shapes must not be described as calibrated mV. No imposed
corruption amplitude or model result exists.

No approved durable archive destination/retention/restore workflow is documented.
Archival remains pending; no upload occurred. Logs and metadata do not preserve
the signal bytes discarded by this acquisition's validate-before-publication path.

The immediate dependency is a separately reviewed publisher-index compatibility
fix, not Phase A. Use the retained index metadata to specify supported delimiter
and safe relative-path semantics while preserving strict SHA256, duplicate,
membership, traversal and required-file checks. Add realistic synthetic coverage;
do not download excluded `old/` files. Review checking metadata before expensive
signal transfers and retaining interrupted/failed acquisition evidence. Preserve
the current failure and obtain explicit authorization before another real
acquisition attempt. Do not alter the format-212 decoder, frozen noise protocol,
proposal authorizations or historical experiments for that fix.

No commit, push or PR update publishes this attempt as successful. The report is
left uncommitted for review; all diagnostic artifacts stay ignored. PR #8's
existing synthetic-only readiness claims are not upgraded to real-data success.
