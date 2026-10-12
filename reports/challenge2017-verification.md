# Challenge 2017 pipeline verification

This report separates synthetic implementation tests from actual dataset findings.
Run date: 2026-10-12 UTC. Python 3.11.16; uv 0.12.5; RhythmLens 0.1.0.
Dependency lock unchanged from environment setup:
`4ff3ef044d9e70d26b8a5c0494478e30cbc3f73ac3ecd94c109e1dd1adf44c8d`.

## Synthetic evidence

`.venv/bin/pytest -q`: **40 passed**. All test signals are generated in pytest
`tmp_path` directories, using NumPy `default_rng(2017 + record_number)` or
explicit small arrays. No real ECG samples enter tests or committed fixtures.
Split randomization uses Python `random.Random(42)` with sorted input keys.

Tests cover reversed/shuffled reference order; invalid/duplicate labels;
missing MAT/header pairs and orphan files; gain, baseline, ADC-zero fallback
and mV/uV/V conversion; preservation of source bytes; calibration-aware exact
duplicates; malformed sampling rate, channel count, sample length, gain, units,
file names, MAT payloads and offsets; nonfinite/missing samples; retained flat
and noisy records; duplicate leakage; conflicting labels; class coverage;
determinism under input reordering; frozen output hashes and overwrite refusal;
changed or newly added records after audit; cached acquisition with network
calls forbidden; unsafe ZIP traversal, links and duplicate targets; and explicit
format-16 initial-value/zero-checksum advisories versus blocking nonzero checksum
mismatches. These test results are software evidence, not dataset measurements
or model-performance results.

`.venv/bin/ruff check src tests`: passed.
`.venv/bin/ruff format --check src tests`: 7 files already formatted.
`UV_CACHE_DIR=/tmp/rhythmlens-uv-cache uv pip check --python .venv/bin/python`:
57 packages compatible. CLI help, subcommand help and version execution passed.
`git diff --check` passed.

## Actual acquisition and audit

Only `training2017.zip` (99,226,822 bytes) and `REFERENCE-v3.csv` (76,752 bytes)
were successfully acquired, using the official PhysioNet public S3 mirror listed
on the dataset page. Actual URLs, canonical URLs, UTC completion/observation
times, versions, license and SHA-256 values are in
`data/manifests/challenge2017-v3/provenance.json`.

Observed checksums:

- Archive: `25fdbab36bb6724b66a4d2c26ff89f6b9d6ff87e6aed714d24cf5bf071cd1255`.
- v3 reference: `ae517e474b0bc89a7517ad876b5b4dadc6c94f680d0665c04746db9432df855b`.

Actual commands:

```sh
.venv/bin/rhythmlens acquire
curl -I --max-time 30 https://physionet.org/files/challenge-2017/1.0.0/training2017.zip
curl -I --max-time 30 https://physionet-open.s3.amazonaws.com/challenge-2017/1.0.0/training2017.zip
.venv/bin/rhythmlens acquire --mirror s3
MPLCONFIGDIR=/tmp/rhythmlens-matplotlib .venv/bin/rhythmlens audit
.venv/bin/rhythmlens split
.venv/bin/rhythmlens acquire
```

The first command was attempted in the restricted sandbox, then retried with
network access. The final acquire ran successfully inside the network-restricted
sandbox: existing downloads and provenance were reused without network requests.
The matplotlib cache path above is disposable and unrelated to dataset storage.

Final actual audit: **passed**. All 8,528 label IDs have valid MAT/header pairs,
one channel, 300 Hz, calibrated mV values, correct declared sample lengths and
finite samples. No missing pairs, orphans, malformed records, nonfinite samples,
missing-value sentinels or flat recordings were found. Archive reference counts
were N=5,050, A=738, O=2,456, ~=284. v3 counts are N=5,076, A=758, O=2,415, ~=279;
148 labels changed. Every record remains in the manifest.

There are two exact duplicate pairs (A00227/A01176 and A02656/A03123), both N,
and no conflicting duplicate labels. An independent inline check compared the
physical arrays using `numpy.array_equal` for each pair before freeze: passed.
Twenty previews were created (five per class); their PNG signatures were checked.
Preview generation and data-integrity inspection happened before holdout assignment.
No additional waveform inspection was performed after freeze.

Header advisories: all 8,528 published checksums are zero; 565 initial-value
fields differ from the first MAT sample. These fields are recorded explicitly.
Zero checksums are unverified, and the initial value is unused for format 16
(as documented by WFDB). Every original file matches archive contents, and MAT
and WFDB readers agree on the digital payload. Nonzero checksum mismatches
remain blocking. See the audit report and policy for the exact validation rules.

## Actual frozen splits

| Split | Records | Proportion | N | A | O | ~ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 5969 | 0.69992964 | 3553 | 530 | 1691 | 195 |
| validation | 1280 | 0.15009381 | 762 | 114 | 362 | 42 |
| holdout | 1279 | 0.14997655 | 761 | 114 | 362 | 42 |

Freeze time: `2026-10-12T06:08:30.952705+00:00`. Assignment SHA-256:
`fb27f37a8f9dcd1d8931116e252290a68be29844b5b81badc3223954184054c7`.

Post-freeze checks recomputed every frozen file hash, the source manifest/audit/
provenance hashes, the dependency lock and implementation hashes, and the separate
configuration hash. All matched. Full record accounting, no record or duplicate-
group overlap, and class coverage were rechecked from partition manifests: passed.
All 279 noisy recordings remain included. No quality exclusions or AF deferrals
were made; there are no AF predictions or performance metrics.

A second `rhythmlens split` returned exit 1 and refused overwrite, as designed.
`rhythmlens audit --no-previews` after freeze also returned exit 1 and refused to
replace the frozen audit snapshot. These are expected protections, not unresolved
blockers. Patient independence cannot be verified because subject IDs are unknown;
record-level grouping and exact duplicates do not substitute for patient IDs.

## Failures and corrections before freeze

- Initial synthetic checks found one unused import and an unsafe-header-filename
  diagnostic mismatch. Removed the import and checked the referenced filename
  before parsing it with WFDB; tests passed.
- During the mirror-option edit, an omitted function parameter caused two lint
  diagnostics and 8 synthetic test failures. Added the parameter; the full suite
  subsequently passed. No real artifacts were frozen with that implementation.
- The first acquisition failed with sandbox DNS error. Approved network access
  succeeded, but the primary host transferred only approximately 3 MB in ten
  minutes. Interrupted that incomplete attempt (exit 130); its partial file was
  removed. The official mirror completed both downloads successfully.
- The initial real audit treated optional checksum and initial-value discrepancies
  as fatal: 7,963 records stopped at checksum mismatch and 565 at initial-value
  mismatch, yielding no decoded eligible records in that preliminary report.
  Inspection showed every published checksum is zero. Updated the explicit
  pre-freeze policy to flag zero checksums as unverified and format-16 initial
  discrepancies as advisory, while requiring archive-byte and independent
  decoder agreement and blocking nonzero checksum mismatches. Added a dedicated
  synthetic test and reran all 40 tests and the full real audit before freezing.

No unresolved blocker remains. All versions are recorded in the audit JSON and
uv.lock. Source, tests, configs, de-identified manifests, provenance, split
snapshots and curated reports were checked as trackable. Downloads, MAT/header
samples and preview PNGs were checked as ignored and absent from the untracked
Git candidate list. No model training, dataset-wide normalization, dashboard,
commit or push was performed.
