# Challenge 2017 v3 data audit

Validation passed: **True**.

Inventory: 8528 records; invalid: 0; label changes: 148.
Exact duplicate groups: 2; conflicting groups: 0.

| Label | Archive count | v3 count | Decoded | Duration min / median / mean / max (s) |
| --- | ---: | ---: | ---: | --- |
| N | 5050 | 5076 | 5076 | 9.047 / 30.000 / 32.111 / 60.953 |
| A | 738 | 758 | 758 | 9.987 / 30.000 / 32.345 / 60.207 |
| O | 2456 | 2415 | 2415 | 9.127 / 30.000 / 34.295 / 60.860 |
| ~ | 284 | 279 | 279 | 9.360 / 30.000 / 24.379 / 60.000 |

Flags: {"zero_header_checksum_unverified": 8528, "initial_value_mismatch_format16": 565}

No quality exclusions. Flat signals are flagged and retained; noisy labels remain eligible.
Subject IDs are unknown: recording and exact-duplicate grouping cannot establish patient independence.
No transformations beyond calibrated unit conversion; original MAT/header files are preserved.
No AF predictions or model metrics were computed; deferral/evaluation work remains pending.

Waveform previews (local ignored outputs; generated before holdout assignment):

- `outputs/challenge2017/previews/A00001.png`
- `outputs/challenge2017/previews/A00002.png`
- `outputs/challenge2017/previews/A00003.png`
- `outputs/challenge2017/previews/A00006.png`
- `outputs/challenge2017/previews/A00007.png`
- `outputs/challenge2017/previews/A00004.png`
- `outputs/challenge2017/previews/A00005.png`
- `outputs/challenge2017/previews/A00009.png`
- `outputs/challenge2017/previews/A00015.png`
- `outputs/challenge2017/previews/A00027.png`
- `outputs/challenge2017/previews/A00008.png`
- `outputs/challenge2017/previews/A00013.png`
- `outputs/challenge2017/previews/A00017.png`
- `outputs/challenge2017/previews/A00020.png`
- `outputs/challenge2017/previews/A00023.png`
- `outputs/challenge2017/previews/A00022.png`
- `outputs/challenge2017/previews/A00034.png`
- `outputs/challenge2017/previews/A00056.png`
- `outputs/challenge2017/previews/A00106.png`
- `outputs/challenge2017/previews/A00125.png`

Validation blockers:

None.

Machine-readable details, changed label IDs, duplicate groups, versions and hashes:
`data/manifests/challenge2017-v3/audit.json`.

Attribution and license: see `data/manifests/challenge2017-v3/provenance.json` and README.

## Published header advisories

All 8,528 original headers contain checksum zero. These values are recorded as
unverified, not used as proof of integrity. 565 header initial values differ
from the first MAT sample; format 16 decoding does not use this field. Both
advisories are retained per record. Every extracted file matches its archive
content, and SciPy MAT decoding agrees exactly with WFDB digital decoding.
Nonzero checksum mismatches remain blocking errors. This policy and its
pre-freeze correction are documented in `configs/challenge2017-audit-v1.json`
and `reports/challenge2017-verification.md`.

## Verified duplicate pairs

- A00227 and A01176: both v3 N.
- A02656 and A03123: both v3 N.

Physical arrays were also directly compared for exact equality before freezing.
Both pairs remain within single partitions. No conflicting duplicate labels
were observed. Exact duplicates do not identify all same-patient recordings
or near duplicates.

## Freeze outcome

All 8,528 eligible records were retained, including all 279 noisy recordings.
Train: 5,969; validation: 1,280; holdout: 1,279. No record or exact-duplicate
group overlaps; every partition contains N, A, O, and ~. See
`splits/challenge2017-v3-seed42-v1/README.md` and `summary.json` for actual
proportions and per-class counts. The holdout is now reserved for final evaluation.
