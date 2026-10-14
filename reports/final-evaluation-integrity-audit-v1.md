# Final-evaluation integrity audit v1

Status: metadata/provenance audit only. No final-evaluation execution is authorized.
Repository HEAD at audit: `47d198e15f0fdcfbd2210ea9cf6700af7a1b3889`.
Existing uncommitted reviewed-v1 report/index edits and role metadata were preserved.
No holdout signal, waveform preview, prediction or performance output was opened
or generated for this audit. Partition labels/group identifiers were read only
for the requested metadata integrity/overlap checks; their already-published
supports are not model performance.

## Conclusion

The existing holdout is **performance-unassessed in the available repository
records, but cannot be described as strictly never-seen or independently blinded**.
All 8,528 recordings were decoded in the pre-freeze data-integrity audit. Twenty
waveform previews were generated before split assignment; **two previewed IDs,
A00004 and A00022, subsequently entered the holdout**. That intersection was
established only by joining recorded preview paths to frozen split metadata; no
images or waveforms were opened. Preview generation does not prove human viewing,
and available records cannot establish who viewed them or whether they influenced
choices. Public labels and signal-derived integrity summaries were available.

No record or verified exact-duplicate group crosses train/validation/holdout.
Patient, source and chronological independence cannot be established. Thus the
current split could support a carefully qualified, first recorded internal
recording-level performance evaluation after review, but not a claim of a wholly
untouched, patient-independent external test. For a strict untouched final test,
use a newly sealed independent cohort after the candidate and protocol are frozen.
Do not silently exclude the two exposed records or regenerate an existing split.

Earlier reports use “untouched holdout” alongside statements that the audit and
previews preceded assignment. That wording supports no recorded post-freeze
performance use; it must not be read as proof that the eventual holdout signals
were never accessed. This audit supplies the narrower, explicit interpretation
without rewriting historical evidence.

## Split creation and identities

Sources: [acquisition provenance](../data/manifests/challenge2017-v3/provenance.json),
[data-audit report](challenge2017-audit.md), [verification chronology](challenge2017-verification.md),
[frozen configuration](../splits/challenge2017-v3-seed42-v1/config.json),
[summary](../splits/challenge2017-v3-seed42-v1/summary.json),
[frozen identities](../splits/challenge2017-v3-seed42-v1/frozen.json), and
[split implementation](../src/rhythmlens/splits.py).

Official Challenge 2017 training archive and REFERENCE-v3 labels were acquired
and integrity-audited. All 8,528 valid labeled recordings were retained; no noisy
or flat exclusions were made. Labels changed between archive and v3 for 148
records; v3 was fixed before the split. The audit decoded signals, compared
MAT/WFDB payloads and exact duplicates, and produced class summaries/previews.
The format-16 initial-value and zero-checksum integrity policy was corrected
before freeze using dataset-wide header/integrity findings. These were decoder
integrity decisions, not recorded holdout model-performance tuning, but demonstrate
that pre-freeze data inspection was not restricted to eventual training records.

Freeze: `2026-10-12T06:08:30.952705+00:00` (UTC). Seed 42, per-class allocation:
sorted exact duplicate groups, `random.Random(42)` for equal-size ties, stable
largest-group-first ordering, then largest remaining target deficit with fixed
train/validation/holdout tie order and an empty-partition class-coverage safeguard.
Target fractions were 70%/15%/15%. Verified duplicates are indivisible; conflicting
labels block freezing. Recording IDs are not inferred patient IDs.

| Partition | N | A | O | ~ | Records | Exact duplicate groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 3553 | 530 | 1691 | 195 | 5969 | 5967 |
| Validation | 762 | 114 | 362 | 42 | 1280 | 1280 |
| Holdout | 761 | 114 | 362 | 42 | 1279 | 1279 |

Assignment SHA256: `fb27f37a8f9dcd1d8931116e252290a68be29844b5b81badc3223954184054c7`.
All frozen partition/config/summary/README byte hashes matched `frozen.json` in
this audit. Full record union equals the source-manifest ID set, with no repeated
IDs across partitions. The two verified exact duplicate pairs, A00227/A01176 and
A02656/A03123, both have label N and both remain entirely in train. All pairwise
record-ID and duplicate-group intersections are empty. Exact duplication checks
do not establish absence of near duplicates or repeated patients.

## Exposure audit: fact versus unknown

| Item | Verified repository evidence | What remains unknown |
| --- | --- | --- |
| Holdout labels | Public source labels were used for class-stratified allocation; holdout IDs/labels and aggregate supports are stored in tracked metadata. Development integrity readers also read all partition labels/groups for accounting. Labels were not blinded. | Human access outside recorded actions and any undocumented decision use cannot be reconstructed. |
| Holdout signals | Pre-freeze audit decoded all recordings, including those later assigned holdout; full source checksums were verified before freeze. Post-freeze development reports record no holdout-signal access. | Complete access history cannot be established from self-reported flags or Git. |
| Previews | Recorded preview IDs partition as 14 train, 4 validation, 2 holdout. Holdout preview IDs are A00004 and A00022. | Whether any particular preview was viewed by a person/agent, or affected decisions, is not logged. |
| Holdout predictions/metrics | No holdout-named output files were found in the current local outputs inventory. All six saved baseline evaluation reports are validation-only; completed reports describe validation results, not holdout performance. | Filename absence is not proof against arbitrary-named, deleted, external or unlogged results. |
| Decisions | Checkpoint/early-stopping, architecture prioritization and exploratory calibration/deferral used validation; training preprocessing fitting used only 5,969 train records. User now provisionally chooses raw temporal predictions without seed ranking. | No evidence establishes that global pre-freeze inspection had zero influence; no causal blindness claim is warranted. |

The available local baseline lifecycle inventory contains 46 run/evaluation/
preflight/fit/status JSON files: 33 have explicit holdout-access flags, all false;
13 lack such flags (absence is not proof of no access). Six are explicitly
validation evaluation reports. The two reviewed-v1 calibration containers
`frozen-policy.json` and `assessment/summary.json` identify validation-derived
analysis and record no holdout-signal access via their registry/summary provenance.
These are corroborating records, not an immutable access log. There is no general
file-access audit trail, label custodian separation or cryptographic proof of
never viewing the public data. No heldout prediction values or performance
metrics were computed or inspected for this integrity audit.

[Development data loading](../src/rhythmlens/rhythm_data.py) rejects holdout as an
execution partition; it does read frozen holdout metadata for hashes, labels and
group accounting. [Existing evaluation](../src/rhythmlens/rhythm_training.py) loads
validation explicitly. [Calibration](../src/rhythmlens/calibration.py) uses saved
validation exports and holdout metadata only for original-split integrity. These
controls support documented post-freeze separation, but cannot undo pre-freeze
exposure or prove events outside this code path.

## Identifiers and independence

The source manifest has recording IDs, `subject_id`, labels, file paths, signal
integrity/decoder fields, duration, duplicate-group and digital checksum fields.
**All 8,528 subject IDs are `unknown`.** No verified patient/subject linkage,
collection timestamps, site/device/source grouping IDs or longitudinal relation
keys are supplied. Duration is not collection time. A shared dataset provenance
(Challenge 2017/AliveCor) identifies a common source, not independent sources.
Record-name prefixes and numeric sequence do not identify patients or chronology.

Verified independence is limited to disjoint recording IDs and disjoint audited
exact-duplicate groups. Patient overlap, source distribution dependence, repeated
sessions, near duplicates and temporal leakage remain unverified. Do not invent
identities or use ordinary recording bootstrap intervals as patient-independent
uncertainty claims.

## Smallest feasible route to stronger evaluation

1. Seek a provenance/access clarification and verified subject/source/time linkage
   for existing records without further signal inspection. If available, audit
   those links before further claims. If verified groups cross the frozen boundary,
   stop the current final-evaluation proposal; do not repair frozen assignments.
   Even successful linkage would not erase documented preview/audit exposure.
2. For a strictly untouched and more independent final test, obtain a separate,
   previously unused ECG cohort with compatible physical units/labels, verified
   subject IDs and source/time provenance. Prefer a custodian who can seal signals
   and labels while duplicate/subject overlap is checked against development IDs
   and existing checksums. Freeze eligibility, label mapping, candidate checkpoints,
   preprocessing and analysis before anyone exposes its outcomes. No particular
   dataset, provider access or destination is assumed to be available.
3. Evaluate the existing frozen raw temporal checkpoints without retraining or
   adapting preprocessing to that cohort, after a cohort-specific amendment and
   explicit execution approval. If compatible labels/units or independence evidence
   are unavailable, remain blocked for the stronger claim. A separately acquired
   external cohort is not automatically independent without verified linkage.

A review may instead authorize a one-time *internal recording-holdout* evaluation
with the disclosed limitations. The [draft final protocol](../docs/experiments/temporal-final-evaluation-v1.md)
specifies that restricted option and the separate stronger-cohort gate. Neither
option is authorized here. Re-splitting previously audited/development-used data
cannot restore a never-seen holdout; dropping only the previewed records also
cannot remove dataset-wide pre-freeze access or unknown patient overlap.

## Checks and preservation

Only existing metadata/provenance/log records, documentation, source inspection,
file inventories and opaque hashes of the three checkpoint files were used.
Checkpoint bytes matched the recorded multiseed identities without deserializing
weights or running a model. No raw ECG, preview image, or holdout performance was
opened. Metadata overlap/union checks, frozen-file hashes and documentation checks
were performed; no experiment or test inference was run. All prior uncommitted
work is preserved. Candidate selection and uncertainty methods remain provisional
until the audit limitations and draft are reviewed. This audit does not certify
an untouched holdout or authorize final evaluation.

## Current implementation scope clarification

The audit above remains the historical exposure and independence record. The
[dedicated protocol](../docs/experiments/temporal-final-evaluation-v1.md) now targets
a qualified internal evaluation of the frozen rhythm-only baseline, not final
validation of the eventual quality-aware system. All three checkpoints remain
fixed and equally reported, without calibration, deferral or quality gating.
Inspecting these outcomes consumes this holdout for future development;
subsequent quality-gating changes require separate evaluation evidence. The
executor is implemented with real authorization false; no result or stronger
independence claim follows from synthetic verification.
