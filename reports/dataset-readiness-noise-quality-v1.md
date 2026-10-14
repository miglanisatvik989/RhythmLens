# Dataset readiness for noise robustness and separate quality assessment v1

Status: **readiness research and frozen proposal only; acquisition, local audits,
implementation and results pending**. Reviewed 2026-10-14 against clean branch
`feat/noise-robustness`, HEAD `564558a460b55f58c5c6ff970abbe78783caeeca`.
No intervening work was found. No dataset was downloaded, access terms accepted,
signal decoded, window/split generated or experiment run. Only public documentation,
header text and annotation-reader documentation were inspected online.

Goal: compare noise augmentation, calibrated uncertainty, confidence deferral and
separate quality-aware deferral while preserving automatic AF detection and a
manageable reviewer workload. Deferred AF recordings are not automatic detections.
Future confidence-only and quality-aware policies need matched coverage comparisons.
The present PR prepares datasets and [noise methods](../docs/experiments/noise-robustness-v1.md),
not those policy implementations or a reviewer UI.

## Selected identities and primary-source facts

### MIT-BIH Noise Stress Test Database (NSTDB)

Use **NSTDB 1.0.0**, DOI **10.13026/C2HS3T**, published 1999-08-03,
not MIT-BIH Arrhythmia recordings as a new rhythm training cohort. The official
release separates noise-only `bw` (baseline wander), `ma` (muscle artifact),
`em` (electrode motion) from twelve already-corrupted ECG examples derived from
118/119. The examples' beat annotations are not Challenge four-class labels.
Only the three noise-only records are proposed inputs. Official noisy examples
use 24, 18, 12, 6, 0, −6 dB; our scaling below is a separate development definition.
Access is public subject to **ODC-By-1.0**, including attribution; no credentialed
access or access-term acceptance was performed. Source: [official release](https://www.physionet.org/content/nstdb/1.0.0/).

Exact future file roots: `https://physionet.org/files/nstdb/1.0.0/`;
metadata `RECORDS`, `SHA256SUMS.txt`, `nstdb.txt`; proposed signal/header pairs
`bw.dat`/`bw.hea`, `ma.dat`/`ma.hea`, `em.dat`/`em.hea`.
The [release notes](https://physionet.org/files/nstdb/1.0.0/nstdb.txt) describe the
second edition and copied example annotations; those examples remain excluded.

The official [bw header](https://physionet.org/files/nstdb/1.0.0/bw.hea),
[ma header](https://physionet.org/files/nstdb/1.0.0/ma.hea), and
[em header](https://physionet.org/files/nstdb/1.0.0/em.hea) each specify two channels,
360 Hz, 650,000 samples/channel, format 212, names `noise1`/`noise2`, zero gain and
no explicit units token. These are noise channels, not named diagnostic ECG leads.
[WFDB header documentation](https://physionet.org/physiotools/wag/header-5.htm)
identifies zero gain as uncalibrated; a decoder's default gain/unit is not a measured
voltage calibration. We will normalize digital noise shapes to unit RMS and impose
target amplitude in ECG mV, recording that distinction. Native calibrated noise
voltage and volunteer identities/channel-to-person linkage are **unverified**;
record names do not establish subjects or independent noise cohorts.

### Brno University of Technology ECG Quality Database (BUT QDB)

Use **BUT QDB 1.0.0**, DOI **10.13026/kah4-0w24**, published 2020-07-22,
under public **CC BY 4.0** access. It contains 18 long recordings from 15 subjects:
one ECG channel at 1,000 Hz and three accelerometer channels at 100 Hz. Quality
classes mean: 1, morphology delineable; 2, morphology partly obscured but QRS
detectable; 3, QRS detection unreliable. Class 0 means unannotated, not good.
CSV annotations contain start/end/class triplets for three experts plus consensus.
Three recordings are fully annotated; others have selected annotated intervals,
including extra poor-quality intervals. Do not assume every sample is labeled or
that fixed project-window labels already exist. The first three record-name digits
are documented subject IDs; the last three identify the measurement. Thus
100001/100002 and 103001/103002/103003 must stay grouped. Source:
[official release](https://physionet.org/content/butqdb/1.0.0/).

Exact future file root: `https://physionet.org/files/butqdb/1.0.0/`;
metadata `RECORDS`, `SHA256SUMS.txt`, `LICENSE.txt`, `subject-info.csv`,
`ann_reader.m`; per-record directories contain `_ECG`, `_ACC`, `_ANN` files.
The [subject metadata](https://physionet.org/files/butqdb/1.0.0/subject-info.csv)
corroborates repeated measurement IDs; do not publish demographics unnecessarily.
The [author's annotation reader](https://physionet.org/files/butqdb/1.0.0/ann_reader.m)
uses four triplet groups and MATLAB sample ranges. A future parser must validate
one-based inclusive endpoints against signal length, then explicitly convert to
zero-based half-open ranges; no annotation CSV was acquired or interpreted here.
Overlaps, missing consensus, endpoint inconsistencies and zero-coded coverage need
an audit before any window labels can be derived.

The public [100001 ECG header](https://physionet.org/content/butqdb/1.0.0/100001/100001_ECG.hea)
has gain 1.996 ADC units/uV and baseline −12200; it supports
`mV = (digital − baseline) / gain / 1000` **for that header**, not a universal gain.
The [matching ACC header](https://physionet.org/files/butqdb/1.0.0/100001/100001_ACC.hea)
uses `mg` and channel-specific gains/baselines. Audit every header separately;
accelerometry is excluded from the current ECG-only scope. This is header-based
unit evidence, not a local signal audit.

## Local readiness and separate tasks

| Input | Local state / identity | Allowed task and pending work |
| --- | --- | --- |
| Challenge 2017 | Existing 1.0.0 / REFERENCE-v3 manifests and frozen splits; train 5969, validation 1280 | Rhythm N/A/O/~; existing train-only preprocessing; consumed holdout 1279 excluded from all new development |
| NSTDB | Not locally acquired or audited | Noise shapes and corruption metadata only; preserve record/channel/time support across roles |
| BUT QDB | Not locally acquired or audited | Separate quality labels only; subject-grouped audit/splits and window definition pending |

No subject identities can be inferred for Challenge or NSTDB. BUT's documented
identifier convention must be verified on all acquired records; only then use
verified subject grouping before windowing. Source/time/near-duplicate independence
across datasets is not established by distinct names. Screen available provenance
and duplicate/linkage metadata; block any verified cross-boundary relation rather
than rewriting existing splits. BUT quality class 2 is not an AF label or a proven
acceptable-for-rhythm decision. Naturally noisy Challenge ~ has unknown underlying
rhythm; synthetic SNR never becomes a human quality label. Dataset shift in device,
lead, sampling and class mix makes quality transfer an **untested hypothesis**.

## Acquisition and audit requirements — future work

Proposed ignored roots: `data/raw/nstdb/1.0.0/` and `data/raw/butqdb/1.0.0/`;
new trackable metadata roots: `data/manifests/nstdb-v1/` and
`data/manifests/butqdb-v1/`. These are names reserved in the proposal, not newly
created datasets/manifests. The [declarative dataset draft](../configs/noise-quality-datasets-v1.yaml)
contains exact URLs; it is not a supported acquisition command.

On a separately authorized acquisition task: record versioned source URL, date,
license/attribution, bytes and SHA256 for each actual file; validate official
SHA256SUMS and file membership, quarantine mismatches, and retain original bytes.
Current local input hashes, inventories and acceptance records for these new
datasets are **pending**, never fabricated from remote filenames. Exclude unused
NSTDB corrupted examples. Preserve upstream metadata licenses and citations.

A future non-destructive audit must check WFDB format, channels, sample counts,
gain/baseline/units, missing-value sentinels, finite conversion, checksums, timing,
and verified duplicates. Do not reuse the Challenge-specific format-16 decoder
contract for NSTDB format 212 or BUT signals by silently relaxing it. New adapters
need their own explicit compatibility evidence and synthetic unit/shape tests.
Native noise gain is unknown; record a unitless-shape path instead of pretending
WFDB defaults establish absolute noise mV. Stop on unexpected headers or files;
do not silently select replacement records or channels after results.

For BUT, inventory all expert/consensus annotations, class-0 gaps, coverage and
interval endpoints. Freeze subject groups before windows. Window duration, stride,
transition handling, consensus aggregation and future quality split proportions
are **unresolved**; do not invent a majority-vote label or expose an unannotated
window as quality ground truth. No BUT quality model is trained in this milestone.
A later harmonization plan may convert per-header uV to mV and resample 1000→300
with up/down 3/10, preserving time-aligned annotation support; that is not proof
that a quality model transfers to Challenge. No ACC resampling/fusion is proposed.

## Noise-role separation and harmonization

The [noise protocol](../docs/experiments/noise-robustness-v1.md) freezes proposed
nonoverlapping temporal pools per noise record, shared across its two channels:
training augmentation [0,600)s, development reserve [660,1140)s, stress
[1200,1800)s. Gaps and tail are unused. Slice pools **before** resampling, resample
each independently 360→300 (5/6), trim one second per end, and forbid wrapping or
cross-role/filter support. Both channels inherit the same role boundaries.
Repeated offsets within a pool are reuse, not independent observations. These
are temporal partitions of the same short source records, **not subject- or
record-independent noise sets**. Unknown common volunteers and long-range
correlation remain limitations despite 60-second gaps. Stronger noise-source
independence needs separately acquired verified sources; this proposal claims none.
No pool, offset plan, decoded noise or resampled cache exists yet.

## Existing evidence and preservation gap

Temporal checkpoints 42/43/44, their preprocessing, resolved configurations and
training reports were verified by opaque hashes against
`reports/rhythm-multiseed-v1.json`; selected epochs remain 31/42/39. Existing
calibration and [internal rhythm-only results](temporal-internal-rhythm-evaluation-v1.md)
are completed historical evidence. No holdout cases or outcomes are used to choose
this design; the original holdout is consumed and excluded. Any subsequent final
claim needs independently sealed evaluation evidence after all choices freeze.

No documented durable destination/workflow was found for the 325 existing local
output artifacts. `completion.json` records external archival pending. Local
existence, Git reports, checksum indexes and dataset ZIPs are not complete artifact
backups. Owner must identify an approved destination, retention/access policy and
restore/integrity verification procedure. No upload or invented backup occurred.

## Proposal verification

Checked primary documentation/header/reader citations, repository-local links,
strict YAML syntax (including duplicate-key rejection), literal false authorization,
unsupported-schema status, all candidate hashes, noise interval separation and
condition counts. Repository ignore/lock contracts passed; frozen verification
reported 17 unchanged snapshot files, original split counts and zero ECG files
opened. Existing source/configuration/protocol/manifest/authorization hashes and
all 325 output artifact hashes remain unchanged. Only the seven intended
new/updated documentation/configuration files are in scope. Git author/committer
identity and authenticated repository write access were verified without exposing
credentials. No package installation, broad tests or experiments were run for this
documentation/configuration milestone.

## Decisions and blockers

- Acquisition and per-file audits are pending; no new dataset is locally ready.
- Proposal parameters below are development choices, not clinical thresholds;
  no AF sensitivity target, maximum review workload, compute budget or deadline
  has been specified. Owner must supply these and the archival destination.
- Before a later quality task, resolve BUT consensus/window semantics and intended
  downstream usability definition. Class 2's QRS detectability alone cannot
  establish adequate AF assessment.
- Before matched-coverage policy comparisons, specify equal coverage targets,
  development selection cases, tie rules and automatic AF denominators in a
  separate protocol; do not retune existing confidence policies here.
- A separate sealed cohort and reviewer workflow remain unresolved. No quality
  model, transfer result, reviewer app or noise-robustness result is claimed.
