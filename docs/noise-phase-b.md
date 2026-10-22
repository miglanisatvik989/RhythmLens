# Phase B implementation and execution boundary

**Execution update, 2026-10-22:** the separately authorized single attempt completed
CUDA training and CPU comparison. See the [outcome and compact evidence](../reports/noise-phase-b-v1-results.md)
for actual results, integrity checks and limitations. The retained true Phase B
flag records that attempt only; the fixed run/supervisor roots are occupied and
another execution is not authorized. Both original proposals and historical
Phase A authority/results are unchanged. The implementation-time disabled and
untested statements below describe the earlier synthetic milestone.

Phase B implements the existing [frozen protocol](experiments/noise-robustness-v1.md)
with generated-fixture verification only. **Real CUDA correctness and augmentation
benefit remain untested.** See the [verification record](../reports/noise-phase-b-v1-verification.md).
The historical [Phase A results](../reports/noise-phase-a-v1-results.md), its true
single-attempt flag and both original false/unsupported proposals are unchanged.

## Interface and authority

The dedicated CLI is separate from the existing launchers:

```sh
.venv/bin/python -m rhythmlens.noise_phase_b --help
# Both reject immediately with the committed disabled configuration:
.venv/bin/python -m rhythmlens.noise_phase_b --config configs/noise-phase-b-v1.yaml --stage train
.venv/bin/python -m rhythmlens.noise_phase_b --config configs/noise-phase-b-v1.yaml --stage evaluate
```

`configs/noise-phase-b-v1.yaml` is a separate strict versioned contract with
literal `execution_authorized: false`. Duplicate keys, unsupported schemas,
nonboolean authorization and changes to any other contract field are rejected.
The sorted compact JSON SHA256 excluding only this flag is
`71bb8347193a0f9e67bdf34686c0c48e42c5fd4525b99a83068c2088d1401b82`.
Real execution additionally requires a separately authorized, committed true
flag, clean committed source and `--authorize-execution`. Rejection occurs before
metadata discovery, input hashing, signal decoding, checkpoint loading, prediction
parsing or output creation. No existing authorization grants Phase B authority.

The two stages deliberately require different existing environments:

| Stage | Torch | Runtime | Purpose |
|---|---|---|---|
| `train` | 2.8.0+cu128 | Original CUDA profile, GPU index 0, two CPU threads | Train three augmented seeds; select on clean validation |
| `evaluate` | 2.8.0+cpu | CPU float32, two threads, deterministic, batches of 16 | Evaluate selected augmented candidates; reuse sealed original predictions |

Both environments pin Python 3.11.16, NumPy 2.2.6, SciPy 1.15.3, WFDB 4.3.1 and
PyYAML 6.0.3; existing CPU/CUDA locks are bound. The stages are separate process
invocations under their matching environments, with identical source/configuration.
No environment installation, upgrade, GPU probe or benchmark was performed during
implementation. An owner-approved budget and external supervision are prerequisites
for real execution; the runner itself does not enforce a wall/RAM/GPU/disk budget.
The separate external supervisor below supplies sampled supervision.
The Phase A supervisor has fixed Phase A destinations and must not launch Phase B.

## Declared inputs and baseline reuse

Declarations were copied from tracked configuration, audit and aggregate evidence,
without inspecting protected local bytes. They bind the original three training
YAMLs and resolved configs, split/source identities, original preprocessing,
checkpoint/report/export identities, reviewed NSTDB provenance/audit/roles/pools,
Phase A configuration and its completed aggregate/report evidence.

The baseline root is the completed fixed Phase A root. Its execution revision and
all ten compactly published exact-byte identities are bound, including status,
registry, plans, accounting, seal, metrics, bootstrap metadata/arrays and artifact
index. Runtime verifies completed status, the exact full indexed inventory/bytes,
the original execution/input contract, unchanged current inputs and existing source
modules, row/label/group
identity and exact reconstructed stress plans. CPU evaluation verifies every
baseline prediction and required gate/batch/ledger seal before comparisons. It
also checks matching physical, corrupted and transformed digests and corruption
metadata across the two arms. This reuses all original candidates without loading
them or launching another Phase A run. Any incompatibility blocks; there is no
baseline regeneration fallback. Hash consistency does not independently authenticate
publisher indexes or establish clinical validity.

Availability and actual matching of these local bytes remain unverified in this
implementation task. Missing declarations are not filled by probing signals,
checkpoints or predictions. A separately authorized readiness audit must verify
inputs/environment/capacity before real execution.

## Training behavior

New runs are identified as `temporal-noise-v1` in the distinct ignored Phase B
root. The original YAML and its numerical resolved settings remain unchanged;
original `_experiment` context is preserved as source provenance. Phase B metadata
identifies the new experiment separately. Runtime compares all resolved numerical
settings with the original candidate's bound training configuration. The historical
architecture, dropout, AdamW learning rate/weight decay, epoch cap, patience,
length buckets, remainder handling, loader generator and each seed are preserved.

Only original clean physical training arrays are cached. For epochs starting at 1,
all record plans are persisted before any optimizer step. The existing
`training_plan` draws PCG64 from the first eight little-endian SHA256 bytes of
compact JSON `[20261014,"train",model_seed,epoch,record_id]`. It draws `random()`
for p<0.5, then selected type bw/ma/em, channel, SNR from 24/18/12/6 dB and
contiguous offset, in exactly that order. No global NumPy/Python/Torch RNG draws
are added by augmentation. Original sampler epochs start at 0, preserving their
historical seed-plus-epoch sequence independently of the one-based corruption key.

Only reconstructed `augmentation-train` pools are used for training, with audited
role descriptors and hashes. Stress uses its isolated `stress` role; the
`development-reserve` remains unused. Existing physical scaling, achieved-SNR,
flat-case and no-redraw behavior are reused unchanged. Corruption happens in
physical 300 Hz mV before the unchanged median/fixed-training-RMS transform.
The original saved scale is reused, never refitted. A fresh planned physical view
is transformed each epoch; normalized or already-corrupted cached samples are not
reused. Plans, original/corrupted/transformed hashes and actual corruption metadata
are retained locally per epoch/batch.

Clean validation uses the original training device and evaluator, without noise.
Only strictly improved four-class clean macro F1 replaces the best checkpoint;
ties increment the original patience. No stress outcome can select a checkpoint.
All three new candidates must freeze before CPU stress evaluation. Checkpoints use
the existing safe tensor/primitive format; `metadata.phase_b`, experiment name,
source, exact config, selected epoch, seed and original preprocessing bindings
distinguish them from historical baseline artifacts. Historical outputs are never
written. There is no resume implementation.

## Evaluation, accounting and uncertainty

CPU evaluation uses the exact 55 Phase A conditions, rows, physical plans,
preprocessing, length-bucket/no-shuffle ordering, padded-prefix masks, float32
eval/inference mode, raw T=1 softmax and original N/A/O/~ argmax order. Only the
three new models are loaded. Original clean reproduction gates are reused as
verified baseline evidence; augmented predictions are not required to reproduce
unaugmented logits.

Every seed/condition retains all eligible cases with completed, failed or blocked
status. Technical errors prevent completed metrics. All 165 new arrays, batch
metadata/arrays, plans, registry, journal and full ledger seal and verify. Then a
composite seal binds those outputs, the complete original baseline evidence and
all three completed training checkpoints. Comparative scoring starts only after
both arms' complete evidence and physical pairing checks pass. Inputs and seals
are checked again before final metric publication.

The original scorer supplies each arm's full prescribed AF accounting, 4×4
confusion/4×7 outcomes, coverage, probability/reliability/false-AF metrics,
repetition summaries, within-record error/AF indicators, equal-seed mean/sample SD
and stress-minus-clean deltas. AF sensitivity divides correct automatic AF by all
AF-labeled eligible cases; zero AF yields null with a reason. Predicted `~` stays
assessed rather than deferred. Flat/inapplicable-SNR findings remain separate.
Calibration, confidence deferral and quality gating stay disabled.

The comparison adds matching-seed augmented-minus-original primary metrics for
every condition and each repetition-mean cell, plus equal-seed mean/sample SD.
One shared 10,000-draw class-stratified duplicate-group PCG64 bootstrap (seed
20261016) retains both arms, all seeds/conditions/repetitions. Repetition metrics
are computed before averaging; no consensus prediction or seed pooling is used.
Saved statistics contain original, augmented and paired differences for absolute
metrics and stress-minus-clean changes. Linear 2.5/97.5 percentile intervals include
paired arm differences and differences of stress deltas. Undefined AF stays null
in metadata (NaN encoding only in local statistics). Invalid draws fail without
replacement. Generated tests reduce draws only within their fixtures.

## Progress, lifecycle and preservation

The fixed root is
`outputs/noise-robustness/noise-robustness-v1/phase-b`. Training creates it
exclusively; an existing root or symlink blocks. Each stage has an exclusive claim
file, preventing concurrent/repeated stage starts. CPU evaluation is a distinct
stage after `training_completed`, with an absent `evaluation/` child. It is not
training resume. Failed/interrupted/complete roots cannot be restarted or bypassed
by choosing another output root.

`progress.json` and terminal bars show training seed/epoch/batches and a measured
current-epoch ETA, plus separate clean-validation, clean-evaluation, stress,
sealing and indeterminate bootstrap/scoring stages. Batch completion advances
after the corresponding successful batch evidence publication. ETA does not
predict early stopping or whole-run completion. Clean validation and bootstrap
have no invented percentage. Training validation uses the unchanged evaluator;
its internal batches are not separately counted. No resource quota is implied.

`training/` retains its registry, rows, pool descriptors, per-seed config/state,
epoch plans/batch evidence, history, best/last checkpoints, status and reports.
A complete artifact index binds every training file. `evaluation/` retains its
registry, Phase A plans, journal, full ledger, new batch/complete predictions,
augmented/composite seals, both-arm metrics, shared bootstrap arrays/metadata,
artifact index and completed status. Top-level status binds completed evaluation.
All bytes stay ignored/local except reviewed compact documentation/configuration.

Failures preserve successfully written files and available in-memory corruption
metadata when the exception handler can write it. Unwritten signal/prediction
buffers are not claimed retained. Optimizer steps can finish before batch evidence
or checkpoint publication; those states are not resumable. Abrupt kill, storage
failure or power loss can leave partial writes/running status, and cleanup is not
guaranteed. Inspect actual retained evidence; do not delete/reuse partial roots.
A partial metrics file is not a completed run without lifecycle/index verification.

## Remaining decisions

Real execution requires separate owner authorization and a concrete resource/time
budget with supervision across CUDA training and CPU evaluation, an input-byte
readiness audit, exact stage environments, available compatible CUDA hardware,
unchanged grouping and original artifacts, compatible complete Phase A evidence,
and an absent fixed root. This milestone proves neither local input readiness nor
actual GPU execution or augmentation benefit.

Unknown patients/noise volunteers, shared noise sources, historical runtime/source
differences, checkpoint selection and reused validation limit future comparisons.
Native voltage calibration and source-label recognizability under severe noise
remain unverified. Durable archival/restore awaits an owner destination. No BUT,
quality model, calibration, UI, consumed holdout or new final evaluation is included.

## External two-stage supervisor

Added 2026-10-22; synthetic verification only. The separate
`scripts/supervise_phase_b.py` launches the two existing stages without changing
runner code, numerical policy, source bindings or any authorization flag. It
reuses the Phase A process-table/output-size primitives, not its launcher or
stage interpretation. Linux procfs, pidfds and main-thread interval timers are
required. Kernel pidfd availability is checked before launching a child.

All budget fields below are mandatory invocation inputs, **not approval**. After
separate owner approval, a focused committed Phase B flag change and a fresh
preflight, the explicit interface is:

```sh
.venv/bin/python -u -B scripts/supervise_phase_b.py --authorize-execution \
  --training-wall-seconds 28800 --training-cleanup-seconds 600 \
  --evaluation-wall-seconds 21600 --evaluation-cleanup-seconds 300 \
  --overall-wall-seconds 50400 --host-rss-bytes 4294967296 \
  --training-gpu-bytes 7516192768 \
  --training-output-bytes 8589934592 --total-output-bytes 12884901888 \
  --min-host-available-bytes 1073741824 --min-disk-free-bytes 10737418240
```

The committed false flag rejects this invocation before telemetry, subprocess
launch or destination creation. Neither supplying limits nor the historical
Phase A true flag grants Phase B authority. Both original proposal byte bindings
and all existing policy fields remain fixed.

The exclusive supervisor directory is `outputs/noise-phase-b-supervision-v1/`.
Existing runner or supervisor destinations, including dangling symlinks, reject
launches; failed claims cannot be reused. Claim creation does not create the
runner root. Training uses `environments/cuda128/.venv/bin/python`; evaluation uses
`.venv/bin/python` with `CUDA_VISIBLE_DEVICES` empty. Both environments' pinned
metadata, the execution configuration, revision and all committed file bytes are
checked before launch/handoff. This prevents a CUDA Torch build being selected
for CPU evaluation without initializing any GPU for the check.

Training must exit successfully with no live owned descendants, top-level
`training_completed`, a verified full training artifact index and all three
completed seed reports/checkpoint byte hashes. Only then can CPU evaluation enter
the existing root. Evaluation must exit successfully and verify its completed
status, bound full artifact index and required final seal/comparison/metric
metadata. No checkpoint deserialization or prediction-array parsing is used by
the supervisor's completion checks. Runner verification remains responsible for
numerical and semantic evidence. No retry, resume or replacement root exists.

Sampling is nominally once per second, readable progress snapshots every five
seconds. It records aggregate live runner/descendant RSS, host `MemAvailable`,
free disk, cumulative runner-plus-supervisor file sizes and GPU 0 memory. GPU
usage is conservatively **device-wide total minus free**, including reservations
and other workloads; there is no invented run/process attribution. The 7 GiB GPU
ceiling applies during training. CPU evaluation's fixed CPU interpreter and empty
CUDA visibility prevent selection of the CUDA backend; unrelated device-wide
activity is still displayed. Required telemetry failure, malformed progress,
monitor/storage error, resource breach or nonzero runner exit blocks handoff.

Each stage's hard deadline is the earlier of its stage limit and the original
global deadline. SIGINT is sent at that deadline minus the stage cleanup
allowance, or earlier on a breach. Cleanup ends at the earlier of the hard
deadline and interruption time plus that allowance. Remaining identified tasks
then receive SIGKILL; forced or unobservable cleanup is recorded incomplete.
Any breach stops the entire milestone even if the runner later exits zero.
Metadata-only source/handoff verification is bounded by an interval timer at the
same work cutoff; it cannot silently reset or consume the next stage's clock.
The global clock begins at the exclusive supervisor claim and includes both
stages, handoff verification and cleanup. Initial authority/source/environment
checks precede that claim and launch no experiment.

Each task is identified by PID and kernel creation time; stable pidfds are used
for signaling, avoiding a PID-reuse race between checking identity and sending a
signal. Private-session membership and observed ancestry retain descendants,
including observed children later reparented or entering another session. A
recycled root PID's unrelated session is not adopted. SIGINT and supervisor
SIGTERM trigger bounded cleanup. Unrelated processes are never signaled.

`claim.json`, stage launch/failure/verified records, supervisor `status.json`,
`progress.json`, `snapshots.jsonl`, readable `progress.log` and separate
`train-stdout.log` / `evaluate-stdout.log` remain ignored/local. Progress reads
only the existing atomic runner `progress.json`: seed, epoch, published batches,
elapsed time and last measured current-epoch/condition ETA. Validation,
preflight, sealing and bootstrap remain indeterminate without incremental
metadata. No whole-training completion percentage or whole-run ETA is invented.

These are sampled limits, not kernel memory/disk quotas or hard real-time
scheduling guarantees. Polling, procfs/file traversal and the bounded two-second
GPU query introduce latency; subsecond peaks and between-poll output growth may
be missed. Shared pages can be counted more than once in aggregate RSS. The
watchdog's own RSS is outside the runner-tree ceiling, while the host available
memory floor observes machine-wide pressure. Verification uses supervisor memory.
Very short-lived descendants that detach/reparent between observations may evade
discovery; normal private-session descendants and previously observed identities
are retained. Abrupt supervisor SIGKILL/power loss cannot guarantee cleanup.
Storage failure may prevent writing new diagnostics; retained files are never
deleted to fit a budget. These limits do not establish CUDA correctness or
augmentation benefit.

Synthetic supervisor verification uses generated metadata, mocked resource
telemetry/clocks and harmless Python children. It covers handoff, failed stages,
all budget categories, unavailable telemetry, deadline/verification cutoffs,
forced/incomplete cleanup, PID reuse/reparenting, unrelated-process protection,
exclusive/dangling destinations, output accounting, readable progress and CPU
interpreter selection. Focused supervisor verification: 39 tests passed in 1.92 s.
The final full synthetic/regression suite passed 537 tests in 54.71 s; lint/format (44 files), frozen
verification (17 identities), repository contracts, dependency compatibility
(58 distributions), both offline locks (60/74 packages), and sdist/wheel builds
passed. All 22 existing Phase A source files and the execution-policy hash remain
unchanged. Final push/PR CI results are recorded in draft PR #10.
No real Phase B run, GPU allocation, dependency installation or budget approval
was performed for this addition. Fresh input/environment/capacity checks and
explicit owner approval remain prerequisites.
