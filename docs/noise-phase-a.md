# Phase A implementation and execution boundary

Phase A is implemented and synthetically verified; **no real Phase A result exists**.
See the [verification record](../reports/noise-phase-a-v1-verification.md),
[frozen protocol](experiments/noise-robustness-v1.md), and
[reviewed NSTDB data evidence](../reports/nstdb-real-data-readiness-v2.md).
Phase B, quality gating, calibration, BUT and consumed-holdout reuse are excluded.

## Supported interface and authority

The dedicated interface is separate from `scripts/manage.py`:

```sh
.venv/bin/python -m rhythmlens.noise_phase_a --help
# Deliberately rejects execution with the committed disabled configuration:
.venv/bin/python -m rhythmlens.noise_phase_a --config configs/noise-phase-a-v1.yaml
```

`configs/noise-phase-a-v1.yaml` is a strict, versioned execution/input contract with
literal `execution_authorized: false`. Only its authorization boolean can vary
without a new versioned implementation review: the parser pins the canonical hash
of all other fields, rejects duplicate keys and rejects unknown schemas/fields.
Both original proposal YAMLs remain byte-identical, unsupported and false. Their
null placeholders are historical proposal state, not inputs silently populated at
execution. Existing analysis authorization does not authorize this runner.

Real execution requires a separately approved, committed change of the **new**
contract's authorization to true, a clean tracked configuration/source tree, and
an explicit `--authorize-execution` runtime flag. Checks occur before any signal,
historical prediction export or checkpoint hash/read/deserialization. Disabled
execution opens only the requested configuration; it does not inspect protected
artifacts, discover replacements or create a run directory. There is no bypass,
resume, overwrite or automatic retry CLI. Internal array/scoring functions accept
generated fixtures; they do not constitute real execution authority.

## Bound inputs and execution prerequisites

The separate contract binds exact bytes of both proposals, protocol, reviewed
NSTDB provenance/audit/roles/review/hash index, original split/source manifests,
training YAMLs, multiseed report, dependency locks and the historical export
identity registry. All nine pool identities are declared; only the three stress
pools are reconstructed/used. Their reconstructed descriptors must agree with the
reviewed JSON. Native noise voltage calibration remains unverified.

Checkpoint, preprocessing, resolved training-config, training-report and historical
CPU validation-export paths/hashes for seeds 42/43/44 are copied from existing
tracked proposal/report/analysis metadata. No real protected file was opened to
obtain or verify those identities during implementation. These declarations do
not establish that local runtime bytes still exist or match. Missing/mismatching
inputs block execution; never regenerate historical exports or fill missing hashes
by probing protected artifacts without separate authorization.

Every registry records the execution config's exact-byte and sorted compact-JSON
hashes, committed revision and tracked-file hashes, exact input byte identities,
invocation/time and pinned Python/NumPy/SciPy/Torch/WFDB/PyYAML environment. Lock
hashes bind other dependencies. Audit `provenance_sha256` means canonical JSON;
role `audit_sha256` means file bytes. These are verified distinctly. Local-source
index consistency remains distinct from independent index authentication.

Before any real execution, the owner must authorize the bounded run and its
compute/time budget. Verify the reviewed input contract, matching CPU environment,
all required original artifacts, unchanged grouping/splits/units and an unoccupied
fixed output root. Any newly verified cross-boundary linkage or result-affecting
protocol ambiguity requires review rather than an invented numerical choice.
Archival destination/restoration remains unspecified; no upload is implemented.

## Numerical and ordering contracts

All three original temporal candidates, seeds 42/43/44 and epochs 31/42/39, are
required. Validation-only eligibility is fixed at 1280, N=762/A=114/O=362/~=42.
The unchanged `FrozenRecords` verifies original grouping/source/split identities;
no patient identities are invented. Physical decoding stays at native 300 Hz/mV.

Before model access, metadata and all required bytes are checked and deterministic
case plans are frozen. Plans use the existing stress primitive and exclude model
seed and severity from the key. A clean control plus bw/ma/em × six SNRs
24/18/12/6/0/-6 × three repeats gives 55 conditions. Both noise channels remain
eligible. No development-reserve or augmentation-train samples are used.
Corruption is float64 physical mV before unchanged saved preprocessing; plan/pool
hashes, exact slice length and achieved SNR are checked by existing primitives.
Flat reference cases remain eligible with zero added noise, null achieved SNR and
explicit accounting. Invalid/zero-RMS slices fail without redraw or substitution.

Inference uses CPU float32, two threads, deterministic algorithms, eval/inference
mode, the original `LengthBucketSampler(..., batch_size=16, shuffle=False)` and
`collate_records` padded-prefix masks. The leading remainder and stable global
length sort are retained. Logits are restored to frozen row order. Raw softmax
T=1, float64 scoring and first-in-N/A/O/~ argmax ties are unchanged.

All three clean runs pass the historical reproduction gate before any stress
condition. Historical ID/label/prediction/confusion identities must agree exactly;
logits/probabilities use atol=1e-7, rtol=1e-5, probability sums atol=1e-6. Historical
softmax/argmax consistency is also checked. Missing exports or mismatches stop the
run; no historical output is generated or tolerance relaxed.

## Lifecycle, failures and artifacts

The fixed ignored root is
`outputs/noise-robustness/noise-robustness-v1/phase-a/`. Any existing directory or
symlink blocks work. Exclusive creation precedes protected runtime byte validation
and backend signal access. Failures after that point preserve status, registry,
accounting and any completed plans/batches; they are never reused automatically.
Failures before creation, including authorization/metadata/collision rejection,
create no run artifacts.

- `registry.json`: declared inputs, source/config/environment identities.
- `plans.json`: every eligible ID/label/group/length and shared deterministic plans.
- `accounting.json`: every seed × condition × eligible ID, with completed,
  technical-failure or explicit blocked/not-started status; denominators never shrink.
- `events.jsonl`: batch-start journal; interruption may leave a started batch.
- `seed*/<condition>/batch*.npy/json`: completed batch predictions, IDs/indices,
  physical/transformed digests, corruption/flat/RMS/SNR metadata and elapsed times.
- `seed*/<condition>/logits.npy`: complete float32 predictions in frozen row order.
- `failure-context.json`: active cases, available corruption metadata and error.
  Buffers not successfully written are not claimed retained. Input/model-stage
  failures may precede any completed signal batch; abrupt process kill or disk
  failure can leave partial writes. Inspect retained files; never assume completeness.
- `seal.json`: all 165 complete prediction arrays plus config/plan/accounting,
  journal and batch-evidence hashes. Every eligible case must be completed, every
  seed/condition present and frozen row identities unchanged before scoring.
- `bootstrap-draws.npy`, `bootstrap-statistics.npy`, `bootstrap.json`: shared group
  multiplicities, statistics, axis/group metadata and intervals.
- `metrics.json`, `artifacts-sha256.json`, `status.json`: final metrics and artifact
  identities. A successful run requires status `completed` and its bound artifact
  index; partial metric writes or failed status are not completed-run evidence.

All predictions seal and verify before scoring; inputs and seals verify again
before metric publication. No completed-run metrics are published on a technical
failure. A publication/storage failure can retain partial files with failed or
running lifecycle state. Arrays, case ledgers and bulk bootstrap files remain
ignored. Hashes and Git reports cannot restore these bytes; durable backup is not
implemented here.

## Scoring, aggregation and uncertainty

Existing metrics supply four-class confusion and 4×7 outcome matrices, complete
AF outcomes, assessed versus N/A/O rhythm coverage, reasoned zero deferrals,
probability metrics (stable NLL, sum-of-four-class Brier, 15-bin equal-width
reliability/ECE), and binary AF metrics on true N/A/O with original p_A and no
renormalization. Added outputs include TP/FP/FN, false AF on true N/O and its
specified denominators, true ~ predicted A, automatic error over all eligible,
zero review workload, and flat-SNR counts/outcomes separately from deferral.
AF sensitivity always uses all AF-labeled cases; no AF means null with a reason.
Zero-support class F1 contributes zero to macro F1 with an explicit flag.

Report each seed and repetition, equal-seed mean/sample SD, within-record mean
error/AF-detection indicators over repeats, and means of repetition metrics rather
than predictions or pooled confusion. Clean is referenced once. Primary paired
deltas subtract the same seed's clean result; no seed winner or ensemble.

Bootstrap uses 10,000 draws, PCG64 seed 20261016, N/A/O/~ strata, sorted homogeneous
duplicate groups, each class's original group count sampled with replacement.
Group multiplicities retain all members, conditions, repeats and seeds together.
Each draw recomputes repetition-specific confusion/metrics before equal averaging;
all comparisons share the draws. Absolute and paired-delta intervals for macro F1,
automatic AF sensitivity and automatic error use linear 2.5/97.5 percentiles.
Undefined AF intervals are null; no invalid draw is replaced. Small generated
fixtures reduce replicates inside tests only; the production path always uses
10,000. Local statistic arrays use NaN solely to encode undefined AF values;
JSON uses null plus reason.

All outcomes remain reused-validation development evidence. Unknown patients,
noise-source dependence, fixed model seeds/plans, historical selection and severe
corruption's unverified recognizability limit interpretation. No clinical,
quality-aware triage or new final-cohort claim follows from this implementation.

## External progress and resource supervision

The separate `scripts/supervise_phase_a.py` launches exactly one documented runner
and does not alter numerical code or the execution policy. It requires the
committed true authorization and a clean tree, absent fixed run/log destinations,
and at least 10 GiB free disk. Invocation, after authorization only:

```sh
.venv/bin/python -u -B scripts/supervise_phase_a.py
```

Its private process session and PID/start-time tracking include live descendants,
including observed descendants subsequently reparented. Resource polling is once
per second; progress refresh is every five seconds. RAM is aggregate resident size
(shared pages may be counted more than once). These are sampled observations, not
kernel memory/disk quotas or an assertion that subsecond peaks were measured.
At an observed aggregate RSS above 4 GiB, generated run/log output above 5 GiB,
or 7 h 55 min elapsed, it sends SIGINT to this run's identified processes. At
8 h it force-stops remaining identified processes and marks cleanup incomplete.
Unexpected monitor failures also interrupt/clean up the specific run; no unrelated
process is signaled. Completed files are retained without overwrite or retry.

The terminal bar counts successful batch-array/complete-batch-metadata pairs,
never started journal events, without parsing prediction arrays. It labels
preflight, clean reproduction, stress inference, sealing, indeterminate bootstrap/
scoring, completion or failure. Percentage and rate-derived ETA describe inference
only; the final inference batch does not establish whole-run completion. Completion
still requires independent lifecycle/seal/artifact-index verification.

Ignored `outputs/noise-phase-a-run-v1/` retains runner stdout/stderr, launch/budget
metadata, sampled progress/resource history and current `progress.json`. The
resource byte count includes both the fixed run root and these external logs.
The source tree must remain unchanged while the runner is active.
