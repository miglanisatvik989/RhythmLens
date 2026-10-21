# Results and verification evidence

## Recorded seed42 development runs

The linked reports are authoritative observed results for the first comparison,
using the same frozen split and training preprocessing.

| Metric | Reference CNN | Temporal candidate |
| --- | --- | --- |
| Validation records | 1,280 | 1,280 |
| Accuracy | 0.6734375 | 0.859375 |
| Four-class macro F1 | 0.5660961 | 0.8142109 |
| AF correct / all AF | 37 / 114 | 88 / 114 |
| AF recall | 0.3245614 | 0.7719298 |
| Best epoch | 23 | 31 |

- [Reference report](baselines/reference-seed42.json)
- [Temporal report](baselines/temporal-seed42.json)

These are historical selected-checkpoint validation results from one seed, not
independent holdout evidence or clinical validation. Architectures differ in
multiple properties. Those runs did not access holdout signals. Later calibration
and internal rhythm-only evaluation milestones are documented below; quality
gating remains pending.

The [completed multiseed report](rhythm-multiseed-v1.md) and
[full-precision summary](rhythm-multiseed-v1.json) record all six reference/temporal
seed42/43/44 runs under the [fixed protocol](../docs/experiments/rhythm-multiseed-v1.md).
The table above remains the original seed42 evidence. All reported performance
is checkpoint-selected validation evidence; the separate completed internal
rhythm-only evaluation is reported below.

## Calibration and confidence deferral

The [completed reviewed-v1 report](temporal-calibration-deferral-v1.md) records all
three seeds and arms, role counts, frozen policies, paired comparisons, artifact
identities and preservation status. The [frozen development protocol](../docs/experiments/temporal-calibration-deferral-v1.md)
and [historical implementation verification](temporal-calibration-deferral-v1-verification.md)
describe the logits-only workflow and its synthetic checks. Execution was explicitly
authorized at revision `47d198e15f0fdcfbd2210ea9cf6700af7a1b3889`.

Calibration improvements were mixed. Deferral increased accepted-case F1 on a
different subset but reduced automatic AF sensitivity; this does not establish
overall improvement. Results are exploratory because validation already informed
checkpoint and architecture selection. No policy retuning or deployment
recommendation follows from these outcomes. Signal-quality gating remains
unavailable. That calibration analysis excluded holdout; the separately authorized
internal rhythm-only evaluation is reported below.

Role metadata are trackable at the protocol-specified paths. Complete outputs
remain ignored and local: no repository archival mechanism was found, and a Git
commit of the report/role metadata would not preserve those output bytes.

## Completed internal rhythm-only evaluation

The [completed report](temporal-internal-rhythm-evaluation-v1.md) records execution
revision `bbdf121a81557bb4e614a452a1ffb42ba6a9b27d`, all three fixed temporal seeds,
1,279 cases per seed (N=761, A=114, O=362, ~=42), confusion matrices, complete AF
accounting, probability metrics, uncertainty and verified artifact hashes.
Mean automatic AF sensitivity was **80.70%**, automatic four-class error **14.33%**,
and assessed coverage **100%**, with zero deferrals and technical failures.
Predicted noisy outcomes remain assessed four-class cases and are reported
separately from rhythm outputs and deferrals. No seed winner was selected.

The [integrity audit](final-evaluation-integrity-audit-v1.md) documents pre-freeze
signal decoding and two previewed holdout IDs, public labels, unavailable complete
access history, and unverified patient/source/time independence. The
[frozen protocol](../docs/experiments/temporal-final-evaluation-v1.md) and
[implementation verification](temporal-final-evaluation-v1-verification.md)
retain their earlier milestone wording and exact experiment identities.
This is qualified internal rhythm-only evidence, not validation of quality-aware
triage or clinical generalization. **The holdout is consumed for future
development**; subsequent quality-gating changes require separate evaluation
evidence. Bootstrap intervals condition on the observed class mix and three fixed
checkpoints; unknown patient clusters and source shifts remain unaccounted for.

Complete ignored outputs remain local and still need external archival.
A Git report is not an artifact backup; recorded hashes cannot restore lost bytes.
No bulk run outputs, predictions, ECG or weights are published in Git.

## Next milestone: dataset readiness and noise proposal

The [readiness report](dataset-readiness-noise-quality-v1.md) identifies NSTDB
1.0.0 noise-only records and BUT QDB 1.0.0 quality data from primary documentation.
Its acquisition/audit findings describe that earlier milestone; subsequent NSTDB
readiness is recorded below. BUT acquisition remains pending. Rhythm labels,
quality annotations and synthetic corruption metadata remain separate; no
cross-dataset quality transfer is claimed.

The [frozen noise proposal](../docs/experiments/noise-robustness-v1.md) specifies
validation-only stress testing of all three original temporal checkpoints and a
paired comparison with three noise-augmented training runs. Its two new YAMLs
are declarative, unsupported and `execution_authorized: false`. No training,
inference or new results exist. Quality models, confidence-policy retuning and
reviewer UI remain outside this PR. The consumed holdout remains excluded, and
complete local artifacts still need an owner-specified durable backup destination.

The subsequent [NSTDB implementation verification](nstdb-adapter-v1-verification.md)
records acquisition/audit support and a synthetic-tested noise adapter. Real
acquisition/audit were pending at that milestone. The later real-data record below
supplies the NSTDB outcome; Phase A/B execution and results remain pending, and
proposal authorizations are unchanged. See [supported commands and boundaries](../docs/nstdb-adapter.md).

## Dated verification milestones

The [real NSTDB acquisition attempt](nstdb-real-data-readiness-v1.md) completed
nine transfers but failed publisher-index parsing before publication; no real
signal audit or noise-role manifest resulted. Its original failure findings are
preserved. The [publisher-index repair](nstdb-publisher-index-repair-v1.md) accepts
the observed delimiter and safe nested metadata, checks metadata before signal
I/O, and records dated synthetic verification. That repair alone did not establish
real readiness.

The subsequent [local-import readiness record](nstdb-real-data-readiness-v2.md)
preserves the repaired adapter's network timeout and records the owner's separately
authorized manual-source import, passing strict audit and all nine reviewed role
descriptors. Compact provenance/audit/role identities are tracked; original noise,
logs and bulky artifacts remain local. Native voltage calibration and durable
archival remain unverified. Reviewed descriptors are not frozen executable inputs;
no Phase A/B implementation or model experiment was performed at that milestone.

The subsequent [Phase A implementation verification](noise-phase-a-v1-verification.md)
records the separate disabled execution contract, deterministic stress runner,
baseline reproduction gate, complete accounting, sealed scoring and grouped
bootstrap. All verification uses generated fixtures and mocked protected I/O;
no real Phase A result exists. See the [supported interface](../docs/noise-phase-a.md).

The subsequent [completed CPU Phase A report](noise-phase-a-v1-results.md) and
[compact aggregate evidence](noise-phase-a-v1-results.json) record all three clean
reproduction gates and the full 55-condition stress run, sealed predictions,
complete case accounting, paired uncertainty and resource observations. These are
validation-reused development results, with no deferrals or technical failures.
The retained new true authorization records that one attempt only; it does not
authorize another run. Both original proposals remain false, and Phase B is pending.

| Report | Evidence |
| --- | --- |
| [Completed internal rhythm-only evaluation](temporal-internal-rhythm-evaluation-v1.md) | Actual three-seed results, uncertainty and artifact integrity |
| [Internal evaluator verification](temporal-final-evaluation-v1-verification.md) | Historical dedicated executor and 273-test synthetic/regression verification |
| [Environment setup](environment-setup.md) | Initial installation and dependencies |
| [Data audit](challenge2017-audit.md) | Integrity, labels, duplicates, limitations |
| [Audit verification](challenge2017-verification.md) | Acquisition and frozen splits |
| [Baseline implementation](rhythm-baseline-implementation.md) | Initial implementation checks |
| [Configuration workflow](configuration-workflow.md) | Earlier launcher/configuration delivery |
| [Pretraining readiness](pretraining-readiness.md) | Hardening, packaging, synthetic CPU/CUDA checks |
| [Repository simplification](repository-simplification.md) | Workflow fixes and behavior-preservation evidence |

Dated reports describe their milestone, not the current quickstart. Their contents
and paths are preserved. Follow [current documentation](../docs/README.md).

Phase B implementation is recorded in [its synthetic verification record](noise-phase-b-v1-verification.md). Its separate execution flag is false; no real augmented training or comparison result exists.
