# Dedicated rhythm-only internal evaluator verification

Date: 2026-10-14. Starting HEAD: `47d198e15f0fdcfbd2210ea9cf6700af7a1b3889`.
Real evaluation remains **pending and unauthorized**. No holdout signal, preview,
prediction or performance was loaded/generated. No production checkpoint was
deserialized, model trained, dependency installed, commit created or push made.
Checkpoint bytes were read only for opaque provenance hashing.

## Starting state and reuse

Read root `AGENTS.md`; root `ECG_Project_Context.md` was absent. Inspected Git
status/history, the integrity audit, final protocol, completed multiseed and
calibration workflow, and existing tests. The dedicated evaluator/configuration
were absent; the final protocol explicitly described them as unimplemented.
The completed calibration evaluator was validation-only and already synthetically
verified. It was retained and its ledger, probability metrics, reliability,
undefined-denominator and clean-source helpers reused. Prior uncommitted
calibration report/role metadata/index work was preserved. The existing analysis
authorization was not changed; it cannot authorize this executor.

## Implemented behavior

- Dedicated `rhythmlens.final_evaluation` module and `final-preflight` /
  `final-evaluate` launcher actions, with an analysis-independent configuration.
  Existing training and validation loaders retain their holdout restrictions.
- Literal false YAML authorization, existing clean committed-source requirement,
  and explicit runtime authorization before execution inputs/output creation.
  No new approval mechanism. The frozen configuration fingerprint protects all
  nonauthorization choices, including the single output path and three models.
- Fixed temporal seeds 42/43/44, selected epochs 31/42/39, original hashes and
  training-only preprocessing. Original numerical/decoder modules and both locks
  are unchanged. Calibration, confidence deferral and quality gating are disabled.
- Metadata-only preflight verifies original split/source/provenance identities,
  disjoint recording/duplicate/verified-subject boundaries, unknown-subject
  limitation, eligibility, training reports/configurations and opaque checkpoint
  hashes. Offline runtime checks verify the existing CPU environment; launcher
  evaluation actions cannot bootstrap/install PyYAML.
- Authorized execution exclusively creates the fixed ignored output root, records
  lifecycle, all eligible cases, per-seed accounting and reasons, invocation,
  environment, source revision and committed/input hashes. All signal-file hashes
  are checked before any prediction. Each seed reloads matching frozen weights and
  preprocessing, using canonical IDs, batch size 16, padded-prefix masks,
  float32 CPU inference and two threads.
- All three logits arrays are sealed before scoring. Reports include per-seed
  confusion/outcome matrices, AF accounting and automatic AF sensitivity over all
  labeled AF, coverage, explicit zero deferrals/reasons, predicted-~ distinction,
  fixed probability metrics and equally weighted means/sample SD. No seed winner
  or ensemble. Primary macro F1 uses the prespecified zero-support convention.
- Shared PCG64(20261014), 10,000 class-stratified duplicate-group bootstrap draws
  yield per-seed and equal-mean primary percentile intervals. Group manifests and
  replicate statistics are retained. Absent AF gives null sensitivity/interval;
  no patient-independent uncertainty claim is made.
- Failure or interruption preserves the partial directory, complete eligible
  accounting, unresolved/failed cases and reasons; prediction-stage failure
  blocks scoring for every seed. Unsealed cases remain unresolved even if
  inference finished before a persistence failure. No resume, overwrite,
  execution-ID bypass, dependency installation or automatic retry exists.
- Completion record and full artifact SHA256 index remain local and ignored.
  External archival is explicitly pending; no preservation destination invented.

## Verification

Commands used the already installed `.venv` without dependency installation:

```bash
.venv/bin/python -m pytest -q tests
.venv/bin/python -m ruff check src/rhythmlens/final_evaluation.py scripts/manage.py tests/test_final_evaluation.py tests/test_calibration.py
.venv/bin/python scripts/check_repository_contracts.py
.venv/bin/python -m rhythmlens.final_evaluation preflight --config configs/temporal-final-evaluation-v1.yaml
.venv/bin/python scripts/manage.py final-evaluate --config configs/temporal-final-evaluation-v1.yaml --authorize-evaluation --dry-run
git diff --check
```

Full regression: **273 passed**. Dedicated evaluator: **13 synthetic checks**
within that suite. Ruff, repository contracts and whitespace checks passed.
The synthetic end-to-end case generates ECG in pytest temporary directories
(generator 2017, split seed 42), creates freshly initialized temporal checkpoints
(model initialization seed 20261014), fits preprocessing on synthetic train only,
then exercises real decoder/checkpoint reload/transform/inference/scoring and
all 10,000 bootstrap replicates. It does not train a model or load real ECG.
External authority/runtime/production metadata boundaries are substituted only
in this temporary synthetic fixture and are separately checked.

Checks cover literal authorization and frozen-choice rejection, clean committed
configuration, no unauthorized input/output access, tie order, assessed noisy
predictions, AF denominators and undefined AF, shared reproducible group draws,
conflicting group labels/nonfinite logits, all-three-seed sealing, failure at
each seed, missing input before prediction, persistence failure, complete case
accounting, artifact hashes and permanent collision rejection.

One existing calibration regression assumed the repository's configuration was
still unauthorized, although the completed reviewed-v1 authorization commit made
it true. Its false-authorization test now explicitly writes false to its synthetic
fixture. The real calibration configuration and behavior are unchanged.

Metadata-only real provenance preflight passed for 1,279 records, class supports
761/114/362/42; no raw signal files or prediction exports were read. Runtime:
Python 3.11.16, Torch 2.8.0+cpu, NumPy 2.2.6, CPU lock hash
`71231d200792985f74f8a2c9ebff633621ad86d6efcc59b2b0063b5a2301c86d`.
A deliberate unauthorized evaluation invocation (including the runtime flag)
rejected `execution_authorized: false` before accessing inputs. All protected
numerical/decoder/lock hashes match the protocol. The real evaluation output root
remains absent. Metadata preflight does not establish raw signal-file readiness;
those opaque hashes are deferred to authorized execution.

## Scientific scope and blockers

This is a **qualified internal evaluation of the frozen rhythm-only baseline**,
not final validation of the eventual quality-aware system. The original audit
remains authoritative: all eventual holdout signals were decoded before split
freeze, A00004/A00022 had pre-freeze previews generated, labels and integrity
summaries were available, and complete access history is unavailable. Recording
and exact-duplicate independence are verified; patient/source/time independence
and absence of near duplicates are not. Validation informed model/checkpoint
selection and exploratory outcomes were observed. No exposure was removed or
reinterpreted as blindness.

Inspecting these outcomes **consumes this holdout for future development**.
Subsequent quality-gating changes require separate evaluation evidence.
Real results are pending. Real execution is currently blocked intentionally by
false authorization and the uncommitted working tree. No missing checkpoint,
metadata or installed-runtime blocker was found. Raw signal readiness has not
been checked in this task. External archival remains unavailable, preventing
claims of complete external preservation; independent evaluation data remain
necessary for stronger scientific claims.

## Exact review/commit file set

Commit these reviewed implementation/docs and preserved evidence files with
`execution_authorized: false`; no commit is performed in this task:

```text
README.md
configs/temporal-final-evaluation-v1.yaml
data/manifests/temporal-calibration-deferral-v1/roles.csv
data/manifests/temporal-calibration-deferral-v1/roles.provenance.json
docs/README.md
docs/experiments/temporal-final-evaluation-v1.md
reports/README.md
reports/final-evaluation-integrity-audit-v1.md
reports/temporal-calibration-deferral-v1.md
reports/temporal-final-evaluation-v1-verification.md
scripts/manage.py
src/rhythmlens/final_evaluation.py
tests/test_calibration.py
tests/test_final_evaluation.py
```

The two role files and completed calibration report are prior uncommitted evidence,
not regenerated evaluator outputs. Do not add raw ECG, checkpoints or local output
artifacts. A clean committed tree is necessary for later authorized execution.

Exact **future**, separately authorized one-line diff in
`configs/temporal-final-evaluation-v1.yaml` (not applied here):

```diff
-execution_authorized: false
+execution_authorized: true
```

After that authorization change is explicitly approved and committed, the existing
one-time commands are:

```bash
.venv/bin/python scripts/manage.py final-preflight --config configs/temporal-final-evaluation-v1.yaml
.venv/bin/python scripts/manage.py final-evaluate --config configs/temporal-final-evaluation-v1.yaml --authorize-evaluation
```

Output: `outputs/final-evaluation/temporal-raw-v1/internal-holdout-v1/`.
Preserve any failed directory; a collision is not permission to retry or rename.
See the [protocol](../docs/experiments/temporal-final-evaluation-v1.md) for all fixed
estimands, uncertainty, exposure and reporting rules.

Next development milestone: reproducible train/validation-only noise perturbation
experiments and a separate signal-quality assessment with explicit quality labels.
Select gates/thresholds on development evidence, freeze them, and evaluate the
quality-aware system on separately sealed evidence. The ~ rhythm class is not a
signal-quality ground truth.
