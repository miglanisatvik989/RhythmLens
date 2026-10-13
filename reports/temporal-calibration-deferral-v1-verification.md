# Temporal calibration/deferral implementation verification

Status: implementation and synthetic verification complete; real calibration,
threshold selection, role assignment and dev-assess outcomes pending. This report
contains verification evidence, not new model performance results.

The [protocol](../docs/experiments/temporal-calibration-deferral-v1.md) and
[analysis YAML](../configs/temporal-calibration-deferral-v1.yaml) define the methods
and exact later commands. Read-only preflight authenticated each original CPU
evaluation report against `reports/rhythm-multiseed-v1.json`, resolved its companion
prediction export and verified logits, frozen IDs/labels, exact baseline confusion
and supports, float32/float64 softmax agreement and baseline NLL tolerance.
All three seeds passed: 1,280 cases each, N=762, A=114, O=362, ~=42.

Prediction hashes below were measured during this preflight, not taken from a
historical prediction-hash registry (none existed). They are now pinned in YAML.

| Seed | Saved CPU prediction CSV SHA256 |
| --- | --- |
| 42 | `e11ad0f1bcac963f82f2143535b57f44c11fd323790333430875f20c06767d78` |
| 43 | `7dc18886c41d6cde5c1e43a70ec96534c09bbee9193332850d843d983e30ae0a` |
| 44 | `c48b6f5fc05a49b78f83f85dcd02c73199b82dc765833dddfacf3925cd45886c` |

Source manifests/audit/provenance and frozen split metadata hashes also passed.
Original train/validation/holdout metadata had no overlapping IDs or duplicate
groups; no holdout signals were read. Subject identifiers remain unknown.
No real role manifest or analysis run directory was created.

Verification commands executed:

```bash
.venv/bin/python scripts/manage.py analysis-preflight --config configs/temporal-calibration-deferral-v1.yaml
.venv/bin/python -m pytest -q tests
.venv/bin/ruff check src/rhythmlens/calibration.py tests/test_calibration.py scripts/manage.py
.venv/bin/ruff format --check src/rhythmlens/calibration.py tests/test_calibration.py scripts/manage.py
.venv/bin/python scripts/verify_frozen.py
.venv/bin/python scripts/check_repository_contracts.py
```

Full suite: **216 passed**, including **39 new synthetic analysis tests**.
Lint and formatting checks passed. Frozen verification preserved 17 snapshot
files and original 5,969/1,280/1,279 partition counts; repository ignore/lock
contracts passed. Model, training and preprocessing numerical source, original
configurations, dependency descriptors, locks, frozen splits, reports and local
model artifacts were not modified. The existing launcher gained analysis actions;
its prior training configuration loader/routing remains covered by existing tests.

Synthetic tests cover indivisible groups, assignment determinism and invalid
metadata, deterministic convex temperature fitting, nonconvergence, argmax
preservation, independently enumerated threshold selection, undefined metrics,
AF/noisy semantics, bin boundaries, complete case/AF accounting, read-only
preflight, provenance failures, all-seed assessment gating, failed-fit recovery
boundaries, exclusive writes and run-directory overwrite protection. Metrics are
recomputed from persisted synthetic ledgers and checked against independent
case counts and probability-score calculations. Synthetic data are constructed
in temporary directories; no data files or fixture allowlist changes enter Git.

Real runs remain blocked pending review, explicit authorization and commitment
of this protocol/YAML/implementation with a clean source tree. No commit or push
was performed. Signal-quality gating is unavailable and disabled. Additional
verified subject/source/time/related-record metadata would require grouping review.
Patient independence and an independent development assessment are not established:
validation already informed architecture and checkpoint selection. Approximate
assessment supports (34 AF, 13 noisy) remain planning estimates, not frozen counts.

Verification runtime: Python 3.11.16, NumPy 2.2.6, PyYAML 6.0.3, pytest 8.4.2.

Implementation/configuration identities at verification:

| File | SHA256 |
| --- | --- |
| `src/rhythmlens/calibration.py` | `39b093722f037a1936f4b525ddc390f4e75da5d4a504c4a90beabbbb466999e2` |
| `scripts/manage.py` | `c12622bd9c9baa08d1f5e8751723eaafdd98fc4edd498c55c1ca7f1bee92d179` |
| `tests/test_calibration.py` | `45734c6c074368ae6ca254679935ab2f60404f727b4f72d68ff94efeb6d33cef` |
| `configs/temporal-calibration-deferral-v1.yaml` | `ed7d73f0d970f3efda78272994a176b0c7e80f9096ddc7608b62cc1f5c8b87ec` |


## Authorization hardening verification — 2026-10-13

This section records checks run after authorization hardening. The earlier
216-test result, real-input preflight evidence and hash table above remain the
historical verification record; real-input preflight was not rerun for this change.

Fitting and assessment now require both YAML `execution_authorized` to be the
literal boolean true (`is True`) and explicit CLI authorization. False, absent,
null, numeric, string, list and mapping values fail before Git checks, analysis
input loading, role assignment, fitting, threshold selection or output writes.
The CLI cannot override a false or invalid YAML value. Read-only preflight remains
available with false. The real configuration remains false and unchanged.

The existing synthetic success tests now authorize only their temporary YAML and
pass the explicit CLI authorization through the same gate used in production.
Only Git/environment checks are mocked for those temporary repositories;
authorization itself is not bypassed. Additional tests prove dirty source still
blocks an otherwise authorized run.

Checks executed for this hardening:

```bash
.venv/bin/python -m pytest -q tests/test_calibration.py
.venv/bin/python -m pytest -q tests
.venv/bin/ruff check src/rhythmlens/calibration.py tests/test_calibration.py scripts/manage.py
.venv/bin/ruff format --check src/rhythmlens/calibration.py tests/test_calibration.py scripts/manage.py
.venv/bin/python scripts/verify_frozen.py
.venv/bin/python scripts/check_repository_contracts.py
git diff --check
```

Results: **83 calibration tests passed**, including 44 new authorization tests;
**260 tests passed** in the full suite. Lint, formatting and whitespace checks
passed. Frozen verification reported 17 unchanged files, original partition
counts 5,969/1,280/1,279, and zero ECG files opened. Ignore/lock repository
contracts passed. Protected numerical source, locks, frozen splits, original
training configurations and multiseed reports have no tracked diff. Existing
uncommitted launcher and documentation-index changes were preserved.

No real role manifest or calibration run directory exists. No real fitting,
threshold selection, assessment, inference, holdout signal access, training,
dependency installation, commit or push was performed. All real analysis results
remain pending. Next approval: review and commit the hardening/protocol with the
real YAML still false. Real execution requires a separate explicit approval to
change authorization to true and freeze the approved configuration before running.

Current hardening identities (distinct from the historical table above):

| File | SHA256 |
| --- | --- |
| `src/rhythmlens/calibration.py` | `d7bbd906ba2d810ddaaf6da6e3d328fc6fb50495c885c3a62a45368dadbf0472` |
| `tests/test_calibration.py` | `a10c325b16daca4879100091a949f4dcf7cbdfecff0896a610c979d88969cac8` |
| `configs/temporal-calibration-deferral-v1.yaml` | `ed7d73f0d970f3efda78272994a176b0c7e80f9096ddc7608b62cc1f5c8b87ec` |
