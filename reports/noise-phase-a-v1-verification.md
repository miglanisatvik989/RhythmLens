# Phase A noise-robustness implementation verification v1

Date: 2026-10-21 (Asia/Kolkata). **Implementation and generated synthetic verification
only. No real Phase A execution or result exists.**

## Starting state and scope

Verified clean `feat/nstdb-adapter` at
`372592c250ad771c31d4687bf66f1aa36e306204`, matching fetched upstream. PR #8 was
open/draft, unmerged, based on `feat/noise-robustness`. The requested new branch
`feat/noise-phase-a` did not exist; it was created from that exact head. This PR
is stacked on `feat/nstdb-adapter` while #8 remains unmerged. No reset, rebase,
merge, force push, branch deletion or dependency installation occurred.

Read AGENTS.md, both frozen proposals, the protocol, real NSTDB readiness evidence,
and relevant decoding/preprocessing/data/batching/inference/noise/evaluation/
lifecycle/test code. A preservation snapshot covers all 117 preexisting tracked
files. Only current documentation indexes and adapter documentation are updated;
all preexisting source, numerical functions, locks, decoder contracts, splits,
protocols, proposal YAMLs, reviewed NSTDB evidence and historical results remain
unchanged. Prior diagnostics and raw artifacts were not modified.

## Delivered behavior

The [dedicated interface](../docs/noise-phase-a.md) uses a separate strict
`rhythmlens-noise-phase-a-v1` contract, `configs/noise-phase-a-v1.yaml`, disabled by
default. Its nonauthorization canonical policy digest is
`32222409868f6b055baa3fc262bc90f2129c8b2910747338ca3e2a52f25f22ea`.
Unknown/duplicate fields, malformed authorization types and changed numerical or
input contracts fail closed. Both frozen proposal schemas remain unsupported and
both flags stay false. There is no new Phase A action in scripts/manage.py.

Execution requires literal true in the new contract, explicit runtime authority,
and a clean committed configuration/source tree before any protected artifact
hash/read/deserialization. Disabled rejection was exercised with the real disabled
configuration: no metadata/protected I/O or output creation followed. No existing
analysis authorization carries into this runner.

Input declarations bind the reviewed provenance/audit/role/pool identities,
original split/source/preprocessing/candidate identities, historical CPU validation
export identities, protocol, locks and exact environment. These were taken from
tracked metadata only. No protected real artifact was opened to discover, verify,
replace or populate an identity. The byte versus canonical-JSON distinctions are
explicit. Runtime revalidation still must establish local artifact availability
and matching bytes after separate authorization.

The runner reuses original primitives: validation-only frozen data loading,
physical-mV decoding, stress-role plans/corruption, unchanged saved transform,
original no-shuffle length bucketing and prefix masks. It requires all three
original candidates and all 55 conditions, shares noise plans across severity and
model seed, freezes row/group/length/condition/plan/role-bound metadata before model
access, and blocks stress until all three clean reproductions pass exact
identity/prediction/confusion and frozen numerical tolerances. Successful baseline
gates are recorded and sealed. No reserve/training role is used.

Exclusive lifecycle creation precedes protected runtime validation/backend I/O.
Every eligible case has a seed/condition accounting entry; failures preserve
partial batches, active/blocked reasons, available metadata and status without
shrinking denominators, retrying, redrawing or substituting. Flat-reference cases
remain eligible, unchanged, with null SNR and separate counts/outcome matrices.
All 165 predictions, complete accounting, plans, gate and batch evidence must seal
and verify before scoring. Inputs/seals verify again before final metrics. Final
completed status binds an artifact hash index. Partial storage writes with failed/
running state are not completed-run metrics; buffers not written are not claimed
retained. There is no resume/retry/archive subsystem.

Scoring includes prescribed confusion/outcome matrices, coverage, full AF and
noisy-class accounting, TP/FP/FN, false AF denominators, probability/reliability
metrics, zero deferrals/workload, equal seed summaries, repetition-specific and
mean-repetition metrics, within-record repetition indicators and paired primary
deltas. Undefined rates use null/reasons; no AF means no AF interval; unsupported
class F1 contributes zero to macro F1 with a flag. No variants/seeds are pooled as
independent people or combined into ensemble predictions.

The bootstrap retains homogeneous duplicate groups, uses shared class-stratified
PCG64 seed 20261016 draws, recomputes repetition-specific metrics and then averages
repeats/seeds. Absolute and paired-delta intervals use linear percentiles.
Production uses exactly 10,000 replicates; end-to-end fixtures use small explicit
counts through a mocked bootstrap call. Nontrivial generated group-size/draw tests
compare against a separate confusion-count oracle and verify shared multiplicities.

## Verification actually run

Existing environment: Python 3.11.16, NumPy 2.2.6, SciPy 1.15.3, Torch 2.8.0+cpu,
WFDB 4.3.1, PyYAML 6.0.3. No installation or upgrade.

| Check | Observed result |
| --- | --- |
| Dedicated Phase A generated-fixture tests | **50 passed in 12.15 s** |
| Final full synthetic/regression suite | **457 passed in 26.11 s** |
| Ruff lint and formatting | Passed; 36 files already formatted |
| Repository contracts | Passed: raw/output ignored; compact evidence trackable; portable credential-free locks |
| Frozen verification | 17 identities unchanged; counts 5969/1280/1279; zero ECG files opened (split metadata only) |
| Both offline lock checks | Passed, 60 and 74 packages |
| Installed dependency compatibility | Passed, 58 distributions |
| Existing-dependency sdist/wheel build, no isolation | Passed |
| Extracted wheel with existing venv, no installation | New modules, strict disabled config, decoder contract, main CLI help/version and Phase A help passed |
| Disabled real-config invocation | Rejected before metadata/protected I/O; actual output root absent |
| Preservation, links, stage and diff whitespace | Reviewed before commit; only explicit implementation/current documentation additions/updates allowed |

Tests use generated NumPy noise/ECG, synthetic preprocessing and mocked production
model/data interfaces in temporary directories. New tests guard repository raw and
historical rhythm artifacts against reads. They cover contract and manifest
mutations, authorization before protected I/O, plan pairing/RNG isolation,
physical-before-transform units/order, clean export mismatches, checkpoint
candidate metadata, original remainder/bucket order, prefix masks, complete
accounting, flat/invalid noise, failures before/after sealing, missing seals,
tampered rows/bytes, collisions, grouped shared draws, uncertainty and null AF.
Existing full-suite model operations also use generated temporary fixtures.
No real Challenge/NSTDB signal, prediction array, checkpoint, training, inference,
calibration, stress evaluation, consumed-holdout signal or BUT data was accessed.

An integration test exposed tuple-versus-JSON-list descriptor comparison; canonical
comparison fixes representation only, preserving resampling numerics. An initial
wheel smoke command resolved the venv interpreter symlink to its base interpreter
and could not import NumPy; rerunning with the existing venv executable passed.
No dependency change or protected artifact access was involved. Final results
above reflect the corrected invocations and final source.

## Implementation identities

These SHA256 values identify exact file bytes at verification:

| File | SHA256 |
| --- | --- |
| `configs/noise-phase-a-v1.yaml` | `2a42a4b939c58b0c1c183d07070e32581db21125b6876cc5e40670fc98999930` |
| `src/rhythmlens/noise_phase_a.py` | `1dadfa1cf631f98870caa74ac81beda006b3e1297c4af70690fd01429a6a18cd` |
| `src/rhythmlens/noise_phase_a_analysis.py` | `ea486ddbd9b12e014015cb7f6121548a6c396512a072e17d4b58c16863e52d37` |
| `tests/test_noise_phase_a.py` | `5e76d439c3676abc82ab3aab5933b01bd2aa30663d51c89b42455820e9b9b4dd` |

## Remaining execution prerequisites and limitations

No result-affecting protocol ambiguity was identified during implementation;
synthetic success does not prove real baseline reproduction or experiment readiness.
The real output root remains absent. Before any real Phase A run:

1. Review the stacked PR/implementation, disabled contract and immutable input
   declarations; obtain explicit bounded execution authorization and compute/time
   budget. Commit authorization only in the separate new execution contract;
   preserve both original proposals as unsupported/false.
2. Under that authorization, verify all original validation signal bytes, all three
   candidates/preprocessing/config/reports and historical CPU exports against their
   declared identities, reviewed noise evidence, unchanged splits/grouping and exact
   pinned CPU environment. Missing/mismatching inputs or new linkage block work;
   never regenerate exports, tune thresholds or relax tolerances.
3. Require the fixed absent output root and complete all-seed clean reproduction
   gates. Preserve any failure and stop without automatic retry. A clean gate cannot
   be claimed passed from this synthetic milestone.

No execution prerequisite was silently filled by reading protected artifacts.
Native noise voltage calibration, subject/source independence, clinical adequacy,
severe-corruption recognizability and a new independently sealed final cohort
remain unverified. Validation reuse, fixed seeds/plans and noise-source dependence
remain limitations. No Phase B, policy retuning, quality model, BUT or UI work was
implemented. Bulk outputs, predictions, case ledgers and bootstrap arrays stay
ignored/local. Durable archive destination/restoration remains owner-pending; no
upload occurred. Git metadata and hashes are not restorable backups.
