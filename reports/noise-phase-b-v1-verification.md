# Phase B v1 implementation verification

Recorded 2026-10-21. This milestone implements the already frozen Phase B method
and verifies generated fixtures/mocked model interfaces only. **No real signal,
checkpoint, prediction array, GPU benchmark, training, inference or bootstrap run
was accessed/executed. Real CUDA correctness and augmentation benefit are untested.**

Starting point: `feat/noise-phase-a` at
`3cae53c295dd094361127222553b6ac97eca8483`, clean and synchronized to its upstream.
PR #9 was open/draft, targeting `feat/nstdb-adapter`. New branch
`feat/noise-phase-b` was created from that exact head; no reset/rebase or historical
artifact recreation occurred. Its draft PR targets `feat/noise-phase-a` while #9
remains unmerged.

## Implemented behavior

- A separate strict disabled execution contract and CLI with explicit `train` and
  `evaluate` stages. Literal flag plus explicit runtime authority and committed,
  clean source are required before protected I/O. Both original proposals, the
  historical Phase A configuration/true one-attempt flag and results are unchanged.
- Original seed42/43/44 CUDA training settings, architecture, dropout, AdamW,
  sampler and clean-validation strict macro-F1 selection. The original saved
  training-only preprocessing state is reused without refitting. The new experiment
  is `temporal-noise-v1`; its files stay in the distinct fixed Phase B root.
- Existing exact per-record/epoch PCG64 plans and physical corruption primitive,
  p=0.5 and 24/18/12/6 dB training grid. Clean physical caching, epoch-specific
  transforms, RNG isolation, immutable role descriptors/hashes, disjoint train/stress
  support, flat-SNR handling and no redraw/substitution are preserved.
- CPU evaluation of all three frozen augmented models on the existing Phase A maps.
  Original predictions are reused only with completed lifecycle, full index/byte
  and gate/batch/accounting seals, exact baseline source/input contract (including unchanged existing source modules)
  and matching
  physical/transformed corruption evidence. There is no automatic baseline rerun.
- Both arms' complete six-candidate evidence is verified/compositely sealed before
  comparative scoring. Full eligible/AF/outcome/coverage/deferral/probability and
  repetition metrics are reused. Matching-seed arm changes and shared grouped
  bootstrap retain both arms, all seeds/conditions/repetitions jointly.
- Exclusive stage claims/destinations, preserved failed/interrupted evidence and
  batch-publication progress with measured stage-local ETA. Clean validation,
  clean/stress evaluation and indeterminate scoring have distinct labels. Training
  completion and evaluation completion are separate; no resume/retry is offered.

See [the supported interface](../docs/noise-phase-b.md) for stage environments,
artifacts, publication order, limitations and remaining execution prerequisites.

## Declaration and preservation review

The configuration declares inputs from tracked Phase A configuration/results,
original training/proposal/report metadata and reviewed noise manifests only.
Baseline status/index/seal/plans/registry/accounting/metrics/bootstrap identities
were copied from the aggregate report, without inspecting their local bytes.
Original candidate/preprocessing/export identities were inherited from the prior
tracked contract. No missing identity was invented or filled from protected files.

Every original tracked file at the starting head was compared byte-for-byte.
Only `docs/README.md` and `reports/README.md` navigation indexes changed; all
preexisting implementation, numerical protocol, authorization flags, source locks,
splits, reviewed noise evidence and historical result files remain unchanged.
All new source/config/tests/documentation are separate files. The real fixed
Phase B output root remains absent. Outputs, signals, weights, predictions and
bulk bootstrap/ledger artifacts remain ignored. No dependency changes, download,
archive upload or hardware experiment occurred.

The contract's canonical sorted compact JSON SHA256 excluding only its new
boolean authorization field is
`71bb8347193a0f9e67bdf34686c0c48e42c5fd4525b99a83068c2088d1401b82`.
This differs from a YAML exact-byte identity, which each future run also records.
Audit canonical JSON bindings and exact byte hashes remain distinct.

## Actual verification

Existing environment/dependencies only:

- Focused generated verification initially passed 31 tests in 24.26 s. The final
  suite also includes the later dangling-symlink collision regression.
- Full synthetic/regression suite initially passed 497 tests in 50.78 s.
  Final full-suite result after the collision fix: **498 passed in 50.00 s**.
- Ruff lint passed; format check passed (42 Python files).
- Repository contracts passed: intended manifests/configs/reports trackable,
  raw data/checkpoints/outputs ignored, CPU and CUDA locks portable/credential-free.
- Frozen verification passed: 17 identities unchanged, train/validation/holdout
  metadata counts 5,969/1,280/1,279, **zero ECG files opened**.
- `uv pip check` passed for 58 installed distributions. Both offline lock checks
  passed (60 CPU / 74 CUDA packages). No sync/install/upgrade was performed.
- sdist and wheel built successfully without isolation in
  `/tmp/rhythmlens-phase-b-build`; wheel inventory included all three new modules.
- Disabled real-configuration CLI rejected before metadata/protected I/O.
- Local documentation links, explicit diff/preservation review, ignored-output and
  whitespace checks passed. Final push/PR CI results are recorded in the draft PR.

Generated tests cover both schemas and authorization-before-I/O, duplicate keys,
metadata declarations, independent hand-reconstructed training draws, global
Python/NumPy/Torch RNG isolation, per-epoch variation, train/stress role separation,
physical-before-transform order, clean-only cache and saved preprocessing,
flat cases and invalid-pool failure without redraw, strict clean-selection ties/
patience, all three original numerical training profiles wired through mocked
model/optimizer/runtime interfaces, original sampler/generator settings and safe
checkpoint metadata. No real model is trained by these tests.

Synthetic lifecycle tests exercise failed optimizer preservation, completed epoch
accounting, collisions/interruptions, measured ETA and indeterminate scoring,
complete two-stage publication, baseline full-index reuse and tamper rejection,
seed pairing, incompatible maps, mismatching transformed input evidence, missing
candidates/ledgers, failed batches, six-candidate seals before scoring, unchanged
baseline files, nonzero paired effects, shared duplicate-group draws, no-AF null
intervals and final artifact-index verification. Generated bootstrap tests use
small replicate counts; the production default remains 10,000.

An initial test attempted to mutate the existing read-only synthetic pool; its
fixture was corrected to construct a replaced invalid pool instead. Final review
also found that resolving a dangling symlink could hide an output collision; the
new runner now rejects the lexical symlink before resolution, with regression
coverage. Neither finding involved protected data or any real experiment.

## Limits and remaining prerequisites

The available CPU environment supports synthetic verification. Actual matching
CUDA hardware/driver/runtime behavior, local real input bytes, checkpoint loading,
training convergence, real baseline-reuse compatibility and augmentation benefit
remain unverified. This implementation does not supply a resource/deadline
supervisor or approve a budget. The Phase A supervisor cannot launch Phase B.

Before execution, separately authorize a read-only readiness audit of the declared
artifact hashes, environment/hardware metadata, destination absence and resource
capacity. Owner approval must then provide a concrete CUDA-training and
CPU-evaluation wall/RAM/GPU/output budget with external interruption/cleanup
supervision and a focused committed change of only the new Phase B flag. Both
original proposals must stay false; the Phase A flag remains historical.

Unknown patients/noise volunteers, shared source recordings, reused validation
selection, fixed seeds/plans and historical source/runtime differences limit any
future results. Native noise voltage calibration and severe-corruption diagnostic
recognizability remain unverified. Durable archival/restore awaits an approved
destination. Quality models, BUT, calibration, consumed holdout, UI and a new
final cohort stay excluded.
