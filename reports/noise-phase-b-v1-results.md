# Phase B: completed bounded CUDA training and CPU comparison

Recorded 2026-10-22. This is reused-validation development evidence, not final-test or clinical validation. The earlier NSTDB failure/repair/readiness records, Phase A results and Phase B implementation verification remain unchanged.

## Outcome and execution identity

One attempt completed from committed revision `4cfc7602e36cffa05fffa1229ffd5f5699011044` on `feat/noise-phase-b`. All three augmented training candidates completed and passed full artifact verification before CPU evaluation. All 13,200 new inference batches and 165 new prediction arrays were sealed, physically paired with the 165 reused Phase A arrays and verified before comparative scoring. No retry, resume, replacement root, tolerance change, regenerated baseline, Phase A rerun or numerical repair occurred.

Training registry creation was 2026-10-22T06:04:27.637164+00:00; final completion was 2026-10-22T11:50:58.194406+00:00 (11:34:27–17:20:58 IST). External supervisor monotonic elapsed time was 17,480.532 seconds (4 h 51 min 20.532 s), including its prelaunch verification. Training stage monotonic elapsed time was 7,736.876 seconds (2 h 8 min 56.876 s); CPU evaluation was 9,736.026 seconds (2 h 42 min 16.026 s). Both processes exited 0; supervisor and runner lifecycles completed, with cleanup complete and no forced termination or reported resource/telemetry interruption.

The recorded UTC span was instead 20,790.557 seconds (5 h 46 min 30.557 s), about 55 min 10 s longer than supervisor monotonic elapsed time. Evaluation registry creation at 08:13:27.885236 UTC through completion spans 13,050.309 seconds (3 h 37 min 30.309 s), versus its 9,736.026-second monotonic stage measurement. The retained evidence does not establish the cause of the clock discrepancy. Supervisor deadlines use monotonic time and do not establish suspend-inclusive civil-clock enforcement. Both recorded UTC stage/overall spans also remained within the approved budgets. No clock adjustment or run artifact modification was made during review.

Authorization commit `4cfc760` changed only the new Phase B flag to true. Its policy canonical SHA256 excluding that flag remained `71bb8347193a0f9e67bdf34686c0c48e42c5fd4525b99a83068c2088d1401b82`. The canonical authorized configuration, including the flag, is instead `52e1efd23d08211a7fb4c3091517834900ea20bf9767dfdc466a6ea478cb0eb1`. Neither is an exact-byte file hash. Both original proposals remain false and byte-identical; the historical Phase A flag/results remain unchanged. The retained true Phase B flag records only this attempt and grants no further execution authority. Fixed run/supervisor roots are occupied.

## Inputs, environments and training review

Preflight verified 14,549 declared input files and the complete 26,577-file Phase A index, including unchanged baseline source modules. This covered 41,150 unique byte-hashed artifacts (1,018,781,763 bytes), without a preliminary model benchmark or protected-array parsing. Both runtime registries bind the same execution revision, committed source, exact configuration bytes and 14,549 input identities. All bound bytes were checked again at completion before subsequent documentation/test edits.

Python 3.11.16, NumPy 2.2.6, SciPy 1.15.3, WFDB 4.3.1 and PyYAML 6.0.3 were pinned in both stages. Training used the existing CUDA environment with Torch 2.8.0+cu128, GPU index 0 (NVIDIA GeForce RTX 5050 Laptop GPU; driver 610.74), two CPU threads and the original seed-specific CUDA profiles. Evaluation used the existing Torch 2.8.0+cpu environment, two CPU threads, deterministic float32 inference and empty CUDA visibility. Corruption/scoring retained float64. No dependencies were installed or upgraded. This successful attempt demonstrates execution on this setup, without establishing general GPU correctness or cross-device reproducibility.

Training used 5,969 original training recordings, original saved training-only preprocessing and isolated augmentation-train noise pools. All seeds preserved architecture, optimizer, dropout, sampler, epoch cap/patience and strict clean-validation macro-F1 selection. Validation alone selected checkpoints; stress metrics did not. Seeds 42/43 stopped after 33/42 epochs; seed 44 reached the prescribed 50-epoch maximum. There were 125 seed-epochs and 46,750 optimizer batches.

Post-run saved-metadata review checked all 746,125 training record/epoch plans and corresponding published batch cases: 373,364 Bernoulli-clean views and 372,761 selected/applied noise views. Every epoch covered all 5,969 training records exactly once; keys, allowed 24/18/12/6 dB SNRs, augmentation-train role/pool identities, original physical hashes, clean views and strict selected epochs matched. No selected view was flat/inapplicable. Maximum saved achieved-SNR error was 7.105427357601002e-15 dB. This checks saved metadata, not an additional signal decode or optimizer replay.

## Clean comparison

The same 1,280 validation recordings were evaluated for all candidates (N=762, A=114, O=362, ~=42). These are repeated development evaluations, not independent people. Selected CUDA validation confusion matrices matched the saved CPU clean evaluation confusion matrices. Deltas below are augmented minus the same seed's original candidate; no seed was chosen as a winner.

| Seed | Selected / completed epochs | Clean F1 | Δ F1 | AF detections / 114 | Δ AF pp | Errors / 1280 | Δ error pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 25 / 33 | 0.793837 | -0.020374 | 91/114 | +2.6316 | 202/1280 | +1.7188 |
| 43 | 34 / 42 | 0.814482 | +0.019096 | 90/114 | -2.6316 | 178/1280 | -1.0938 |
| 44 | 46 / 50 | 0.809868 | +0.004614 | 87/114 | -6.1404 | 189/1280 | +0.1563 |

Equal-seed clean mean F1 changed from **0.804951 to 0.806062** (Δ +0.001112). Automatic AF sensitivity changed from **80.4094% to 78.3626%** (Δ −2.0468 percentage points); four-class error changed from **14.5573% to 14.8177%** (Δ +0.2604 points). Clean F1 did not decline on average, but seeds differed and clean AF detection/error worsened on average. Paired clean intervals below include zero; these observations do not establish equivalence, noninferiority or a clinical threshold.

## Complete prescribed stress grid

Each cell first averages repetition metrics within a seed, then weights all three seeds equally. Deltas subtract the corresponding original metric before averaging. AF/error values are percentages; their deltas are percentage points. Confusion matrices and predictions were not pooled to calculate these means.

| Noise | SNR dB | Augmented F1 | Δ F1 | AF sensitivity % | Δ AF pp | Error % | Δ error pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| bw | 24 | 0.804742 | +0.001041 | 78.0702 | -1.8519 | 14.8785 | +0.3299 |
| bw | 18 | 0.803524 | +0.001658 | 77.9727 | -1.2671 | 14.9479 | +0.3906 |
| bw | 12 | 0.803099 | +0.002866 | 77.7778 | -0.4873 | 14.9653 | +0.3125 |
| bw | 6 | 0.795342 | +0.020260 | 77.4854 | +2.2417 | 15.0955 | -0.6858 |
| bw | 0 | 0.765195 | +0.064956 | 74.5614 | +6.8226 | 16.2240 | -3.6979 |
| bw | -6 | 0.699899 | +0.165489 | 64.6199 | +18.0312 | 19.8264 | -14.9306 |
| ma | 24 | 0.804959 | +0.000742 | 77.7778 | -1.8519 | 14.8698 | +0.4340 |
| ma | 18 | 0.801948 | +0.012028 | 77.8752 | -0.0975 | 14.9132 | -0.0694 |
| ma | 12 | 0.793849 | +0.070030 | 77.5828 | +4.5809 | 15.1389 | -3.2986 |
| ma | 6 | 0.757614 | +0.173422 | 75.4386 | +17.9337 | 16.5538 | -16.3715 |
| ma | 0 | 0.674178 | +0.274345 | 68.4211 | +33.5283 | 22.5434 | -35.7378 |
| ma | -6 | 0.528031 | +0.332883 | 46.0039 | +30.3119 | 39.4705 | -42.9688 |
| em | 24 | 0.804441 | -0.000051 | 77.9727 | -1.1696 | 14.9132 | +0.3819 |
| em | 18 | 0.800831 | -0.003051 | 77.6803 | -1.0721 | 14.9740 | +0.4340 |
| em | 12 | 0.786293 | -0.003678 | 76.8031 | -1.3645 | 15.5642 | +0.4080 |
| em | 6 | 0.740208 | +0.008212 | 72.2222 | -0.7797 | 17.6476 | -0.5990 |
| em | 0 | 0.616798 | +0.042129 | 58.2846 | +4.4834 | 28.1771 | -4.6962 |
| em | -6 | 0.362834 | +0.086445 | 22.4172 | +6.9201 | 59.4792 | -10.4861 |

Augmentation improved mean F1 in 15/18 cells, with the strongest gains under muscle artifact and severe baseline wander. At muscle artifact −6 dB, mean F1 rose from 0.195149 to 0.528031 and error fell by 42.9688 points, but error remained 39.4705%. Under electrode motion −6 dB, AF sensitivity rose by 6.9201 points to only 22.4172%, and error remained 59.4792%. Mild electrode-motion F1 declined, and several mild conditions had lower AF sensitivity and higher error. Robustness improved substantially in parts of this frozen grid; it did not improve uniformly and did not establish adequate performance under severe noise.

Both arms retained 100% assessed coverage (1,280/1,280 per seed/condition), zero technical failures, zero confidence/quality/other-policy deferrals, zero deferred AF cases and zero flat/inapplicable stress cases. Automatic AF sensitivity is correct automatic AF detections divided by all 114 AF-labeled recordings; wrong predictions remain missed detections. Predicted `~` is an assessed noisy-class prediction. Accepted-case metrics use all eligible cases here because no deferral policy was enabled. The 422,400 arm/seed/condition case entries are repeated evaluations, not a cohort size.

## Paired uncertainty and secondary evidence

The single saved bootstrap used 10,000 shared PCG64 draws, seed 20261016, stratified by N/A/O/~ duplicate groups. All 1,280 validation groups were singletons; verified patient identities were unavailable. Pairing retained arms, model seeds, conditions and repetitions. Linear 2.5/97.5 percentiles are conditional on these fixed models/plans and reused validation data; they omit model-development and population uncertainty and are not multiplicity-adjusted confirmatory tests.

| Condition | Paired Δ F1 95% interval | Paired Δ AF pp 95% interval | Paired Δ error pp 95% interval |
|---|---:|---:|---:|
| clean | [-0.0176, +0.0204] | [-5.2632, +1.1696] | [-0.6510, +1.1458] |
| bw-snr24 | [-0.0172, +0.0200] | [-5.1657, +1.2671] | [-0.5642, +1.2153] |
| bw-snr18 | [-0.0164, +0.0204] | [-4.3860, +1.7544] | [-0.4861, +1.2587] |
| bw-snr12 | [-0.0135, +0.0194] | [-3.4113, +2.3392] | [-0.5471, +1.1545] |
| bw-snr6 | [+0.0040, +0.0360] | [-1.0721, +5.5556] | [-1.5625, +0.1562] |
| bw-snr0 | [+0.0476, +0.0823] | [+2.9240, +11.0136] | [-4.6962, -2.7257] |
| bw-snr-6 | [+0.1482, +0.1836] | [+14.0327, +22.2222] | [-16.3802, -13.5156] |
| ma-snr24 | [-0.0169, +0.0183] | [-5.2632, +1.3645] | [-0.4516, +1.3021] |
| ma-snr18 | [-0.0056, +0.0291] | [-3.2164, +3.0214] | [-0.9375, +0.7812] |
| ma-snr12 | [+0.0509, +0.0888] | [+1.1696, +8.1871] | [-4.3056, -2.3177] |
| ma-snr6 | [+0.1531, +0.1946] | [+13.2554, +23.0019] | [-17.7865, -14.9653] |
| ma-snr0 | [+0.2567, +0.2935] | [+28.5575, +38.5989] | [-37.3698, -34.0712] |
| ma-snr-6 | [+0.3152, +0.3509] | [+24.8538, +35.8674] | [-44.5833, -41.3455] |
| em-snr24 | [-0.0177, +0.0181] | [-4.1910, +1.7544] | [-0.4861, +1.2413] |
| em-snr18 | [-0.0202, +0.0148] | [-3.9961, +1.9493] | [-0.3995, +1.2760] |
| em-snr12 | [-0.0187, +0.0116] | [-4.4834, +1.7544] | [-0.3819, +1.2066] |
| em-snr6 | [-0.0051, +0.0214] | [-3.7037, +2.1442] | [-1.4236, +0.2344] |
| em-snr0 | [+0.0309, +0.0540] | [+0.7797, +8.2846] | [-5.6597, -3.7413] |
| em-snr-6 | [+0.0745, +0.0984] | [+4.1910, +9.8441] | [-11.5885, -9.3924] |


The [compact aggregate JSON](noise-phase-b-v1-results.json) preserves unrounded means/sample SD, every seed/condition primary outcome, per-seed/equal-seed repetition means and paired deltas, both arms' stress-minus-clean changes, all saved bootstrap intervals (including differences of stress deltas), clean confusion/AF accounting, probability metrics and routing summaries. No case identifiers or case-level ledgers are exported to Git. Full per-class, reliability-bin and within-record evidence remains in the indexed local metrics.

Four-class Brier sums over four classes. Top-label ECE uses the frozen 15 bins. Binary AF Brier excludes true noisy recordings (1,238 eligible N/A/O); false AF on true N/O uses 1,124 cases. False-AF share among AF predictions has a separate prediction denominator. No temperature, confidence threshold or quality policy was fitted or selected.

## Supervision and integrity

Approved ceilings were training 8 h including 10 min cleanup, evaluation 6 h including 5 min cleanup, total 14 h; aggregate runner/descendant RSS 4 GiB per stage; GPU0 device-wide memory 7 GiB during training; output/logs 8 GiB through training and 12 GiB cumulative. Floors were 1 GiB available host RAM and 10 GiB free disk.

| Retained snapshot measurement | Training | CPU evaluation |
|---|---:|---:|
| Peak runner/descendant RSS bytes | 2,814,771,200 | 1,868,271,616 |
| Minimum available host RAM bytes | 1,935,794,176 | 3,831,676,928 |
| Peak GPU0 device-wide total−free bytes | 2,896,166,912 | 252,706,816 |
| Peak cumulative output/log bytes | 1,801,639,113 | 2,957,845,393 |
| Minimum free disk bytes | 942,952,615,936 | 941,720,760,320 |

Sampling was configured every second with readable/JSON publication every five seconds. These extrema are from retained published snapshots (1,404 training and 1,546 evaluation), not every in-memory sample or subsecond peaks. Final run/log usage was about 2.755 GiB. RSS may double-count shared pages and excludes the supervisor itself; GPU totals are device-wide, including other reservations/workloads. These sampled guards are not kernel quotas or hard-real-time guarantees. No budget breach was reported; no process-attributed GPU usage is claimed.

Run root: `outputs/noise-robustness/noise-robustness-v1/phase-b`. Supervisor logs: `outputs/noise-phase-b-supervision-v1`. Post-run review: `outputs/noise-phase-b-review-v1`. SHA256 values below identify exact file bytes, not canonical JSON.

| Artifact relative to run root | Bytes | SHA256 |
|---|---:|---|
| `training/artifacts-sha256.json` | 4759207 | `38cdb7fca434ee90dad00caf14e76a4255f8967e0be3178afd23b5d9a3d2955b` |
| `training/registry.json` | 1947045 | `bdce92ce7472b0e49a20ad45fbffff2cba78f91ed698ffbab93d6f1094c18c37` |
| `evaluation/artifacts-sha256.json` | 2892204 | `76d9e76f2bac68e16d10ca43de3ae0b8a72effa34578126f5295d0ba675bfe8f` |
| `evaluation/comparison-seal.json` | 6235 | `6ad11160a588c905dd6982a07c9c1c66fbbc86e5ea0f7d59305586cab500c7ae` |
| `evaluation/metrics.json` | 19858423 | `a6a5bd9db4a1c5ce44021f5169ebc123d0f9aba3af314ac8dcc2e599461c9e7d` |
| `evaluation/bootstrap.json` | 1434532 | `0d02373dd48db97647711b8cebfa89be3e31e835c33cc9274d4b327acff2a103` |
| `evaluation/bootstrap-draws.npy` | 102400128 | `20198394b8bd78f758b2ac3e19023b0b81eabe411b9ff90a3c8cdf220d2b6c08` |
| `evaluation/bootstrap-statistics.npy` | 420480128 | `b08e1fddd3f29489472d592d6d25b859304d3459a3da3e09090707f907404c22` |
| `status.json` | 180 | `ac5b033a1d5f006fd842fe027b5fa3ec17bf2c6d71d9288df3bb14584e21fc2e` |

Post-run verification passed exact inventories and every indexed byte hash for 46,899 training files, 26,575 evaluation files and 26,577 baseline files. It rechecked all 330 complete prediction arrays, required batch/ledger seals and comparison-seal identities, all 13,200 physical-input/corruption-metadata batch pairings, and 330 independently reconstructed confusion matrices/AF/error/F1 outcomes. Bootstrap shapes were (10000,1280) draws and (10000,3,4,73,2,3) statistics; all statistics were finite, original class/group multiplicities were retained, paired differences matched and all 5,256 saved intervals matched the saved statistics. No inference, training or bootstrap was rerun.

The first post-run verifier assertion incorrectly compared the full canonical configuration identity with the policy identity excluding authorization. The verifier was corrected to check both distinct identities and passed. No run bytes or numerical results changed. Its final code, preflight record, integrity/training reviews and supervisor log identities are recorded in the aggregate JSON.

Actual repository checks: full synthetic/regression suite **537 passed in 52.24 s**; supervisor tests **39 passed in 1.88 s** after focused fixture corrections; Ruff lint/format passed (44 files), repository contracts passed, frozen verification passed (17 identities; 5,969/1,280/1,279 metadata rows, zero ECG files opened), CPU/CUDA dependency compatibility passed (58/74 packages), both offline lock checks passed (60/74 packages), and sdist/wheel build passed with existing dependencies/no isolation. No dependency installation or real model probe was performed for checks.

Initial post-run suite: 534 passed and three failed in 53.39 s because fixtures assumed the real execution flag would always be false or the real output root absent. Authorization-commit CI also failed its test step. After the completed run, those tests were restricted to generated disabled configurations/temporary roots; production source and frozen policy were unchanged. The experiment remains attributed to `4cfc760`, not the later outcome/test commit. Final diff/documentation checks and Git/CI status accompany the outcome commit/PR.

## Limits and next task

Validation reuse, historical candidate selection and training/runtime provenance limit generalization and causal attribution. Verified patient/noise-volunteer identities are unavailable; separate temporal noise roles do not establish independent noise sources. Fixed seeds/maps and conditional bootstrap intervals omit development/population uncertainty. Retaining source rhythm labels under severe corruption remains an unverified recognizability assumption; native NSTDB voltage calibration remains unverified. No consumed-holdout access, final-cohort evaluation, BUT, quality-model work, calibration, new architecture or UI work occurred.

Signals, checkpoints, batch predictions, case ledgers, bulk bootstrap arrays and logs remain ignored/local-only. Compact aggregate identities do not substitute for those bytes. Durable archival and restoration remain pending an approved destination; no archive upload was made. The single next task is an owner review of the complete Phase B tradeoffs to freeze the requirements and protocol for future confidence/quality work before implementation or another experiment. No future execution is authorized by this result or retained flag.
