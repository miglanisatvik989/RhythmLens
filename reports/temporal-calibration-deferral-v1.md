# Temporal calibration and confidence deferral v1 — reviewed-v1 results

Completed exploratory development analysis using all three existing temporal checkpoints, seeds 42, 43 and 44, with no best-seed selection. These are observed saved-artifact results, not a new experiment during report preparation. Calibration improvements were mixed. Confidence deferral increased accepted-case macro F1 but reduced automatic AF sensitivity in every seed. Accepted-case F1 measures a different subset and does not establish overall improvement. No policy retuning or deployment recommendation is based on these outcomes.

## Protocol and execution

The [committed protocol](../docs/experiments/temporal-calibration-deferral-v1.md), [analysis YAML](../configs/temporal-calibration-deferral-v1.yaml), and [implementation verification](temporal-calibration-deferral-v1-verification.md) define the analysis. Implementation revision: `f7834b73013f8363b3be6fb071e8957ec73efd00`. Authorization/execution revision: **`47d198e15f0fdcfbd2210ea9cf6700af7a1b3889`**. Its only change was literal YAML `execution_authorized: false` to `true`; explicit CLI authorization was also supplied. No source/configuration changes or further commits occurred between fitting and assessment.

Historical commands, all completed successfully, in order (do not rerun against existing outputs):

```bash
.venv/bin/python scripts/manage.py analysis-preflight --config configs/temporal-calibration-deferral-v1.yaml
.venv/bin/python scripts/manage.py analysis-fit --config configs/temporal-calibration-deferral-v1.yaml --execution-id reviewed-v1 --authorize-analysis
.venv/bin/python scripts/manage.py analysis-assess --config configs/temporal-calibration-deferral-v1.yaml --execution-id reviewed-v1 --authorize-analysis
```

Preflight verified all three saved CPU validation exports, their recorded training/evaluation provenance, required finite logits, exact frozen IDs/labels and baseline confusion/supports, and float32/float64 probability and baseline NLL tolerances. Fit performed role assignment, temperature fitting and threshold selection; the frozen-artifact review passed before assessment. No failures or silent retries occurred. Every assessment arm is complete, with zero technical failures. No new inference, preprocessing fit, model training, dependency installation, holdout signal access or holdout evaluation occurred.

Reuse of `median-fixed-training-rms-v1` retains the training-only fitted scale 0.18008390945457742 mV, native 300 Hz and median centering. Calibration used one equal-record-weighted scalar temperature per seed, minimizing calibration-fit four-class NLL over inverse temperature [0.05,20] with deterministic golden-section search. Threshold selection minimized accepted error fraction on the fixed 0.00–1.00 grid in 0.01 steps, requiring at least 90% overall and AF selection coverage, with fixed tie rules. Positive temperature preserves argmax. Signal-quality gating was disabled.

Execution used CPU float64 analysis, 2 threads, Python 3.11.16, NumPy 2.2.6, PyYAML 6.0.3 and threadpoolctl 3.7.0. Exact installed versions, platform, source hashes, invocations and UTC timestamps remain in the frozen policy and assessment summary. Fit recorded `2026-10-13T17:48:53.988113+00:00`; assessment recorded `2026-10-13T17:50:07.280384+00:00`.

## Roles and grouping

Partition seed `20261013`; algorithm `validation-group-greedy-sha256-v1`, with target fractions 0.40/0.30/0.30 and indivisible frozen exact duplicate groups. All seeds use identical roles. Original train/validation/holdout splits remain unchanged.

| Role | N | A (AF) | O | ~ (noisy) | Total | Duplicate groups |
| --- | --- | --- | --- | --- | --- | --- |
| calibration-fit | 305 | 46 | 145 | 17 | 513 | 513 |
| threshold-select | 229 | 34 | 109 | 13 | 385 | 385 |
| dev-assess | 228 | 34 | 108 | 12 | 382 | 382 |
| All validation | 762 | 114 | 362 | 42 | 1280 | 1280 |

Every eligible validation recording and every labeled AF recording is assigned exactly once. No duplicate-group overlap exists between roles; all four classes occur in each role. Calibration-fit and threshold-select diagnostics are separate local artifacts; headline tables below use **dev-assess only: 382 cases, including 34 AF and 12 noisy cases**. Patient identifiers are unknown; no identities were inferred.

## Frozen fits and selection decisions

| Seed | T | 1/T | Threshold | Fit NLL | Iterations | Selection coverage | Selection AF coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 1.221916 | 0.818387 | 0.51 | 0.446874 | 45 | 371/385 (96.36%) | 33/34 (97.06%) |
| 43 | 1.321548 | 0.756689 | 0.53 | 0.471267 | 45 | 357/385 (92.73%) | 31/34 (91.18%) |
| 44 | 1.104764 | 0.905170 | 0.58 | 0.452387 | 45 | 351/385 (91.17%) | 32/34 (94.12%) |

All three fits converged with finite objectives and positive temperatures, with no boundary solution. All thresholds satisfied both selection coverage constraints. Temperatures, thresholds and complete selection tables were frozen before dev-assess; no policy changes followed assessment. Rounded values here are descriptive; full precision remains in `frozen-policy.json` and each seed’s fit/selection JSON.

## Assessment decisions and deferrals

Assessed coverage includes accepted four-class noisy predictions. Rhythm-output coverage counts only accepted predicted N/A/O. Predicted `~` is an assessed noisy-class outcome, never a confidence or quality deferral and never a normal/non-AF rhythm determination. Every deferred AF case remains in the automatic AF sensitivity denominator.

| Seed | Arm | Assessed coverage | Rhythm-output coverage | Confidence deferrals/all | AF confidence deferrals/all AF | Automatic AF sensitivity | Accepted macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | Raw | 382/382 (100.00%) | 372/382 (97.38%) | 0/382 (0.00%) | 0/34 (0.00%) | 24/34 (70.59%) | 0.835974 |
| 42 | Calibrated | 382/382 (100.00%) | 372/382 (97.38%) | 0/382 (0.00%) | 0/34 (0.00%) | 24/34 (70.59%) | 0.835974 |
| 42 | Calibrated + deferral | 369/382 (96.60%) | 361/382 (94.50%) | 13/382 (3.40%) | 3/34 (8.82%) | 23/34 (67.65%) | 0.853875 |
| 43 | Raw | 382/382 (100.00%) | 374/382 (97.91%) | 0/382 (0.00%) | 0/34 (0.00%) | 26/34 (76.47%) | 0.805150 |
| 43 | Calibrated | 382/382 (100.00%) | 374/382 (97.91%) | 0/382 (0.00%) | 0/34 (0.00%) | 26/34 (76.47%) | 0.805150 |
| 43 | Calibrated + deferral | 361/382 (94.50%) | 354/382 (92.67%) | 21/382 (5.50%) | 2/34 (5.88%) | 24/34 (70.59%) | 0.829881 |
| 44 | Raw | 382/382 (100.00%) | 370/382 (96.86%) | 0/382 (0.00%) | 0/34 (0.00%) | 26/34 (76.47%) | 0.794727 |
| 44 | Calibrated | 382/382 (100.00%) | 370/382 (96.86%) | 0/382 (0.00%) | 0/34 (0.00%) | 26/34 (76.47%) | 0.794727 |
| 44 | Calibrated + deferral | 347/382 (90.84%) | 337/382 (88.22%) | 35/382 (9.16%) | 5/34 (14.71%) | 23/34 (67.65%) | 0.850056 |

All deferrals have primary reason confidence: 100% of deferrals and AF deferrals in the deferral arm; the raw/calibrated arms have no deferrals, so reason shares are undefined (0/0). Signal-quality deferrals and technical failures are zero in all nine assessments. Eligible accounting is `382 = accepted + confidence_deferred + quality_deferred + technical_failure` for every arm.

AF accounting, columns are accepted predicted class followed by the three non-accepted dispositions; every row sums to 34. Accepted N/O/~ outcomes are AF detection misses; confidence-deferred AF cases are also not detections.

| Seed | Arm | N | A | O | ~ | Confidence defer | Quality defer | Technical failure | AF coverage | Accepted AF recall | Accepted AF precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | Raw | 2 | 24 | 7 | 1 | 0 | 0 | 0 | 34/34 (100.00%) | 24/34 (70.59%) | 24/27 (88.89%) |
| 42 | Calibrated | 2 | 24 | 7 | 1 | 0 | 0 | 0 | 34/34 (100.00%) | 24/34 (70.59%) | 24/27 (88.89%) |
| 42 | Calibrated + deferral | 1 | 23 | 6 | 1 | 3 | 0 | 0 | 31/34 (91.18%) | 23/31 (74.19%) | 23/25 (92.00%) |
| 43 | Raw | 2 | 26 | 6 | 0 | 0 | 0 | 0 | 34/34 (100.00%) | 26/34 (76.47%) | 26/36 (72.22%) |
| 43 | Calibrated | 2 | 26 | 6 | 0 | 0 | 0 | 0 | 34/34 (100.00%) | 26/34 (76.47%) | 26/36 (72.22%) |
| 43 | Calibrated + deferral | 2 | 24 | 6 | 0 | 2 | 0 | 0 | 32/34 (94.12%) | 24/32 (75.00%) | 24/33 (72.73%) |
| 44 | Raw | 1 | 26 | 5 | 2 | 0 | 0 | 0 | 34/34 (100.00%) | 26/34 (76.47%) | 26/38 (68.42%) |
| 44 | Calibrated | 1 | 26 | 5 | 2 | 0 | 0 | 0 | 34/34 (100.00%) | 26/34 (76.47%) | 26/38 (68.42%) |
| 44 | Calibrated + deferral | 1 | 23 | 3 | 2 | 5 | 0 | 0 | 29/34 (85.29%) | 23/29 (79.31%) | 23/29 (79.31%) |

**Seed 44 assessment AF coverage was 29/34 = 85.29%.** Selection constraints do not guarantee assessment coverage. The policy was not retuned in response. Accepted noisy predictions across all true labels numbered 10/8/12 for raw and calibrated arms and 8/7/10 for deferral (seeds 42/43/44). Full true-class-by-outcome matrices and per-class accepted metrics remain in each assessment metrics file.

## Probability metrics

NLL, multiclass Brier (sum over four classes, no division by class count) and top-label ECE use all 382 eligible assessment recordings before deferral. ECE/reliability use 15 fixed equal-width bins. Binary AF Brier and AF reliability use **370 true N/A/O cases**, excluding all **12 true noisy-label cases**, whose underlying rhythm is unknown. They use the existing four-class `p_A` without renormalization. Noisy cases remain in four-class metrics, coverage and the complete ledger. Lower scores are better.

| Seed | Arm | NLL | Multiclass Brier | Top-label ECE | Binary AF Brier | AF-probability eligible | Excluded ~ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | Raw | 0.355350 | 0.188763 | 0.030277 | 0.030373 | 370 | 12 |
| 42 | Calibrated | 0.362876 | 0.191886 | 0.053549 | 0.030637 | 370 | 12 |
| 42 | Calibrated + deferral | 0.362876 | 0.191886 | 0.053549 | 0.030637 | 370 | 12 |
| 43 | Raw | 0.424808 | 0.228928 | 0.045659 | 0.041201 | 370 | 12 |
| 43 | Calibrated | 0.418156 | 0.226992 | 0.042563 | 0.039769 | 370 | 12 |
| 43 | Calibrated + deferral | 0.418156 | 0.226992 | 0.042563 | 0.039769 | 370 | 12 |
| 44 | Raw | 0.336260 | 0.185674 | 0.085030 | 0.033269 | 370 | 12 |
| 44 | Calibrated | 0.345979 | 0.188072 | 0.084885 | 0.033044 | 370 | 12 |
| 44 | Calibrated + deferral | 0.345979 | 0.188072 | 0.084885 | 0.033044 | 370 | 12 |

Calibrated and calibrated-with-deferral probability metrics coincide because they score the same probabilities on all eligible cases before deferral. AF and top-label reliability-bin counts, observed fractions and mean probabilities are saved in the per-arm metrics JSON. Undefined denominators produce null with a reason, not zero performance; an explicitly named legacy zero-fill macro F1 is also available. All headline metrics above are defined.

## Seed summaries and paired differences

All summaries use n=3, arithmetic mean and sample SD with n−1 denominator, from unrounded saved metrics. **Seeds assess the same cases and are not independent cohorts.** Repeated recordings were not pooled as independent observations. No statistical significance is claimed.

| Metric (proportions) | Raw | Calibrated | Calibrated + deferral |
| --- | --- | --- | --- |
| Coverage | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.939791 ± 0.029151 |
| Rhythm-output coverage | 0.973822 ± 0.005236 | 0.973822 ± 0.005236 | 0.917976 ± 0.032310 |
| AF coverage | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.901961 ± 0.044927 |
| Automatic AF sensitivity | 0.745098 ± 0.033962 | 0.745098 ± 0.033962 | 0.686275 ± 0.016981 |
| Accepted accuracy | 0.865620 ± 0.009911 | 0.865620 ± 0.009911 | 0.888636 ± 0.011550 |
| Selective risk | 0.134380 ± 0.009911 | 0.134380 ± 0.009911 | 0.111364 ± 0.011550 |
| Accepted AF recall | 0.745098 ± 0.033962 | 0.745098 ± 0.033962 | 0.761680 ± 0.027511 |
| Accepted AF precision | 0.765107 ± 0.108870 | 0.765107 ± 0.108870 | 0.813459 ± 0.097963 |
| Accepted macro F1 | 0.811951 ± 0.021448 | 0.811951 ± 0.021448 | 0.844604 ± 0.012893 |
| NLL | 0.372139 ± 0.046601 | 0.375670 ± 0.037751 | 0.375670 ± 0.037751 |
| Multiclass Brier | 0.201122 ± 0.024131 | 0.202316 ± 0.021454 | 0.202316 ± 0.021454 |
| Top-label ECE | 0.053655 ± 0.028239 | 0.060333 ± 0.021961 | 0.060333 ± 0.021961 |
| Binary AF Brier | 0.034948 ± 0.005605 | 0.034483 ± 0.004733 | 0.034483 ± 0.004733 |

Decision-metric paired differences: calibrated-with-deferral minus raw. Calibrated-minus-raw hard-label decision differences are zero. Deferral-minus-calibrated decision differences therefore equal those below. Differences are absolute proportions, not relative percent changes.

| Seed | Δ coverage | Δ automatic AF sensitivity | Δ accepted macro F1 |
| --- | --- | --- | --- |
| 42 | -0.034031 | -0.029412 | +0.017901 |
| 43 | -0.054974 | -0.058824 | +0.024730 |
| 44 | -0.091623 | -0.088235 | +0.055328 |
| Mean ± sample SD | -0.060209 ± 0.029151 | -0.058824 ± 0.029412 | +0.032653 ± 0.019932 |

Probability-metric paired differences: calibrated minus raw. Calibrated-with-deferral minus raw has exactly the same probability differences; deferral minus calibrated is zero for these all-eligible metrics.

| Seed | Δ NLL | Δ multiclass Brier | Δ top-label ECE | Δ binary AF Brier |
| --- | --- | --- | --- | --- |
| 42 | +0.007526 | +0.003123 | +0.023272 | +0.000264 |
| 43 | -0.006652 | -0.001937 | -0.003096 | -0.001431 |
| 44 | +0.009719 | +0.002398 | -0.000145 | -0.000226 |
| Mean ± sample SD | +0.003531 ± 0.008887 | +0.001195 ± 0.002736 | +0.006677 ± 0.014447 | -0.000464 ± 0.000872 |

Calibration improvements were mixed: seed 43 improved all four scores, while NLL and multiclass Brier worsened for seeds 42 and 44. Mean NLL, multiclass Brier and ECE worsened; mean binary AF Brier improved slightly. Deferral increased accepted-case F1 but reduced automatic AF sensitivity in every seed. Accepted-case F1 measures a different, selected subset and does not establish overall improvement. No best seed was selected.

## Identities and artifact chain

All hashes below are SHA256. The original [multiseed provenance](rhythm-multiseed-v1.json) identifies training configuration/source/checkpoint and evaluation identities. The input registry binds those identities to the actual saved prediction export hashes; those prediction hashes were first measured during earlier preflight, not invented as historical training-export hashes.

| Artifact | SHA256 |
| --- | --- |
| Authorized analysis YAML | `b357ea551d68fecdee42661e36b0bab52a05f9b8b5eaed79959ce761ef7f0d2d` |
| Protocol | `cd670b8438270012fc958bba641d8cab32318ea4befe54a3fa741b724e7af923` |
| Original multiseed JSON | `b4b98cef0fedf2e1da7d1a904a86572c5434dd799a033a7625fd348632c5b3f8` |
| Role manifest | `3314e9a1dc45fddb8df770a89d72f02b61a82e4491800fa3ba64b6daed7e880b` |
| Role provenance | `79bb0b0f5d9142d8425db9b4272916da9dcfa7f26e2c4d3ed3eafba034579fdd` |
| Input registry | `d7fc2aacd1a36cf0a6c93b852d9c4878570b3605e4759e3ab52bda936a76440e` |
| Frozen policy | `2ed9af297d8789bd3be81794ca6c1b9270c5c3f69886668cbaae6ab349de509b` |
| Assessment summary | `0cd221684d8cda3f592a25a9d5b0bbf229272e77eb524e7508263abb1702b5d0` |
| Assessment hash manifest | `4c9e715b36ec7ed32ba6bbb28c96275f067963308edcc18fb7ea000a9090b57a` |

| Seed | Saved CPU prediction SHA256 | Checkpoint identity SHA256 |
| --- | --- | --- |
| 42 | `e11ad0f1bcac963f82f2143535b57f44c11fd323790333430875f20c06767d78` | `217110e48c995fcb8eac9a7eb3885356170fd461dc50b333cd62fd1022d2031d` |
| 43 | `7dc18886c41d6cde5c1e43a70ec96534c09bbee9193332850d843d983e30ae0a` | `c420c2a96ecf1e9f25ad4a7417a321a96b9eb47c217db613d9502c26e2bcde74` |
| 44 | `c48b6f5fc05a49b78f83f85dcd02c73199b82dc765833dddfacf3925cd45886c` | `e6965190d29b0214a732f6bf5ef4172daaae3ede30b2e549c8d7c9760a8791c5` |

| Seed | Training report SHA256 | CPU evaluation report SHA256 |
| --- | --- | --- |
| 42 | `2c2781bf904b7df2b9c37397d14bdb4378439f50c641f560dbb0301de6abad14` | `cc140dc714b07e93ba1cbe1b94d9f57958b735fb0758bdd90c60d4b045a1c6c8` |
| 43 | `205e31627a781b0b968c5bea46a1d5bd146037ec8c0fa526122d4336b9badb6b` | `60065126c5c8a15fc73bbabeaefc0892f9123e2fa2c0012c86f9c4bb453c1848` |
| 44 | `b52dc8f78b46c5e7b03fd65eba8f0c72a37b18b1c0aee852e089f3d6b6905e99` | `cf6d480784478eeca3cdc5b817593a23f5b0aa8a71ebbcb10189301fb3680200` |

Original validation manifest SHA256: `19b515593afa35b98a3c86fe0c45f32e67e6a29a56aab6186b81e8155552e5cd`. Frozen metadata SHA256: `2c887a1934fc93f40861a87522b5b2b5aee310aa622335e7912beffc0e26c121`. Original assignment and preprocessing identities remain pinned in the authorized YAML and original report; no split or preprocessing changes occurred.

| Source metadata | SHA256 |
| --- | --- |
| source_audit | `2e85add08052656eed0b72ddf0bb40c1d7163ee131aa18d796625ffb78516b40` |
| source_manifest | `a390f3103023df679d08544ce6f4ca9f972fb59394901a78c3027de93f1d0560` |
| source_provenance | `d7d79603a51d5ef35f93f5d2ae55344f0b7a6fe567d8c46b29a31575c6c4b636` |

Both active dependency locks remain unchanged; their identities are in the authorized YAML. The historical data dependency-lock identity is distinct from active environment locks. Exact training/evaluation environments and original YAML/resolved-config/source hashes remain in original provenance and the input registry. Checkpoint identities here are recorded provenance identities; report preparation did not deserialize or evaluate model weights.

The frozen policy binds configuration, protocol, role manifest/provenance, all three fitted policies, execution source/environment and 44 pre-assessment artifact hashes. The assessment summary binds its parent frozen-policy hash; `assessment/artifact-provenance.json` binds 19 assessment files. Those files plus the two hash-bearing container files account for all **65 existing run-output files**. Report preparation verified every indexed hash, exact complete output membership, execution revision, current protocol/configuration/source identities and preserved role identities. The frozen-policy hash also matches the execution record `2ed9af297d8789bd3be81794ca6c1b9270c5c3f69886668cbaae6ab349de509b`.

## Preservation and verification boundary

Complete local output root: `outputs/calibration/temporal-calibration-deferral-v1/reviewed-v1/`. It contains the input registry, a role-manifest copy, seed42/43/44 calibration fits and selection tables, calibration-fit/threshold-select ledgers and metrics, frozen policy, and all development-assessment ledgers/metrics/reliability tables, summary and assessment hash manifest. Existing artifacts are unchanged and no fitting or assessment was rerun to prepare this report.

Trackable role files remain at their protocol-specified paths: [roles.csv](../data/manifests/temporal-calibration-deferral-v1/roles.csv) and [roles.provenance.json](../data/manifests/temporal-calibration-deferral-v1/roles.provenance.json). This aggregate report and documentation indexes can be committed with those two de-identified metadata files after review; predictions, full local outputs, ECG, weights and checkpoints must remain out of Git.

**No experiment-output archival mechanism or configured archival destination was found** in the tracked launcher/scripts/source, Makefile, reproduction docs, dependency descriptors or CI workflow. Dataset ZIP acquisition and `docs/archive/` are unrelated to output preservation; separate worktrees only retain separate local ignored files. No new storage destination or archival mechanism was invented, and no archive/upload was performed. **Ignored reviewed-v1 outputs remain local and are not preserved by a Git commit.** Recorded hashes permit integrity checking but cannot restore missing bytes. Durable external preservation of the complete outputs remains unresolved until an approved destination/workflow is provided.

Checks for this documentation/provenance task are limited to existing artifact and identity verification, ledger/summary consistency, aggregate/paired-statistic transcription, Markdown link/whitespace checks, role-file trackability/output ignore rules, and unchanged protected-file hashes. The earlier 260-test implementation result is historical evidence, not a newly run test suite here. No authorization, policies, thresholds, source code, frozen artifacts, baseline results, role metadata or original provenance were changed.

## Interpretation and limits

Validation already informed checkpoint selection (including early stopping) and architecture prioritization; seed42 informed the multiseed follow-up. Dividing that validation set into fitting/selection/assessment roles does not remove this prior reuse. This separate development assessment is exploratory, not an independent holdout or confirmatory test.

The assessment has only 34 labeled AF and 12 noisy-label recordings. Patient identifiers are unavailable; patient/source/time independence remains unverified despite exact-duplicate grouping. Three seeds assess training variability on the same cases, not independent cohorts. Selection constraints do not guarantee assessment coverage, as seed 44’s 85.29% AF coverage demonstrates. Confidence is not a signal-quality measure; no quality gate or underlying rhythm labels for noisy recordings were inferred.

No policy retuning or deployment recommendation is based on these outcomes. No claim of overall improvement, statistical significance, clinical readiness or holdout performance is made. Holdout evaluation remains excluded and untouched. Further changes require a separately reviewed protocol with explicit acknowledgement that these assessment outcomes have now been observed.
