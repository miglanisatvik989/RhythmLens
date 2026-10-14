# Completed internal rhythm-only evaluation v1

Date: 2026-10-14. **Qualified internal rhythm-only evidence**, not final validation
of quality-aware triage, clinical safety or deployment readiness. All three fixed
temporal checkpoints are reported equally. Mean automatic AF sensitivity was
**80.70%**, mean automatic four-class error **14.33%**, and assessed coverage
**100%**. This holdout is **consumed for future development**; subsequent
quality-gating changes require separate evaluation evidence.

## Execution and fixed inputs

Execution revision: **`bbdf121a81557bb4e614a452a1ffb42ba6a9b27d`**.
Implementation revision: `9818a49e9d9c7bec2b75eb642628c35db41f62b6`.
The execution commit changed only literal YAML authorization from false to true.
Source, configuration, checkpoints and preprocessing remained fixed during the run.
The [frozen protocol](../docs/experiments/temporal-final-evaluation-v1.md),
[configuration](../configs/temporal-final-evaluation-v1.yaml),
[integrity audit](final-evaluation-integrity-audit-v1.md) and
[synthetic verification](temporal-final-evaluation-v1-verification.md) supply the
methods and history. Those protocol/verification documents describe their earlier
pre-execution milestones; their original text and experiment commits are preserved.
The input registry binds the exact execution-revision bytes, including historical
documentation hashes, rather than later index updates.

Both commands completed with exit code 0, sequentially (historical record, **do not
rerun** against the occupied one-time output root):

```bash
.venv/bin/python scripts/manage.py final-preflight --config configs/temporal-final-evaluation-v1.yaml
.venv/bin/python scripts/manage.py final-evaluate --config configs/temporal-final-evaluation-v1.yaml --authorize-evaluation
```

Preflight verified frozen metadata, candidate identities and the locked CPU
runtime. The executor checked all audited raw signal-file hashes before model
inference, then decoded through the approved decoder. All three logits arrays were
sealed before scoring. No failure, retry, exclusion, best-seed selection, ensemble,
calibration, confidence deferral, quality gating, retraining or preprocessing refit
occurred. This report reads existing artifacts only; no evaluation was rerun.

The model is `rhythm-dilated-v1`, native 300 Hz physical mV, four-class N,A,O,~
argmax at T=1. Median centering and fixed training RMS scale
0.18008390945457742 mV reuse preprocessing fitted on 5,969 train recordings only.
Inference used float32 CPU, batch size 16, canonical recording-ID order, padded
prefix masks, evaluation/inference mode, two threads and deterministic algorithms.
Probability scoring used float64. Runtime: Python 3.11.16, Torch 2.8.0+cpu,
NumPy 2.2.6. Full package/platform/invocation details remain in `environment.json`.

Execution started `2026-10-14T03:04:26.379968+00:00` and completed `2026-10-14T03:10:13.194619+00:00` (UTC).

| Seed | Selected epoch | Checkpoint SHA256 |
| --- | ---: | --- |
| 42 | 31 | `217110e48c995fcb8eac9a7eb3885356170fd461dc50b333cd62fd1022d2031d` |
| 43 | 42 | `c420c2a96ecf1e9f25ad4a7417a321a96b9eb47c217db613d9502c26e2bcde74` |
| 44 | 39 | `e6965190d29b0214a732f6bf5ef4172daaae3ede30b2e549c8d7c9760a8791c5` |

All three preprocessing JSON SHA256: `a4719420f5b867e3870386b488c954fb85f1b695f2264f57235cf0e0b80783b6`.

## Eligible cases and complete accounting

Each seed assessed exactly the same **1,279 unique recording IDs** and **1,279
singleton exact-duplicate groups**: N=761, A=114, O=362, ~=42. No identity was
invented and no exposed or noisy case was removed. True-~ cases remain in all
four-class metrics. Predicted ~ is an assessed four-class outcome, not a deferral
or automatic non-AF determination. Rhythm-output coverage counts predicted N/A/O.

| Seed | Assessed / eligible | Rhythm outputs / eligible | Predicted ~ / eligible | Confidence deferrals | Quality deferrals | Other policy deferrals | Technical failures |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 42 | 1279/1279 (100.00%) | 1247/1279 (97.50%) | 32/1279 (2.50%) | 0 | 0 | 0 | 0 |
| 43 | 1279/1279 (100.00%) | 1249/1279 (97.65%) | 30/1279 (2.35%) | 0 | 0 | 0 | 0 |
| 44 | 1279/1279 (100.00%) | 1240/1279 (96.95%) | 39/1279 (3.05%) | 0 | 0 | 0 | 0 |

For every seed, E=1,279=accepted+confidence-deferred+quality-deferred+technical
failure. Overall and AF deferral rates are 0/1,279 and 0/114, respectively;
confidence, quality and other policy reason counts are zero. Shares among deferred
cases are undefined (0/0), rather than evidence of a dominant reason. AF assessed
coverage is 114/114. Every ID occurs exactly once per seed.

AF outcomes below count **all 114 labeled AF**, including misses. Automatic AF
sensitivity is correct automatic A predictions divided by all labeled AF; any
deferred or failed AF case would remain in that denominator and not be detected.

| Seed | Predicted N on A | Predicted A on A | Predicted O on A | Predicted ~ on A | AF deferred | AF technical failures | Automatic AF sensitivity | AF precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 42 | 3 | 86 | 25 | 0 | 0 | 0 | 86/114 (75.44%) | 86/100 (86.00%) |
| 43 | 8 | 93 | 13 | 0 | 0 | 0 | 93/114 (81.58%) | 93/120 (77.50%) |
| 44 | 4 | 97 | 12 | 1 | 0 | 0 | 97/114 (85.09%) | 97/129 (75.19%) |

## Confusion matrices and decision metrics

Rows are true labels; columns are predictions, both in N,A,O,~ order. Row sums
are 761/114/362/42 for every seed. Automatic error is an incorrect four-class
prediction divided by all eligible recordings; with full coverage it equals
accepted-case selective risk. This is not an AF-only error metric.

### Seed 42

| True / predicted | N | A | O | ~ |
| --- | ---: | ---: | ---: | ---: |
| N | 701 | 2 | 56 | 2 |
| A | 3 | 86 | 25 | 0 |
| O | 53 | 10 | 294 | 5 |
| ~ | 12 | 2 | 3 | 25 |

### Seed 43

| True / predicted | N | A | O | ~ |
| --- | ---: | ---: | ---: | ---: |
| N | 716 | 4 | 38 | 3 |
| A | 8 | 93 | 13 | 0 |
| O | 87 | 19 | 253 | 3 |
| ~ | 9 | 4 | 5 | 24 |

### Seed 44

| True / predicted | N | A | O | ~ |
| --- | ---: | ---: | ---: | ---: |
| N | 694 | 6 | 57 | 4 |
| A | 4 | 97 | 12 | 1 |
| O | 57 | 21 | 277 | 7 |
| ~ | 8 | 5 | 2 | 27 |

| Seed | Accuracy | Automatic error | Macro F1 |
| --- | --- | --- | ---: |
| 42 | 1106/1279 (86.47%) | 173/1279 (13.53%) | 0.797587 |
| 43 | 1086/1279 (84.91%) | 193/1279 (15.09%) | 0.780348 |
| 44 | 1095/1279 (85.61%) | 184/1279 (14.39%) | 0.789016 |

Per-class precision is TP/predicted count; recall is TP/true support; F1 is
2TP/(support+predicted count). Macro F1 averages the four class F1 values.
All present denominators are nonzero. The protocol assigns zero F1 when both
supports are zero and flags that class; no class required that convention here.

| Seed | Class | Support | Predicted count | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | N | 761 | 769 | 0.911573 | 0.921156 | 0.916340 |
| 42 | A | 114 | 100 | 0.860000 | 0.754386 | 0.803738 |
| 42 | O | 362 | 378 | 0.777778 | 0.812155 | 0.794595 |
| 42 | ~ | 42 | 32 | 0.781250 | 0.595238 | 0.675676 |
| 43 | N | 761 | 820 | 0.873171 | 0.940867 | 0.905756 |
| 43 | A | 114 | 120 | 0.775000 | 0.815789 | 0.794872 |
| 43 | O | 362 | 309 | 0.818770 | 0.698895 | 0.754098 |
| 43 | ~ | 42 | 30 | 0.800000 | 0.571429 | 0.666667 |
| 44 | N | 761 | 763 | 0.909567 | 0.911958 | 0.910761 |
| 44 | A | 114 | 129 | 0.751938 | 0.850877 | 0.798354 |
| 44 | O | 362 | 348 | 0.795977 | 0.765193 | 0.780282 |
| 44 | ~ | 42 | 39 | 0.692308 | 0.642857 | 0.666667 |

## Probability metrics and reliability

NLL, multiclass Brier (sum across four classes, no division by class count) and
top-label ECE use all 1,279 cases. Binary AF Brier/reliability use **1,237 true
N/A/O cases**, excluding **42 true-~** cases, and original p_A without
renormalization. Lower scores are better; no calibration is fitted.

| Seed | NLL | Multiclass Brier | Top-label ECE | Binary AF Brier | AF ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.389905 | 0.207022 | 0.015304 | 0.024705 | 0.016576 |
| 43 | 0.448003 | 0.229970 | 0.037566 | 0.028442 | 0.019080 |
| 44 | 0.395993 | 0.207126 | 0.037241 | 0.024219 | 0.014441 |
| Equal mean ± sample SD | 0.411300 ± 0.031931 | 0.214706 ± 0.013219 | 0.030037 ± 0.012760 | 0.025789 ± 0.002311 | 0.016699 ± 0.002322 |

Reliability uses 15 fixed equal-width bins on [0,1], left-closed/right-open except
the final bin includes 1. Each table bin i is [(i−1)/15,i/15). Empty bins have
undefined means and contribute zero ECE. Top-label observed fraction is accuracy;
AF observed fraction is the AF label fraction within the binary scope.

| Seed | Bin | Top count | Mean confidence | Observed accuracy | AF count | Mean p_A | Observed AF fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1 | 0 | undefined | undefined | 1071 | 0.003467 | 0.005602 |
| 42 | 2 | 0 | undefined | undefined | 27 | 0.090947 | 0.185185 |
| 42 | 3 | 0 | undefined | undefined | 12 | 0.166534 | 0.333333 |
| 42 | 4 | 0 | undefined | undefined | 9 | 0.251314 | 0.333333 |
| 42 | 5 | 0 | undefined | undefined | 7 | 0.303405 | 0.285714 |
| 42 | 6 | 3 | 0.365938 | 0.000000 | 6 | 0.364737 | 0.666667 |
| 42 | 7 | 11 | 0.442307 | 0.363636 | 5 | 0.431883 | 0.800000 |
| 42 | 8 | 38 | 0.510532 | 0.631579 | 5 | 0.493886 | 0.400000 |
| 42 | 9 | 56 | 0.567769 | 0.535714 | 8 | 0.574560 | 0.500000 |
| 42 | 10 | 70 | 0.634452 | 0.628571 | 11 | 0.640909 | 0.909091 |
| 42 | 11 | 48 | 0.703931 | 0.687500 | 8 | 0.708176 | 0.875000 |
| 42 | 12 | 74 | 0.767928 | 0.689189 | 11 | 0.773019 | 0.818182 |
| 42 | 13 | 98 | 0.834005 | 0.846939 | 8 | 0.825338 | 1.000000 |
| 42 | 14 | 212 | 0.905929 | 0.905660 | 22 | 0.901089 | 0.863636 |
| 42 | 15 | 669 | 0.959843 | 0.964126 | 27 | 0.959652 | 1.000000 |
| 43 | 1 | 0 | undefined | undefined | 1038 | 0.004066 | 0.007707 |
| 43 | 2 | 0 | undefined | undefined | 28 | 0.093416 | 0.035714 |
| 43 | 3 | 0 | undefined | undefined | 14 | 0.172569 | 0.000000 |
| 43 | 4 | 0 | undefined | undefined | 16 | 0.237499 | 0.125000 |
| 43 | 5 | 1 | 0.332279 | 1.000000 | 10 | 0.308082 | 0.200000 |
| 43 | 6 | 7 | 0.369588 | 0.428571 | 7 | 0.368758 | 0.714286 |
| 43 | 7 | 8 | 0.434368 | 0.375000 | 9 | 0.428366 | 0.555556 |
| 43 | 8 | 30 | 0.502702 | 0.466667 | 7 | 0.485362 | 0.571429 |
| 43 | 9 | 43 | 0.559713 | 0.604651 | 1 | 0.558293 | 0.000000 |
| 43 | 10 | 48 | 0.637362 | 0.541667 | 9 | 0.646421 | 0.666667 |
| 43 | 11 | 58 | 0.700316 | 0.551724 | 5 | 0.696721 | 0.400000 |
| 43 | 12 | 71 | 0.768894 | 0.704225 | 10 | 0.767317 | 0.700000 |
| 43 | 13 | 80 | 0.835454 | 0.837500 | 13 | 0.835015 | 0.846154 |
| 43 | 14 | 182 | 0.907701 | 0.835165 | 21 | 0.900264 | 0.714286 |
| 43 | 15 | 751 | 0.964460 | 0.948069 | 49 | 0.975448 | 0.938776 |
| 44 | 1 | 0 | undefined | undefined | 1036 | 0.004287 | 0.005792 |
| 44 | 2 | 0 | undefined | undefined | 30 | 0.093352 | 0.100000 |
| 44 | 3 | 0 | undefined | undefined | 19 | 0.165647 | 0.000000 |
| 44 | 4 | 0 | undefined | undefined | 9 | 0.224584 | 0.111111 |
| 44 | 5 | 0 | undefined | undefined | 9 | 0.301684 | 0.222222 |
| 44 | 6 | 6 | 0.365004 | 0.333333 | 5 | 0.363270 | 0.400000 |
| 44 | 7 | 18 | 0.445841 | 0.611111 | 9 | 0.429550 | 0.555556 |
| 44 | 8 | 40 | 0.508624 | 0.400000 | 10 | 0.504720 | 0.300000 |
| 44 | 9 | 50 | 0.570178 | 0.560000 | 6 | 0.582332 | 0.500000 |
| 44 | 10 | 63 | 0.634232 | 0.444444 | 9 | 0.636618 | 0.444444 |
| 44 | 11 | 67 | 0.702476 | 0.656716 | 11 | 0.694174 | 0.727273 |
| 44 | 12 | 87 | 0.771387 | 0.804598 | 13 | 0.774426 | 0.846154 |
| 44 | 13 | 112 | 0.838583 | 0.883929 | 11 | 0.841600 | 1.000000 |
| 44 | 14 | 423 | 0.909482 | 0.936170 | 20 | 0.902490 | 0.800000 |
| 44 | 15 | 413 | 0.958034 | 0.970944 | 40 | 0.961398 | 0.975000 |

## Seed aggregation and uncertainty

All summaries are equally weighted arithmetic means and sample SD (n−1) across
three fixed checkpoints. They do not pool repeated predictions as new patients,
resample seeds or choose a winner. Seed SD is descriptive training variability,
not evidence of three independent cohorts.

| Metric | Equal mean | Sample SD | 95% interval for equal mean |
| --- | ---: | ---: | --- |
| macro_f1 | 0.788984 | 0.008620 | [0.754993, 0.819263] |
| automatic_af_sensitivity | 0.807018 | 0.048840 | [0.751462, 0.862573] |
| selective_risk | 0.143341 | 0.007832 | Not prespecified |
| accuracy | 0.856659 | 0.007832 | Not prespecified |
| coverage | 1.000000 | 0.000000 | Not prespecified |
| rhythm_output_coverage | 0.973677 | 0.003695 | Not prespecified |

| Seed | Macro F1 95% CI | Automatic AF sensitivity 95% CI |
| --- | --- | --- |
| 42 | [0.758044, 0.833319] | [0.675439, 0.833333] |
| 43 | [0.740229, 0.817247] | [0.745614, 0.885965] |
| 44 | [0.751211, 0.824425] | [0.789474, 0.912281] |

The prespecified uncertainty procedure used **10,000** class-stratified
exact-duplicate-group bootstrap replicates, NumPy Generator **PCG64(20261014)**,
canonical class order N,A,O,~, ascending group IDs, and the same sampled group
multiplicities for every seed. Each class resamples its original group count with
replacement and retains all sampled group members. Groups are label-homogeneous;
all holdout groups are singletons, so observed class counts stay fixed. Intervals
are 2.5th/97.5th percentiles with `np.quantile(method="linear")` for per-seed and
equal-mean macro F1 and automatic AF sensitivity. Group membership and all
replicate statistics are retained locally. No replicate was replaced and no
uncertainty fallback used. AF is present; the no-AF undefined rule was not invoked.

Intervals condition on this observed class mix and these three fixed checkpoints.
They do not account for unknown patient clusters, source shift, training-data
sampling, or the full training-seed distribution and may be overconfident for
repeated patients. No patient-level interval, significance test, multiplicity
adjustment, clinical claim or CI-based seed selection is justified.

## Provenance and artifact identities

The local ignored output root is
`outputs/final-evaluation/temporal-raw-v1/internal-holdout-v1/`.
The index covers 23 artifacts; the index itself is the 24th file and is hashed
separately below. All file membership and hashes were verified before report
preparation. The index includes per-case logits/ledgers, metrics, lifecycle,
configurations, registry, environment, shared bootstrap statistics and completion.
No such bulk output file is added to Git.

Artifact index SHA256: `2e423e6da684f8472b9ec59652574a2d265e5bfa86fb1afd2257b4752dc56370`.

| Local artifact relative to output root | SHA256 |
| --- | --- |
| `accounting-summary.json` | `8afd9762557fabebc688c1e0a5195b19a1e8f115dc8008014e34716150b28fe4` |
| `audit-log.json` | `99e41b43c26eb711b83181f00d94ee2b5b7971a6a296823ed8973c53b1b4420d` |
| `bootstrap-statistics.npy` | `1cf1babe5cd6cbfc2a0375b4e3dabd20757c651d0b2d484fcb837d22fa7deabd` |
| `bootstrap.json` | `786138968f7d3dba2531638496b44a856561dc15db4734c47122b3965d10c742` |
| `completion.json` | `b75014737572081ede04439e8745273f6131054d698b80cff183375bc4d9872b` |
| `eligible-cases.json` | `b52fa5baa846484f7a2ce84eff7889436d364b7635b6ce89e59eb2c09ed6108a` |
| `environment.json` | `077aac916c90e52e14fecdd71a5190855a334775a30ae27990f7581b8927cab8` |
| `execution-accounting.json` | `b64f12ce5c44731f338b0c58aebe84d6f75ba22600305a32c532b14508b6875e` |
| `input-registry.json` | `c30771ac8e321ff6478a51f347b70c37e650a532dff84e75a464ef3ebd6a9f79` |
| `original-config.yaml` | `04551d430e3ee3a4ec04c04e7128e36ebfa578c5a595963465050fc5eaefd463` |
| `prediction-seals.json` | `c9d5d83094341127264fa57cca059c6905b4ebdf82a7f0aa587dbc3c52b82fa1` |
| `resolved-config.json` | `1e06313b4817fcae550f5301acfff4a75bc77d352df44cb6d10df8acace59e6d` |
| `seed42/cases.json` | `d23097008711b3e1cf5c5459bcd495b2d6aa7e2735182d93734e456938d75e5f` |
| `seed42/logits.npy` | `7cb0814631ec0bdfbbe2d14a56301b2d63f68c2e841c9e98d41e670391f7a174` |
| `seed42/metrics.json` | `8e17fa503e376375dff6c344dbebfb0cc4be6249daa022dcf9a64df650ac5c8e` |
| `seed43/cases.json` | `c5bdb4f737cc6d0ff78d273b8438b720e897d5a4d37d768656596ce6ef86ff83` |
| `seed43/logits.npy` | `21f2da024911f98cbde197d4a10fed73f34be1630f46eaf4cc6701f7869df06d` |
| `seed43/metrics.json` | `0507418e0f40df3723c9925708a7824bd25ee5b67ef64140776ebc697c14eb3e` |
| `seed44/cases.json` | `af47a5f0d618c51557b43621df972d9ec252dece2efbd0b4bf99fe1ffab42f5c` |
| `seed44/logits.npy` | `a9f8c2aa33fb3ec510c5f5b934a5a07a01ca9ca81bb8c11df2e929c624c9cf2c` |
| `seed44/metrics.json` | `d884d9673ce74b442d38a9388de0328c203e4e529edc94d30a5654f51745895b` |
| `status.json` | `f3b35bf3bf9915a5e6df2aeeeb5130aff537a8ddf15daefbc174264d5f79f489` |
| `summary.json` | `b37cf0d937a9796c997303c3fec5b7c3f16b41f52c80db567bab035ef3801b39` |

Execution input identities (SHA256):

| Input | SHA256 |
| --- | --- |
| Authorized evaluation YAML | `04551d430e3ee3a4ec04c04e7128e36ebfa578c5a595963465050fc5eaefd463` |
| `docs/experiments/temporal-final-evaluation-v1.md` | `05edd7f610c89179a094dfdec7d398587ecde75df33278245208a60f2a01b69f` |
| `reports/final-evaluation-integrity-audit-v1.md` | `ccb17018dbc80ff36ff932bdc63203dff00475984c3bf30335a092b26210a5f3` |
| `reports/rhythm-multiseed-v1.json` | `b4b98cef0fedf2e1da7d1a904a86572c5434dd799a033a7625fd348632c5b3f8` |
| `splits/challenge2017-v3-seed42-v1/frozen.json` | `2c887a1934fc93f40861a87522b5b2b5aee310aa622335e7912beffc0e26c121` |
| `splits/challenge2017-v3-seed42-v1/holdout.csv` | `d7aad3212e0052f33ad86666a3844dee146f99cc584c972854caf91b331b0717` |
| `data/manifests/challenge2017-v3/records.csv` | `a390f3103023df679d08544ce6f4ca9f972fb59394901a78c3027de93f1d0560` |
| `data/manifests/challenge2017-v3/audit.json` | `2e85add08052656eed0b72ddf0bb40c1d7163ee131aa18d796625ffb78516b40` |
| `data/manifests/challenge2017-v3/provenance.json` | `d7d79603a51d5ef35f93f5d2ae55344f0b7a6fe567d8c46b29a31575c6c4b636` |
| `uv.lock` | `71231d200792985f74f8a2c9ebff633621ad86d6efcc59b2b0063b5a2301c86d` |
| `environments/cuda128/uv.lock` | `2c5ed7cb262d94c1e2d334af9955de796ddf43e8d8be73f0c32976cdc75474a1` |

The immutable input registry includes all remaining checkpoint/configuration/
preprocessing/training-report identities and SHA256 for every committed file at
the execution revision. Historical source and training dependencies remain in
[original multiseed provenance](rhythm-multiseed-v1.json). The recorded execution
revision and hashes distinguish frozen numerical sources from subsequent report
and documentation changes. Exact cross-hardware identity is not promised.

## Verification and preservation boundary

Report preparation verified the complete artifact index and membership, all
saved input/source hashes against execution-revision bytes, prediction seals,
1,279 unique canonical case IDs per seed, class supports, confusion/AF accounting,
zero deferrals/failures, per-class and probability metrics, reliability-bin
accounting, all primary intervals from saved replicate statistics, and seed
means/sample SD. Only existing outputs, metadata and committed source were read;
no raw ECG was opened, checkpoint deserialized, inference rerun or policy changed.
The previously recorded **273 passing synthetic/regression tests** are historical
implementation verification, not a newly executed experiment. Documentation
links, staged whitespace, trackability and repository-contract checks are also
part of this publication review.

A metadata-only publication regression check,
`tests/test_final_evaluation.py::test_fixed_config_authorization_and_policy`,
fails because it still asserts that the real YAML is false, while the preserved
execution authorization commit made it true. The historical 273-test result
predates that authorization change. This is a current CI blocker, disclosed in
the publication PR; this documentation-only milestone does not modify tests,
execution authorization or experimental policies. No inference was run by this
check.

**Ignored artifacts still need external archival. A Git report is not an artifact
backup.** No configured durable destination or complete external archive exists;
`completion.json` records `external_archival: pending`. Hashes detect changes but
cannot restore missing bytes. This commit/push preserves only compact aggregate
research documentation and provenance, not signals, model weights, predictions,
case ledgers or bulk run outputs. No output directory is overwritten or retried.

## Exposure, independence and interpretation limits

The [integrity audit](final-evaluation-integrity-audit-v1.md) remains authoritative.
All 8,528 source recordings, including eventual holdout signals, were decoded
before split freeze. Twenty previews were generated before assignment; **A00004
and A00022** subsequently entered holdout. Generation does not establish human
viewing, and available records cannot establish who viewed them or whether they
influenced choices. Public labels, class supports and signal-derived integrity
summaries were available; there is no complete immutable access log or proof of
blinding. Earlier absence of recorded holdout performance did not prove never-seen
status or rule out arbitrary-named, deleted, external or unlogged results.
Pre-freeze decoder/integrity decisions used the full dataset. Neither the two
exposed IDs nor any other eligible record was removed to make the cohort appear
cleaner.

Recording IDs and audited exact-duplicate groups are disjoint across frozen
train/validation/holdout. **All 8,528 subject IDs are unknown.** Verified
patient/subject, site/device/source subgroup, collection-time and longitudinal
linkage are unavailable. Near duplicates, repeated sessions and shared patients
remain unverified. Numeric record order is not patient identity or chronology;
duration is not collection time. The common public Challenge 2017/AliveCor source
does not establish cross-source independence. Recording bootstrap intervals are
not patient-independent uncertainty.

Validation informed early stopping, checkpoint/architecture selection and the
exploratory calibration/deferral study; this is not prospectively independent
model selection. Seeds evaluate the same recordings. These outcomes cannot
establish patient-independent, temporal, external, cross-source or clinical
generalization, safety or deployment readiness. Confidence is not signal quality,
and true ~ labels do not provide underlying rhythm or signal-quality ground truth.

**This holdout is consumed for future development. These results do not validate
quality-aware triage.** No checkpoint, architecture, preprocessing, threshold,
quality gate, routing, exclusion, uncertainty procedure or primary estimand is
tuned from these outcomes. Further quality-aware development needs separate,
independently sealed evaluation evidence after its choices are frozen. Re-splitting
the existing data or removing exposed records cannot restore an untouched test.
