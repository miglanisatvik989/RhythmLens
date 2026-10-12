# RhythmLens multiseed v1 results

All six runs completed: reference and temporal seeds 42, 43, 44. Four additional runs were executed sequentially on the existing cuda128 environment, GPU 0, from committed inputs `05d8cebf4af570aaf2ab40d80c7c715b573be534`. No run failed or required resume. Seed42 artifacts were reused and preserved. The sandbox-only initial doctor could not see CUDA; rerunning doctor outside the sandbox passed before any training output was created.

These are checkpoint-selected validation results on the same 1,280 recordings (N=762, A=114, O=362, ~=42), trained on 5,969 recordings. Seed42 informed this follow-up; it was not planned prospectively before seed42. The [frozen protocol](../docs/experiments/rhythm-multiseed-v1.md) defines settings and analysis.

| Architecture | Seed | Accuracy | Macro F1 | F1 N | F1 A | F1 O | F1 ~ | AF precision | AF recall / automatic sensitivity | AF correct/all AF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rhythm-cnn-v1 | 42 | 0.673438 | 0.566096 | 0.788831 | 0.451220 | 0.413223 | 0.611111 | 0.740000 | 0.324561 | 37/114 |
| rhythm-cnn-v1 | 43 | 0.705469 | 0.618601 | 0.813213 | 0.610687 | 0.444444 | 0.606061 | 0.540541 | 0.701754 | 80/114 |
| rhythm-cnn-v1 | 44 | 0.674219 | 0.576804 | 0.794902 | 0.571429 | 0.337711 | 0.603175 | 0.548387 | 0.596491 | 68/114 |
| rhythm-dilated-v1 | 42 | 0.859375 | 0.814211 | 0.905072 | 0.830189 | 0.784741 | 0.736842 | 0.897959 | 0.771930 | 88/114 |
| rhythm-dilated-v1 | 43 | 0.850000 | 0.795386 | 0.908060 | 0.775000 | 0.748485 | 0.750000 | 0.738095 | 0.815789 | 93/114 |
| rhythm-dilated-v1 | 44 | 0.853906 | 0.805254 | 0.909209 | 0.783333 | 0.766571 | 0.761905 | 0.746032 | 0.824561 | 94/114 |

Arithmetic mean and sample standard deviation (n-1 denominator), computed directly from observed reports. All metrics derive from the same 1,280 assessed validation cases, with their respective class denominators preserved. Accuracy uses 1,280; class recall uses true class support; class precision uses predicted class count. Class F1 is 2TP/(true support + predicted count), and macro F1 is the arithmetic mean of the four class F1 values. Sample SD is sqrt(sum((x - mean)^2)/(n - 1)); aggregates use unrounded per-run values, without pooling repeated recordings across seeds into independent cases.

| Architecture (n) | Metric | Mean | Sample SD |
| --- | --- | ---: | ---: |
| rhythm-cnn-v1 (3) | accuracy | 0.684375 | 0.018272 |
| rhythm-cnn-v1 (3) | macro_f1 | 0.587167 | 0.027744 |
| rhythm-cnn-v1 (3) | F1_N | 0.798982 | 0.012693 |
| rhythm-cnn-v1 (3) | F1_A | 0.544445 | 0.083088 |
| rhythm-cnn-v1 (3) | F1_O | 0.398460 | 0.054877 |
| rhythm-cnn-v1 (3) | F1_~ | 0.606782 | 0.004017 |
| rhythm-cnn-v1 (3) | AF_precision | 0.609643 | 0.112961 |
| rhythm-cnn-v1 (3) | AF_recall | 0.540936 | 0.194637 |
| rhythm-dilated-v1 (3) | accuracy | 0.854427 | 0.004709 |
| rhythm-dilated-v1 (3) | macro_f1 | 0.804951 | 0.009416 |
| rhythm-dilated-v1 (3) | F1_N | 0.907447 | 0.002136 |
| rhythm-dilated-v1 (3) | F1_A | 0.796174 | 0.029751 |
| rhythm-dilated-v1 (3) | F1_O | 0.766599 | 0.018128 |
| rhythm-dilated-v1 (3) | F1_~ | 0.749582 | 0.012537 |
| rhythm-dilated-v1 (3) | AF_precision | 0.794029 | 0.090094 |
| rhythm-dilated-v1 (3) | AF_recall | 0.804094 | 0.028198 |

Paired differences: temporal minus reference (absolute proportions).

| Seed | Macro F1 difference | AF recall difference |
| ---: | ---: | ---: |
| 42 | +0.248115 | +0.447368 |
| 43 | +0.176785 | +0.114035 |
| 44 | +0.228450 | +0.228070 |

| Architecture | Seed | Best epoch | Stop reason | Total seconds | Train/eval seconds | Parameters | Smoke steps |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| rhythm-cnn-v1 | 42 | 23 | early_stopping_validation_macro_f1 | 849.291 | 840.362 | 242244 | 66 |
| rhythm-cnn-v1 | 43 | 48 | epoch_limit | 1368.885 | 1360.546 | 242244 | 40 |
| rhythm-cnn-v1 | 44 | 18 | early_stopping_validation_macro_f1 | 724.460 | 716.287 | 242244 | 48 |
| rhythm-dilated-v1 | 42 | 31 | early_stopping_validation_macro_f1 | 2104.353 | 2096.450 | 97548 | 84 |
| rhythm-dilated-v1 | 43 | 42 | early_stopping_validation_macro_f1 | 2731.634 | 2723.443 | 97548 | 28 |
| rhythm-dilated-v1 | 44 | 39 | early_stopping_validation_macro_f1 | 2568.098 | 2559.730 | 97548 | 54 |

Total seconds is the training command wall duration including preprocessing and in-command checkpoint/evaluation checks; it excludes separate doctor, smoke and CPU evaluation commands. Durations are observations, not controlled speed benchmarks.

Full training retained batch 16, AdamW 0.001 / weight decay 0.0001, dropout 0.1, 50 epochs maximum and patience 8, validation four-class macro F1 selection. Smoke used four training recordings, batch 4, dropout 0.0, unchanged 200-step/1,200-second budgets, accuracy 1.0 and loss <=0.1 targets. All six recorded smokes passed target and checkpoint reload checks. Smoke outcomes are engineering diagnostics, not validation performance.

Every run assessed all 1,280 eligible validation recordings: coverage 100%, deferrals 0/1,280 (0%), reasons {}. All 114 AF recordings per run are in the automatic AF sensitivity denominator; correct counts are above. Accepted-case metrics equal all-assessed metrics because none were deferred. There is no quality-based deferral policy; deferral is never treated as a negative AF prediction.

| Architecture | Seed | Configuration | YAML SHA256 | Resolved configuration SHA256 | Source revision | Source bundle SHA256 |
| --- | ---: | --- | --- | --- | --- | --- |
| rhythm-cnn-v1 | 42 | config.yaml | 6054ca4a58ef683f8843918839c19ab0b4173eba6bc03ec74fd29b28ed63212a | 35b43f15258cd570bd135c8cfb7a486e5eacb6718da1802c84cea0febaaf18fc | 4754c324578c3022b0e213afb210ffc37556bb77 | 939e099ef4e62a161c67515bb5d3cde5947add4054727deadb475f1f1afc963c |
| rhythm-cnn-v1 | 43 | configs/reference-cuda128-seed43.yaml | 7021626ddff486900d98bde5c46441f33447bb47a02e86d312222f35a5db95d8 | 1df0308a7f1e258e6fdc2daa20cf83ca355cb7afbf24f0d8d0fb5c7724dce596 | 05d8cebf4af570aaf2ab40d80c7c715b573be534 | 58a59b557754cd15bdffa9c5aac78eb9a93a87525ead89445d27f1acb5678f9f |
| rhythm-cnn-v1 | 44 | configs/reference-cuda128-seed44.yaml | 78856d0a440df3ae83d90c1f784707b561a60a4f58b58fd68dfc136f37f86a70 | 434ce86c866d79dc2e2767b2d08ac75f9d698a473e60f32bc3a32918746a23d3 | 05d8cebf4af570aaf2ab40d80c7c715b573be534 | 902e0e4c7e9a492be7bb43060e9d67d68a856bd4b033ffc79cebf80c69807319 |
| rhythm-dilated-v1 | 42 | configs/temporal-cuda128.yaml | df2bc25e8fc7fd62654cde86efd2fe4946ecab43cdd698f2f6cff364e590503b | 2ff2e965d2f1f09e9bb4919e3d7fb413e8164d3dae788bb44a853163f77d7a61 | cc0317e8389895ce094b5521b6f40986ac6ae5ed | 939e099ef4e62a161c67515bb5d3cde5947add4054727deadb475f1f1afc963c |
| rhythm-dilated-v1 | 43 | configs/temporal-cuda128-seed43.yaml | 00f07878fc338420eb7e2d7708aa3e8addea73874d5436148f0da7e989e573ac | 9d4f8d648b8658b45f15fb925267563cb13f1861ffba3700cbc378fb73f8fdc4 | 05d8cebf4af570aaf2ab40d80c7c715b573be534 | 87c1c2d8023d32f3e9cb67c0439636d22b549c8d01dc85b60a7fcb3e230078aa |
| rhythm-dilated-v1 | 44 | configs/temporal-cuda128-seed44.yaml | a931d40ab2ea9fdae0e3508b4b4c555f0c9744eb2c0e14a8493f5032bee1f3bb | cd954668cc7ca393da6d041b307102477133cd3a4b4745f47eebb6a94442b0e5 | 05d8cebf4af570aaf2ab40d80c7c715b573be534 | d844a635fb966af3317a953ea8a7eddf427b5e910a42f151c8ee7f06ab3a1dff |

Full precision metrics, report/checkpoint hashes, environment identities, exact installed training dependency versions, split identity and preprocessing identity are in [the compact JSON summary](rhythm-multiseed-v1.json). Source bundles differ because they include configuration files; the implementation, decoder, environment descriptors and both locks are unchanged across all six runs. Each run records clean canonical source and the expected CUDA runtime. CPU evaluation preserves immutable training provenance.

| Active lock | SHA256 |
| --- | --- |
| uv.lock | 71231d200792985f74f8a2c9ebff633621ad86d6efcc59b2b0063b5a2301c86d |
| environments/cuda128/uv.lock | 2c5ed7cb262d94c1e2d334af9955de796ddf43e8d8be73f0c32976cdc75474a1 |

Shared preprocessing SHA256: `a4719420f5b867e3870386b488c954fb85f1b695f2264f57235cf0e0b80783b6`. Training-fitted fixed RMS scale: 0.18008390945457742 mV; 5,969 fitting records, native 300 Hz, per-record median centering. No validation fitting or resampling. Shared split assignment SHA256: `fb27f37a8f9dcd1d8931116e252290a68be29844b5b81badc3223954184054c7`.

Verification passed: configuration validation (30 tests) and readiness contracts (41 tests), repository ignore/lock contracts, 17 frozen snapshot hashes and partition counts, 99 original-file hashes, unchanged seed42 artifacts/configs, clean canonical source, CUDA doctor and bounded smoke for each new run, checkpoint reload, saved best-checkpoint CPU validation evaluation, exact class metrics/confusion agreement and close loss agreement (absolute/relative tolerance 1e-5). All six lifecycle states are completed. Prediction IDs and true labels exactly match the frozen validation manifest; training and separate evaluation predictions agree on IDs, true labels and predicted labels. Confusion matrices were independently counted from predictions, and metric formulas recomputed from those matrices; means, sample SDs and paired differences use unrounded observed values. All run, evaluation and training preflight reports explicitly record holdout_signals_accessed=false. Final whitespace and Git scope checks passed. Local outputs, predictions, weights and ECG remain ignored.

Three seeds assess training variability, not patient independence. Patient independence remains unverified with recording-level splits. The holdout remains untouched (1,279 recordings); only frozen metadata integrity/accounting was inspected. Calibration and quality-based deferral remain pending. This comparison changes capacity, widths and temporal structure together and does not isolate temporal context or a causal architectural mechanism. It does not establish clinical readiness. No statistical significance is claimed from three seeds.

Recommendation: retain both architectures as recorded comparators and use all three observed seeds when deciding the next development step. The temporal candidate has higher macro F1 and AF recall in every paired seed; prioritize it for a separately specified development follow-up, subject to the unresolved independence and calibration/quality limitations. No quality gating, calibration, tuning or holdout evaluation was started.
