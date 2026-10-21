# Phase A: completed bounded CPU execution

Recorded 2026-10-21. This is reused-validation development evidence, not final-test or clinical validation. The earlier implementation verification and NSTDB failure/repair/readiness records remain unchanged.

## Outcome and execution identity

One attempt completed successfully from committed source `9e86a99b36a677d0e87156350c86be448207af44` on `feat/noise-phase-a`. All three clean reproduction gates passed before stress inference. All 13,200 inference batches and 165 condition/seed prediction arrays were published, sealed and verified before scoring. No retry, resume, tolerance change, export regeneration or alternate output root was used.

Runner lifecycle: 2026-10-21T05:17:28.860568+00:00 through 2026-10-21T07:42:30.841763+00:00 (10:47:28–13:12:30 IST). Supervisor wall time was 8,704.94 seconds (2 h 25 min 4.94 s); runner lifecycle was 8,701.98 seconds. Exit status was 0, cleanup completed, and there was no resource interruption or forced termination.

Approved budget: eight hours including five minutes for cleanup, two inference threads, 4 GiB aggregate runner/descendant RSS, 5 GiB generated output and at least 10 GiB free disk at launch. Peak sampled RSS was 1,222,500,352 bytes (1.139 GiB); peak observed run plus supervisor output was 860,186,929 bytes (0.801 GiB). Resource polling was once per second, with five-second progress snapshots. RSS can double-count shared pages; these observations do not measure subsecond peaks or provide kernel quotas.

The helper was committed as `d50a85e`; the subsequent authorization commit `9e86a99` changed only the new Phase A execution flag to true. Both original proposals remain false and unchanged. The retained true flag documents this attempt only; it grants no further execution authority. The fixed run and supervision roots are occupied.

## Inputs and protocol

The registry binds the exact input bytes and committed source used during this run. Preflight checked all 2,608 bound artifacts against the frozen contract, including the three temporal candidates, training-only preprocessing states, training metadata, historical CPU validation exports, NSTDB inputs and validation artifacts. The policy canonical hash excluding the authorization field was `32222409868f6b055baa3fc262bc90f2129c8b2910747338ca3e2a52f25f22ea`.

The pinned environment was Python 3.11.16, NumPy 2.2.6, SciPy 1.15.3, PyTorch 2.8.0+cpu, WFDB 4.3.1 and PyYAML 6.0.3. Inference used two CPU threads, deterministic settings and float32; corruption/scoring used float64. No dependencies were installed or upgraded.

Seeds 42, 43 and 44 each used the same 1,280 validation recordings (N=762, A=114, O=362, ~=42), one clean condition and all 54 stress conditions: bw/ma/em × 24/18/12/6/0/−6 dB × three repetitions. The 211,200 eligible case entries are repeated evaluations, not independent people. Training, calibration, GPU execution, quality models, BUT and consumed-holdout access were excluded.

## Clean reproduction

All gates had exact record IDs, labels, predictions and confusion matrices, and zero maximum absolute logit difference. Maximum absolute probability differences were 1.6517449830111275e-7, 1.5676233378059123e-7 and 1.6630813470808192e-7 for seeds 42/43/44. They passed the frozen combined atol=1e-7 and rtol=1e-5 test; probability sums passed atol=1e-6. This is not a claim that every absolute probability difference was below 1e-7.

| Seed | Macro F1 | Automatic AF detections / all AF | Four-class errors / eligible | Rhythm outputs / eligible |
|---|---:|---:|---:|---:|
| 42 | 0.814211 | 88/114 | 180/1280 | 1246/1280 |
| 43 | 0.795386 | 93/114 | 192/1280 | 1250/1280 |
| 44 | 0.805254 | 94/114 | 187/1280 | 1238/1280 |

Equal-seed clean means (sample SD): macro F1 0.804951 (0.009416), automatic AF sensitivity 80.4094% (2.8198 percentage points), four-class error 14.5573% (0.4709 points).

## Complete primary stress grid

Each cell first averages the three repetition metrics within each seed, then weights the three seeds equally. Deltas subtract that same seed’s clean metric before averaging. No predictions or confusion matrices were pooled to obtain these means. AF and error columns are percentages; their deltas are percentage points.

| Noise | SNR dB | Macro F1 | Δ F1 | AF sensitivity % | Δ AF pp | Error % | Δ error pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| bw | 24 | 0.803701 | -0.001250 | 79.9220 | -0.4873 | 14.5486 | -0.0087 |
| bw | 18 | 0.801866 | -0.003085 | 79.2398 | -1.1696 | 14.5573 | -0.0000 |
| bw | 12 | 0.800233 | -0.004717 | 78.2651 | -2.1442 | 14.6528 | +0.0955 |
| bw | 6 | 0.775082 | -0.029869 | 75.2437 | -5.1657 | 15.7813 | +1.2240 |
| bw | 0 | 0.700239 | -0.104712 | 67.7388 | -12.6706 | 19.9219 | +5.3646 |
| bw | -6 | 0.534410 | -0.270541 | 46.5887 | -33.8207 | 34.7569 | +20.1997 |
| ma | 24 | 0.804217 | -0.000734 | 79.6296 | -0.7797 | 14.4358 | -0.1215 |
| ma | 18 | 0.789920 | -0.015031 | 77.9727 | -2.4366 | 14.9826 | +0.4253 |
| ma | 12 | 0.723820 | -0.081131 | 73.0019 | -7.4074 | 18.4375 | +3.8802 |
| ma | 6 | 0.584191 | -0.220759 | 57.5049 | -22.9045 | 32.9253 | +18.3681 |
| ma | 0 | 0.399833 | -0.405117 | 34.8928 | -45.5166 | 58.2812 | +43.7240 |
| ma | -6 | 0.195149 | -0.609802 | 15.6920 | -64.7173 | 82.4392 | +67.8819 |
| em | 24 | 0.804493 | -0.000458 | 79.1423 | -1.2671 | 14.5313 | -0.0260 |
| em | 18 | 0.803882 | -0.001068 | 78.7524 | -1.6569 | 14.5399 | -0.0174 |
| em | 12 | 0.789971 | -0.014980 | 78.1676 | -2.2417 | 15.1562 | +0.5990 |
| em | 6 | 0.731997 | -0.072954 | 73.0019 | -7.4074 | 18.2465 | +3.6892 |
| em | 0 | 0.574669 | -0.230282 | 53.8012 | -26.6082 | 32.8733 | +18.3160 |
| em | -6 | 0.276389 | -0.528561 | 15.4971 | -64.9123 | 69.9653 | +55.4080 |

All conditions had 100% assessed coverage (1,280/1,280 per seed/condition), zero technical failures, zero deferrals and zero flat/inapplicable cases. AF sensitivity counts only correct automatic AF predictions divided by all 114 AF recordings. Wrong AF predictions count as missed detections. Predicted `~` is an assessed noisy-class output, not a deferral or a negative AF classification.

Severe artifacts degraded these fixed models: at −6 dB muscle artifact, mean error was 82.4392% and macro F1 0.195149; at −6 dB electrode motion, AF sensitivity was 15.4971%. These are descriptive outcomes over the full prescribed grid, not a selection of a winning seed or evidence for augmentation benefit.

## Paired uncertainty and secondary findings

The saved shared bootstrap used 10,000 PCG64 draws, seed 20261016, stratified by N/A/O/~ and sampling frozen duplicate groups. All 1,280 validation duplicate groups were singletons; verified patient identities were unavailable. Each draw retained pairing across conditions, repetitions and model seeds. Linear 2.5/97.5 percentiles describe uncertainty conditional on these fixed models, plans and reused validation data.

| Noise/SNR | Δ F1 95% interval | Δ AF sensitivity 95% interval (pp) | Δ error 95% interval (pp) |
|---|---:|---:|---:|
| bw/24 | [-0.0056, +0.0030] | [-1.1696, +0.0975] | [-0.1736, +0.1563] |
| bw/18 | [-0.0086, +0.0021] | [-2.3392, -0.2924] | [-0.2257, +0.2257] |
| bw/12 | [-0.0131, +0.0040] | [-3.8986, -0.7797] | [-0.2517, +0.4516] |
| bw/6 | [-0.0452, -0.0138] | [-8.0897, -2.6316] | [+0.5903, +1.8924] |
| bw/0 | [-0.1272, -0.0803] | [-16.8616, -8.8694] | [+4.2274, +6.5104] |
| bw/-6 | [-0.2984, -0.2413] | [-39.6686, -28.1676] | [+18.3594, +22.0660] |
| ma/24 | [-0.0061, +0.0049] | [-1.8519, +0.0000] | [-0.3212, +0.0781] |
| ma/18 | [-0.0266, -0.0030] | [-4.2885, -0.8772] | [+0.0174, +0.8594] |
| ma/12 | [-0.1005, -0.0597] | [-10.4288, -4.5809] | [+3.0295, +4.7396] |
| ma/6 | [-0.2455, -0.1942] | [-27.9727, -18.0312] | [+16.8490, +19.8524] |
| ma/0 | [-0.4342, -0.3758] | [-51.6569, -39.4737] | [+41.7708, +45.6163] |
| ma/-6 | [-0.6397, -0.5790] | [-71.0551, -58.1871] | [+66.0938, +69.6267] |
| em/24 | [-0.0057, +0.0047] | [-2.2417, -0.4873] | [-0.2606, +0.2083] |
| em/18 | [-0.0080, +0.0059] | [-3.4113, -0.0975] | [-0.3472, +0.3212] |
| em/12 | [-0.0291, +0.0004] | [-4.5809, -0.0975] | [+0.0087, +1.1979] |
| em/6 | [-0.0954, -0.0487] | [-11.5984, -3.6062] | [+2.6302, +4.7483] |
| em/0 | [-0.2582, -0.2005] | [-33.1384, -20.3704] | [+16.4323, +20.1999] |
| em/-6 | [-0.5592, -0.4966] | [-71.3450, -58.2846] | [+53.2726, +57.4740] |

The companion [aggregate JSON](noise-phase-a-v1-results.json) preserves unrounded means/sample SD, every seed/repetition primary result, paired seed deltas, absolute and paired bootstrap intervals, clean confusion/AF outcomes and probability/routing summaries. Its 165 condition rows retain NLL, four-class Brier, top-label ECE, binary AF Brier, false AF counts, true-noisy-to-A counts and rhythm-output counts. Full per-class, confusion, reliability-bin and within-record repetition evidence remains in the indexed local metrics rather than case-level Git exports.

Four-class Brier sums squared errors over all four classes; ECE uses the frozen 15 bins. Binary AF metrics include only true N/A/O (1,238 cases), excluding 42 true-noisy recordings. False AF on true N/O uses 1,124 cases; false AF share among AF predictions has its own prediction denominator. No confidence or quality threshold was selected.

## Integrity, checks and local evidence

Run root: `outputs/noise-robustness/noise-robustness-v1/phase-a`. External logs/reviews: `outputs/noise-phase-a-run-v1`. All hashes below are SHA256 of exact file bytes, not canonical JSON hashes.

| Run artifact | Bytes | SHA256 |
|---|---:|---|
| `accounting.json` | 51421443 | `72d3bb2f77162fbce31fc55ec118a77fe621c9d193e96d973ecaab30d4628130` |
| `artifacts-sha256.json` | 2892425 | `341ff870205cb5164443bdbda02439b0e71a6751ad6d28f9554d0969ed9e97e1` |
| `bootstrap-draws.npy` | 102400128 | `20198394b8bd78f758b2ac3e19023b0b81eabe411b9ff90a3c8cdf220d2b6c08` |
| `bootstrap-statistics.npy` | 140160128 | `48a2187bb0a6ca0b30f0dcb3888fb3a2207515bd1d758293e3544b860e5c558c` |
| `bootstrap.json` | 583480 | `c72f948b9500b4432e981dfc5e162bb6ea46342d980654a158a76a55e744f35c` |
| `metrics.json` | 8483921 | `3c59b8a7b1b5d75c880103ce87b8f046f1d410364f23731678151b1992980464` |
| `plans.json` | 18730498 | `7bbfab28408445be3164f032e80b620172b336ca33da650c33a1d0292c0f13a0` |
| `registry.json` | 366582 | `ac6f7b31931f79cfd472ed8e3c5f0f347ab7dc26562ab77e3ea0c94b5bdfa179` |
| `seal.json` | 2945217 | `5252af2af3526e725a13fd6825dd0e9460d6254f765238c2b553051527e9a0ab` |
| `status.json` | 501 | `bc9eb5332cfd19da1bee9074414b643cb2da0b007bed135ad912f0eccd42fff6` |

Post-run review verified the completed lifecycle, the status-bound artifact-index hash, the exact 26,577 indexed file inventory and every indexed byte hash. It verified all registry-bound input/source bytes at completion before subsequent report/test edits. The sealed-prediction validator rechecked all 165 arrays and required plans, gates and ledger/batch seals. Independent confusion/accounting checks matched every seed/condition. Saved bootstrap arrays had the required shapes, finite statistics and original per-class group multiplicities; no bootstrap or inference was rerun.

Saved corruption metadata matched all 207,360 stress plan entries. Original physical digests were stable across conditions, and corrupted physical/post-transform digests paired across model seeds. Maximum recorded achieved-SNR error was 7.105427357601002e-15 dB; flat/inapplicable entries were zero. This review checked saved metadata, not an additional signal decode.

One initial post-run verification script incorrectly used alphabetically sorted JSON keys for the prediction condition axis. Its assertion failed; the verifier was corrected to the frozen condition ordering and then passed. No run artifact or numerical result changed.

Local `integrity-review.json`, `corruption-review.json`, `verify-completion.py`, `launch.json`, `progress.json`, `samples.jsonl` and the empty `runner.log` have exact byte identities in the companion aggregate JSON. The supervisor sampled only metadata/stat information for progress, not prediction arrays.

Actual repository verification: full synthetic/regression suite 466 passed before execution (28.18 s) and again after the focused fixture correction (27.19 s); supervisor tests 9 passed (1.14 s); Ruff lint/format, repository contracts, frozen-manifest verification (17 identities; 5,969/1,280/1,279 split rows, zero ECG files opened), dependency compatibility (58 packages), both offline lock checks and sdist/wheel build passed. Final documentation/diff checks are recorded with the completion commit/PR.

The authorization commit exposed two CI tests that assumed the real execution flag would always remain false. After the real run completed, their fixtures were narrowed to explicitly generated disabled configurations; production runner/backend and frozen numerical policy were unchanged. The completed experiment remains attributed to revision `9e86a99`, not the later documentation/test commit.

## Limitations and next dependency

Validation reuse and historical candidate selection limit generalization. Patient identifiers and independent noise-volunteer identities are unavailable; temporal roles do not make independent noise sources. Fixed seeds/plans and conditional bootstrap intervals omit model-development and population uncertainty. Retaining source rhythm labels under severe corruption is an unverified recognizability assumption. Native NSTDB voltage calibration remains unverified. These findings establish neither clinical adequacy nor a new final-cohort result.

Signals, checkpoints, predictions, case ledgers, bulk bootstrap arrays, acquisition diagnostics and run logs remain ignored/local-only. Only this report and compact aggregate evidence are tracked. Durable archival and restoration remain pending an approved destination; no upload was made.

The next task is a separately authorized Phase B implementation with synthetic verification only. Training/experiment execution, quality-model work, BUT and the consumed holdout remain excluded. The retained Phase A flag does not authorize another attempt.
