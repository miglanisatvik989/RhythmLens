# Methods and research limits

## Data and preprocessing

The frozen recording split has 5,969 training, 1,280 validation, and 1,279 holdout
records. Labels are N (normal), A (AF), O (other rhythm), and ~ (too noisy to classify).
Verified duplicate groups stay together. Patient identifiers are unavailable;
these splits do not establish patient independence.

Signals retain native 300 Hz sampling and physical mV units. Per-record median
centering is followed by a fixed scale: the median training-record RMS.
Fit only on all training records. Flat signals remain centered zeros.
There is no per-record amplitude normalization, resampling, or new class weighting.

## Models

| Version | Architecture | Default receptive field |
| --- | --- | --- |
| `rhythm-cnn-v1` | Reference; stem 32, stages 32/64/128 | 81 samples / 0.27 seconds |
| `rhythm-dilated-v1` | Temporal candidate; stem 16, widths 16/24/32 | 3,075 samples / 10.25 seconds |

The reference uses stem kernel 15 and three two-convolution residual stages with
kernel 7 and strides 1,2,2. Masked global averaging aggregates local features and
discards their order.

The candidate uses eight two-convolution blocks at native rate with fixed
dilations 1,2,4,8,16,32,64,128. Its architectural receptive field is
15 + 2 × 6 × 255 = 3,075 samples; learned effective dependence can be smaller.
Both validate prefix masks and mask after convolutions.

Padding and pulse-spacing tests are engineering evidence. Comparing these models
changes capacity, depth, and context together, not just temporal context.

## Evaluation

The fixed protocol uses AdamW, learning rate 0.001, weight decay 0.0001, batch size 16,
training dropout 0.1, maximum 50 epochs, and patience 8. Validation four-class macro
F1 selects checkpoints. Smoke may override dropout and is not performance evidence.

Report support, precision, recall, F1, confusion matrix, and accuracy. Undefined
class precision/recall/F1 are assigned zero. Automatic AF sensitivity is correct
automatic AF predictions divided by all AF-labeled records. Future deferred AF
records remain in that denominator.

The recorded baseline classifier assesses every eligible record: coverage 100%, implemented
deferrals zero. Predicting the noisy class is not an implemented deferral mechanism.
Probabilities are uncalibrated softmax.

The original baseline results are checkpoint-selected validation evidence. The
[multi-seed protocol](experiments/rhythm-multiseed-v1.md) reports all seeds without
best-seed selection. The completed [calibration/confidence-deferral analysis](../reports/temporal-calibration-deferral-v1.md)
uses validation-derived roles and is exploratory because those recordings already
informed model selection. Its null undefined metrics and explicitly named legacy
zero-fill macro F1 remain distinct. Confidence is not signal quality.

The [completed internal rhythm-only evaluation](../reports/temporal-internal-rhythm-evaluation-v1.md)
uses all three fixed temporal checkpoints equally, with calibration, deferral and
quality gating disabled. Mean automatic AF sensitivity is 80.70%, automatic
four-class error 14.33%, and assessed coverage 100%. Predicted ~ cases remain
assessed outcomes, separately reported from rhythm outputs and deferrals. The
[exposure audit](../reports/final-evaluation-integrity-audit-v1.md) documents
pre-freeze decoding, two holdout preview exposures, public labels and unknown
patient/source/time independence. Bootstrap intervals condition on the observed
class mix and fixed checkpoints, without accounting for unknown patient clusters.
This is qualified internal rhythm-only evidence, not clinical validation or
validation of quality-aware triage. **The holdout is consumed for future
development.** Signal-quality assessment remains pending; future quality-aware
choices must be frozen on development evidence and evaluated separately. Ignored
artifacts still need external archival; a Git report is not an artifact backup.

CPU deterministic resume has synthetic regression coverage. Exact mid-epoch
continuation, hardware-independent identity, and deterministic CUDA resume
equivalence are not promised.
