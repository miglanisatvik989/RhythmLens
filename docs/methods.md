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

Recorded results are checkpoint-selected validation evidence, not final holdout
or clinical validation. The [multi-seed protocol](experiments/rhythm-multiseed-v1.md)
reports all seeds without best-seed selection. A separate
[calibration/confidence-deferral workflow](experiments/temporal-calibration-deferral-v1.md)
is implemented and synthetically tested; real fitting and assessment are unauthorized
and pending. It uses null for undefined metrics, with an explicitly named legacy
zero-fill macro F1 for comparison. Signal-quality gating and final holdout evaluation
remain pending. Freeze model, thresholds,
quality rules, and analysis choices before accessing holdout signals.

CPU deterministic resume has synthetic regression coverage. Exact mid-epoch
continuation, hardware-independent identity, and deterministic CUDA resume
equivalence are not promised.
