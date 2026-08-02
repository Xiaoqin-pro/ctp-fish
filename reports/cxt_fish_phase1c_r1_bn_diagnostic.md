# CXT-Fish Phase 1C R1: BatchNorm-only diagnostic

## Question and frozen procedure

R1 tests whether the Phase 1C F3 context result is materially explained by
BatchNorm running statistics.  It is a diagnostic only: no model parameter,
classifier, loss, sampler, augmentation, or validation protocol is changed.

The frozen F3 seed-3407 checkpoint was processed once as follows:

1. Freeze every parameter and use no gradients or labels.
2. Set the model to evaluation mode, then set only BatchNorm modules to train
   mode.
3. Reset BatchNorm running statistics and recompute them cumulatively
   (`momentum=None`) over the original-RGB training records with the fixed
   deterministic evaluation transform.
4. Save an independent checkpoint and evaluate it with the same locked
   validation-only Phase 1C evaluator.

The pass used 18,681 train records in 292 batches.  It did not read the
internal test partition or the outer folds.  The generated checkpoint has
SHA-256 `15796FDA13A9F3B6C68CA1D363387030E7027939A164B5D68F725894A5AAEE32`.

## Results

| Metric | Frozen F3 | F3 + BN recalibration | Difference (R1 - F3) |
| --- | ---: | ---: | ---: |
| Original-view macro F1 | 0.9598 | 0.9338 | -2.60 pp |
| Original-view track-balanced accuracy | 0.9897 | 0.9830 | -0.66 pp |
| Foreground-view macro F1 | 0.9619 | 0.9413 | -2.05 pp |
| Cross-class context-swap macro F1 | 0.5991 | 0.5472 | -5.19 pp |
| Conditional donor-attraction rate (DARflip) | 0.1410 | 0.1714 | +3.04 pp |
| Prediction agreement under cross-class swap | 0.7992 | 0.7435 | -5.58 pp |
| Cross-swap macro-F1 drop | 0.3606 | 0.3866 | +2.60 pp |

## Decision

R1 shows no material recovery.  The BN-only pass degrades clean and
context-swapped behaviour and increases conditional donor attraction.
Therefore, under the preregistered rule, it is not replicated on the other
seeds and it does not authorize an R2 hyperparameter repair.  Phase 1C is
interpreted using its frozen F0--F3 replication results; this diagnostic is
retained as negative evidence that a simple BatchNorm-statistics explanation
does not account for the observed context-robustness pattern.
