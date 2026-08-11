# CXT-Select frozen evaluation protocol

## Purpose

CXT-Select is a zero-training selective-recognition evaluation built from the
already frozen F1 and F3 checkpoints. F1 remains the deployed prediction; F3
supplies a complementary disagreement signal. No risk head, temperature tuning,
score weighting, or checkpoint modification is allowed.

## Data boundary

The current track-level validation partition is used as exploratory development
evaluation because it has historically been used during method development.
Internal test, outer folds, and official test remain locked and are not read.
Cross-swap views are evaluation labels only and never enter the risk score.

## Frozen risk scores

1. F1 MSP risk: `1 - max softmax(F1 original logits)`.
2. F1 predictive entropy.
3. F1/F3 original-label disagreement.
4. Jensen-Shannon divergence between F1/F3 original probabilities.
5. Fixed-seed random review baseline.

## Events and selection

- `clean_error`: F1 original prediction differs from the target.
- `cross_swap_flip`: F1 original is correct and F1 cross-swap is wrong.
- `dar_flip`: F1 original is correct and F1 cross-swap predicts the donor class.

Cross-swap events are evaluated only on the F1-original-correct recipient set.
Risk-coverage uses standard selective risk (error rate among accepted samples),
not macro-F1 area under a curve. At 100%, 95%, 90%, and 80% coverage we also
report macro-F1, tail-F1, balanced accuracy, track-balanced accuracy,
cross-swap metrics, per-class coverage, and per-track coverage.

## Statistics

Each seed is reported separately. Paired bootstrap resamples `(species_id,
group_id)` trajectory clusters within species for 2,000 replicates using seed
3407. The primary exploratory comparison is JS disagreement versus F1 MSP.

## Interpretation boundary

The CXT-Select gate is relative and exploratory: it requires consistent JS
risk-coverage and context-event detection gains without materially degrading
clean selective performance or tail coverage. A failure closes this selective
extension; it does not invalidate the previously frozen F1 training result.
