# CXT-Fish final statistical synthesis

This synthesis is generated only from frozen outer-evaluation JSON/CSV outputs. No model was trained or re-inferred.

## Primary interpretation

- On the frozen ResNet18 post-development, group-disjoint confirmation analysis, foreground-sufficiency training is interpreted as a context-robustness intervention rather than a clean-accuracy improvement method.
- MobileNetV3 shows average robustness improvement with heterogeneous cross-swap effects and a larger clean-accuracy cost; it is architecture-sensitivity evidence only.
- Route C is retained as a fixed two-stage mechanism control and is not treated as an exhaustive contrastive-learning comparison.

## Bootstrap protocol

The primary estimand is an equal-weight mean of paired cell differences. Each replicate resamples groups within fold-by-species strata, shares the same draws across paired methods and registered seeds, computes macro-F1 separately for every fold-seed cell, and averages the resulting cell differences. Official TEST remains locked.

The earlier pooled-prediction aggregation is retained only as a supplementary sensitivity analysis and is not paired with the primary cell-mean estimate.

## Generated files

- `experiments/cxt_fish_final_method_summary.csv`
- `experiments/cxt_fish_resnet_outer_bootstrap_5000.json`
- `experiments/cxt_fish_mobilenet_per_fold_results.csv`
- `experiments/cxt_fish_mobilenet_bootstrap_5000.json`
- `experiments/cxt_fish_view_gap_summary.csv`
