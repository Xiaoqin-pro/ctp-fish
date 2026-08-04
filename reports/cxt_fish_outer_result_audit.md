# CXT-Fish outer result audit

This is a statistical and class-coverage audit of the frozen outer confirmation. It does not retrain models or alter any checkpoint/result.

- Frozen class count: 16
- Gate-0 metadata class count: 23
- Tail classes (fixed by descending frozen-train image count): 11, 12, 13, 14, 15
- Bootstrap: 1,000 paired resamples of trajectory groups, averaged across 9 fold-seed pairs
- Official TEST accessed: false

## Paired F1 - F0 bootstrap intervals

| Metric | 95% CI |
|---|---:|
| Original macro-F1 | [-0.0116, +0.0071] |
| Tail-F1 | [-0.0263, +0.0254] |
| Track-balanced accuracy | [-0.0002, +0.0019] |
| Same-class swap macro-F1 | [-0.0000, +0.0002] |
| Cross-class swap macro-F1 | [+0.0601, +0.0817] |
| DAR-flip | [-0.0383, -0.0327] |
| Swap disagreement | [-0.0479, -0.0400] |

## Fold heterogeneity

The fold-level table is stored in `experiments/cxt_fish_outer_fold_audit.csv`. Fold 3 is retained as a pre-specified heterogeneity audit, not removed or reweighted.

## Artifacts

- `experiments/cxt_fish_outer_class_track_audit.csv`: 23-class Gate-0 metadata versus frozen 16-class split image/track counts.
- `experiments/cxt_fish_outer_per_class.csv`: per-class F1 for every method, seed, fold and view.
- `experiments/cxt_fish_outer_fold_audit.csv`: fold-level clean macro-F1 summary.
