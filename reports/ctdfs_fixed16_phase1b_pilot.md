# CT-DFS fixed-16-class Phase 1B pilot

## Protocol correction

The first CT-DFS pilot used a 23-output classifier because it inferred the
vocabulary from the full metadata table. The frozen F4K-16T train/validation
protocol contains 16 classes. Those earlier absolute metrics are invalid and
are retained only as an audit trail; they are not used for scientific claims.

This rerun uses the committed `splits/f4k16t_class_ids.json` vocabulary for
both P1 and P2. Both variants use seed 3407, the same train/validation split,
the same ResNet18 dual-stream path, and the same fixed evaluation program.
Internal test and outer folds were not accessed.

## Fixed-16 results

| metric | P1 | P2 | P2 - P1 |
|---|---:|---:|---:|
| original macro-F1 | 0.969257 | 0.972195 | +0.002938 |
| original balanced accuracy | 0.961011 | 0.965527 | +0.004516 |
| original track-balanced accuracy | 0.988348 | 0.988104 | -0.000243 |
| foreground macro-F1 | 0.971048 | 0.979256 | +0.008209 |
| cross-swap macro-F1 | 0.573912 | 0.543391 | -0.030521 |
| cross-swap track-balanced accuracy | 0.679569 | 0.741131 | +0.061562 |
| DAR-flip | 0.217667 | 0.195360 | -0.022307 |

P2 therefore provides a small clean/balanced-accuracy gain and lower
DAR-flip, but it substantially lowers pixel-level cross-swap macro-F1. The
effect is mixed rather than a clear CT-DFS win. This single-seed pilot does
not authorize TAP-Fish or any additional architecture search.

## Reproducibility

- class protocol commit: `3bca074`
- resume/RNG fix: local commit `e340ab1` (push pending network retry)
- seed: `3407`
- evaluation partition: frozen `val`
- `internal_test_accessed`: `false`
- `outer_folds_accessed`: `false`
- prior 23-class pilot: invalid for absolute comparison
