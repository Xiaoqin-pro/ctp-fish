# DLE-Fish inner-fold frozen evaluation

## Scope and access boundary

This is the pre-registered two-fold inner evaluation of the four frozen DLE
variants. Both held-out sets are subsets of the original `train` partition and
are disjoint from their corresponding training groups. The evaluation did not
read current validation, internal test, calibration, outer folds, or official
test. No model was trained or tuned during evaluation.

## Metrics

The table reports the mean over the two group-disjoint inner folds. `DAR-flip`
is the conditional donor-attraction rate on samples that were correct on the
original view. The context-swap view is generated deterministically from the
inner held-out records only.

| Variant | Original macro-F1 | Tail F1 | Track-balanced accuracy | Foreground macro-F1 | Masked macro-F1 | Cross-swap macro-F1 | DAR-flip |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q0 | 0.968052 | 0.949348 | 0.991288 | 0.963942 | 0.971804 | 0.580601 | 0.203301 |
| Q1 | 0.935604 | 0.858536 | 0.988458 | 0.917863 | 0.947628 | 0.533302 | 0.178140 |
| Q2a | 0.950712 | 0.908465 | 0.988570 | 0.930523 | 0.951903 | 0.547739 | 0.195350 |
| Q2b | 0.956834 | 0.925613 | 0.989061 | 0.940251 | 0.955030 | 0.576533 | 0.175690 |

## Pre-registered comparison

Relative to Q0, Q1 loses 3.245 percentage points of original macro-F1 and
9.081 points of tail F1. Q2a recovers part of this loss but remains 1.734 and
4.088 points below Q0. Q2b is the closest DLE candidate, but still loses 1.122
points of original macro-F1 and 2.374 points of tail F1. None of the DLE
variants improves cross-swap macro-F1 over Q0. Q2b lowers DAR-flip, but that
change is accompanied by a substantial clean/tail accuracy loss and therefore
does not satisfy the Pareto rule.

## Decision

The constrained DLE-Fish pilot does **not** pass its inner-fold Pareto screen.
The masked classification and positive-evidence concentration objectives do not
provide a reliable improvement over the Q0 control under the frozen protocol.
No additional DLE loss, sampler, backbone, mask rule, or weight search is
authorized. The previously validated F1 foreground-sufficiency training remains
the main method baseline; DLE is retained as a negative constrained pilot.

## Provenance

- Seed: 3407.
- Folds: frozen group-disjoint inner folds 0 and 1.
- Evaluation output: `outputs/cxt_fish/phase2_dle/inner_evaluation.json`.
- Per-image outputs: eight `dle_inner_fold*_per_image.csv` files in the same local output directory.
- Current validation accessed: false.
- Internal test accessed: false.
- Calibration accessed: false.
- Outer folds accessed: false.
- Official test accessed: false.
