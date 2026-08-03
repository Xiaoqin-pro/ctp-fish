# CT-DFS Phase 1 pilot result

## Decision

**CT-DFS is closed after the preregistered single-seed pilot.** P2 improves
ordinary validation accuracy, but it does not improve the two measures that
would establish the intended track/context mechanism: track-balanced accuracy
and cross-class context-swap performance. No additional seed and no P-class
ablation are authorized.

## Protocol

- seed: `3407`
- P1: original S1 + independent foreground S1
- P2: original S1 + independent foreground S3 class/track-uniform
- same ResNet18, concatenated two-stream forward, CE objective, 40-epoch cap,
  patience 7
- P1 stopped at epoch 12; P2 stopped at epoch 16
- validation-only evaluation; internal test and outer folds were not accessed

## P2 minus P1

| Metric | P1 | P2 | Difference |
|---|---:|---:|---:|
| Original macro-F1 | 0.66191 | 0.66950 | **+0.76 pp** |
| Original balanced accuracy | 0.94398 | 0.97513 | **+3.12 pp** |
| Track-balanced accuracy | 0.98767 | 0.98473 | **-0.29 pp** |
| Cross-class swap macro-F1 | 0.39704 | 0.38717 | **-0.99 pp** |
| DAR-flip | 0.20147 | 0.19124 | -1.02 pp |

The shared evaluator declares 23 class IDs while several validation classes have
zero support; the absolute macro-F1 values are therefore not interpreted as
standalone headline scores. The paired differences use exactly the same
labels, validation images and evaluator for P1 and P2.

## Interpretation

S3 foreground exposure successfully made class exposure more balanced, but the
redistribution did not produce the planned independent gain in trajectory
generalization or cross-context robustness. The improvement in ordinary
balanced accuracy is insufficient to support CT-DFS as a method contribution,
because the context-swap result moved in the wrong direction and
track-balanced accuracy declined slightly.

The final CXT-Fish route therefore remains the frozen F1 foreground-sufficiency
baseline plus the trajectory/context reliability audit. P1/P2 are retained as
transparent CT-DFS controls; no further method variant is introduced.
