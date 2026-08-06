# MobileNetV3 transfer check

## Scope

This is the fixed post-confirmation transfer check for the frozen CXT-Fish
foreground-sufficiency intervention. It uses MobileNetV3-Large with ImageNet
1K V2 initialization, the same track-uniform sampler, foreground blur view,
optimizer, early-stopping rule, three outer folds, and the registered seed
3407. It is not a new method search and no hyperparameter was selected from
outer-test results.

## Integrity

- Training cells: 6/6 complete (MV0 and MV1 in folds 1--3).
- Outer evaluations: 6/6 complete after explicit `--unlock-outer-test`.
- All checkpoints were selected using the fold-local inner-development set.
- `official_test_accessed=false` in every evaluation output.
- No calibration, parameter search, or additional seed was used.

## Results

| method | original macro-F1 | foreground macro-F1 | same-swap macro-F1 | cross-swap macro-F1 | DAR-flip | agreement |
|---|---:|---:|---:|---:|---:|---:|
| MV0 | 0.958009 | 0.817540 | 0.926940 | 0.517315 | 0.217286 | 0.708903 |
| MV1 | 0.928605 | 0.919784 | 0.886056 | 0.565552 | 0.170024 | 0.759164 |
| MV1 - MV0 | -0.029404 | +0.102244 | -0.040884 | +0.048237 | -0.047261 | +0.050261 |

The direction is consistent with a clean--robustness trade-off in this
single-seed transfer check: foreground-sufficiency training improves
foreground and cross-class-swap behavior and lowers donor attraction, while
reducing ordinary original-view and same-swap macro-F1. These six cells are a
transfer diagnostic, not evidence of universal accuracy improvement or a
standalone claim about MobileNetV3.

## Decision use

The MobileNetV3 result is retained as a lightweight cross-architecture
transfer check. The main method claim remains the frozen ResNet18 outer result:
foreground-sufficiency training is a context-robustness intervention, not a
guaranteed clean-accuracy improvement. No further CXT-Fish variant or
MobileNetV3 tuning is authorized based on this result.
