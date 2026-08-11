# CXT-Fish Phase 1B single-seed pilot

## Scope

This report compares C0/C1/C2 with seed 3407 on the fixed track-level validation partition only. Internal test and outer folds were not accessed.

## Metrics

| ID | Macro F1 | Tail F1 | Track-balanced accuracy |
|---|---:|---:|---:|
| C0 | 0.9529 | 0.9117 | 0.9797 |
| C1 | 0.9589 | 0.9182 | 0.9833 |
| C2 | 0.9580 | 0.9209 | 0.9838 |

## Mechanism comparisons

- C1 - C0: tail F1 +0.0064; track-balanced accuracy +0.0036; macro F1 +0.0061.
- C2 - C1: tail F1 +0.0027; track-balanced accuracy +0.0006; macro F1 -0.0009.
- C2 exceeds C1 on 7/16 classes and 11/1302 validation tracks.

This is a single-seed development result. It determines whether reserved seeds should be run; it is not an internal-test or outer-fold claim.
