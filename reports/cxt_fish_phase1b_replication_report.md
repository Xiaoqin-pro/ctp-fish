# CXT-Fish Phase 1B three-seed replication

## Scope

C0/C1/C2 were evaluated only on the frozen track-level validation partition for seeds 3407, 2026, and 17. Internal test and outer folds were not accessed.

## Per-seed validation metrics

| Seed | Variant | Macro F1 | Tail F1 | Track-balanced accuracy |
|---:|---|---:|---:|---:|
| 17 | C0 | 0.9496 | 0.9090 | 0.9720 |
| 17 | C1 | 0.9483 | 0.9051 | 0.9736 |
| 17 | C2 | 0.9465 | 0.9008 | 0.9834 |
| 2026 | C0 | 0.9575 | 0.9355 | 0.9796 |
| 2026 | C1 | 0.9498 | 0.9180 | 0.9772 |
| 2026 | C2 | 0.9483 | 0.9032 | 0.9814 |
| 3407 | C0 | 0.9529 | 0.9117 | 0.9797 |
| 3407 | C1 | 0.9589 | 0.9182 | 0.9833 |
| 3407 | C2 | 0.9580 | 0.9209 | 0.9838 |

## Paired mechanism comparisons

| Comparison | Metric | Mean delta | SD | Positive seeds |
|---|---|---:|---:|---:|
| c1_minus_c0 | macro_f1 | -0.0010 | 0.0069 | 1/3 |
| c1_minus_c0 | track_balanced_accuracy | +0.0009 | 0.0031 | 2/3 |
| c1_minus_c0 | tail_f1 | -0.0049 | 0.0120 | 1/3 |
| c2_minus_c1 | macro_f1 | -0.0014 | 0.0004 | 0/3 |
| c2_minus_c1 | track_balanced_accuracy | +0.0049 | 0.0046 | 3/3 |
| c2_minus_c1 | tail_f1 | -0.0055 | 0.0088 | 1/3 |

## Frozen decision

- Decision: **retain for Phase 1C base**.
- The core C2-vs-C1 comparison is interpreted from direction consistency across the three registered seeds; it is not an internal-test or outer-fold result.
- C0 remains the structured-batch control. A1 is an earlier natural training baseline and is not treated as a one-variable comparison with C0.
- No context-control module, prototype loss, memory bank, new backbone, internal test, or outer-fold evaluation was used in this phase.
