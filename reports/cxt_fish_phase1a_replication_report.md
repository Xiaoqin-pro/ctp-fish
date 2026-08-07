# CXT-Fish Phase 1A Three-Seed Replication

## Scope

The three validation-selected configurations were independently retrained with frozen seeds 3407, 2026, and 17. Results below use only the fixed track-level validation partition; no internal test or outer fold was accessed.

## Aggregate validation results

| ID | Sampler | Objective | Prior | Tail F1 | Track-balanced accuracy | Macro F1 |
|---|---|---|---|---:|---:|---:|
| A1 | S1 | CE | none | 0.9427 ± 0.0198 | 0.9866 ± 0.0019 | 0.9619 ± 0.0111 |
| A2 | S2 | CE | none | 0.9052 ± 0.0069 | 0.9810 ± 0.0021 | 0.9486 ± 0.0015 |
| A4-I | S0 | Balanced Softmax | image-count | 0.9041 ± 0.0390 | 0.9842 ± 0.0039 | 0.9533 ± 0.0162 |

These are development results. The internal test remains locked pending method-component freezing.
