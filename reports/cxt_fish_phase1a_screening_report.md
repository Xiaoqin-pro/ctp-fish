# CXT-Fish Phase 1A Screening

## Scope

All eight frozen sampling/loss configurations were trained with seed 407 and evaluated on the fixed track-level validation partition only. The internal test and outer folds were not accessed.

## Fixed validation ranking

Ranking is lexicographic: tail F1, then track-balanced accuracy, then macro F1; an exact tie prefers the simpler configuration.

| Rank | ID | Sampler | Objective | Prior | Tail F1 | Track-balanced accuracy | Macro F1 |
|---:|---|---|---|---|---:|---:|---:|
| 1 | A1 | S1 | CE | none | 0.9703 | 0.9899 | 0.9707 |
| 2 | A4-I | S0 | Balanced Softmax | image-count | 0.9446 | 0.9871 | 0.9746 |
| 3 | A2 | S2 | CE | none | 0.9200 | 0.9840 | 0.9531 |
| 4 | A5-T | S1 | Balanced Softmax | track-count | 0.9137 | 0.9887 | 0.9641 |
| 5 | A3 | S3 | CE | none | 0.9057 | 0.9880 | 0.9535 |
| 6 | A5-I | S1 | Balanced Softmax | image-count | 0.8836 | 0.9874 | 0.9485 |
| 7 | A0 | S0 | CE | none | 0.8685 | 0.9843 | 0.9442 |
| 8 | A4-T | S0 | Balanced Softmax | track-count | 0.8012 | 0.9746 | 0.9124 |

## Three-seed replication selection

The top three configurations are **A1, A4-I, A2**. Each will be retrained with the three frozen replication seeds [3407, 2026, 17] before any internal-test access.
