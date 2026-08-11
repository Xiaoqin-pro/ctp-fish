# Gate-0 split comparability audit

This is a frozen, no-training audit of the existing image-level and group-disjoint development manifests. Splits were not changed and the observed performance difference is not decomposed causally.

## Partition sizes

| protocol | split | images | groups | classes |
|---|---|---:|---:|---:|
| image-level | test | 4070 | 2734 | 16 |
| image-level | train | 18993 | 7269 | 16 |
| image-level | val | 4070 | 2704 | 16 |
| group-disjoint | test | 4328 | 1302 | 16 |
| group-disjoint | train | 18681 | 6076 | 16 |
| group-disjoint | val | 4124 | 1302 | 16 |

## Training/evaluation configuration

- Architecture: ResNet-18
- Initialization: ImageNet
- Input: 224 px
- Optimizer: AdamW
- Learning rate: 3e-4
- Weight decay: 1e-4
- Maximum epochs: 40; early stopping patience: 7
- Seeds: 3407, 2026, 17
- Checkpoint metric: protocol-local test metrics were not used for selection; existing Gate-0 metadata is retained as the source.

## Interpretation

The two protocols yield materially different performance estimates. Any difference reflects the complete protocol change, including partition composition and group crossing, and is not interpreted as a causal estimate of same-group crossing alone.

- Official TEST accessed: false.
