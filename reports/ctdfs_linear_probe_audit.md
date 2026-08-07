# CT-DFS frozen-representation audit

This audit follows the corrected 16-class CT-DFS pilot. For each frozen
checkpoint, the ResNet18 encoder was frozen and a fresh 512-to-16 linear
classifier was trained with the same seed, optimizer, batch size and 20
epochs. Only the frozen train and validation partitions were used.

| checkpoint | linear-probe macro-F1 | linear-probe balanced accuracy |
|---|---:|---:|
| F1 seed 3407 | 0.970798 | 0.957454 |
| P1 seed 3407 | 0.971114 | 0.964500 |
| P2 seed 3407 | 0.966458 | 0.961081 |

P2 is below P1 by 0.47 percentage points in macro-F1 and 0.34 points in
balanced accuracy under the common linear probe. Therefore the small P2
original-head gain does not establish a superior frozen representation. It is
consistent with a head/training interaction, while the cross-swap degradation
seen in the fixed-16 evaluation remains unexplained by a better representation.

The result is an audit of mechanism, not a new model-selection step. It does
not authorize TAP-Fish, PCGrad, BN variants, or additional hyperparameter
search. Internal test and outer folds remain locked.

- schema: `ctdfs_fixed16_linear_probe_v1`
- seed: `3407`
- probe epochs: `20`
- train samples: `18681`
- validation samples: `4124`
- feature dimension: `512`
- `official_test_accessed`: `false`
- `outer_folds_accessed`: `false`
