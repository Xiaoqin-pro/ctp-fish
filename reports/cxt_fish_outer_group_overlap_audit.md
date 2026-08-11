# Historical-development and outer-test group overlap audit

This is a split-structure audit only. It reads the frozen historical
`f4k16t_track_level_dev.csv` and the three frozen outer-test manifests. It does
not read model predictions, select a method, retrain a model, or access the
official Fish4Knowledge TEST set.

## Frozen counts

| Source | Unique groups |
|---|---:|
| Historical train | 6076 |
| Historical validation | 1302 |
| Historical internal test | 1302 |
| Outer fold 1 test | 2894 |
| Outer fold 2 test | 2893 |
| Outer fold 3 test | 2893 |

## Historical-group presence in each outer-test fold

| Historical source | Fold 1 groups | Fold 2 groups | Fold 3 groups |
|---|---:|---:|---:|
| Historical train | 2019 (69.77%) | 2022 (69.89%) | 2035 (70.34%) |
| Historical validation | 428 (14.79%) | 436 (15.07%) | 438 (15.14%) |
| Historical internal test | 447 (15.45%) | 435 (15.04%) | 420 (14.52%) |

All 1302 historical-validation groups occur in at least one outer-test fold;
the same is true for all 1302 historical-internal-test groups. This is
expected because the historical split and the outer folds are partitions of
the same corpus. Each individual outer fold remains group-disjoint internally,
but the collection of outer folds is not an independent corpus relative to
historical development.

## Interpretation

The final result is therefore described as a **frozen post-development,
group-disjoint confirmation analysis**. It must not be described as a fully
blind nested evaluation, an unseen external test, or a
method-selection-independent test set.
