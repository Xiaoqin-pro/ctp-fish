# CXT-Fish group-weighted robustness sensitivity

This post-hoc analysis uses only frozen outer per-image predictions and manifests; it does not train or re-infer models and does not access the official Fish4Knowledge TEST partition.

The primary manuscript cross-class composite macro-F1 remains class-balanced but image-weighted within species. Here, each recorded group receives equal weight within species, and species then receive equal weight. The nine fold–seed cells are averaged with equal weight.

| Metric | F0 mean | F0 SD | CXT-Fish mean | CXT-Fish SD | Delta (pp) | Delta SD (pp) | Favourable cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| group_balanced_accuracy | 0.6979 | 0.0278 | 0.7426 | 0.0161 | 4.47 | 3.13 | 8/9 |
| species_group_balanced_accuracy | 0.5271 | 0.0156 | 0.5944 | 0.0269 | 6.73 | 2.58 | 9/9 |

The 16 × 3 fold–species table contains 48 strata. The number of recorded groups per stratum ranges from 5 to 1414; this finite-cluster structure is reported explicitly rather than regularized.

All rows record official_test_accessed=false.