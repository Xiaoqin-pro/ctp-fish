# CXT-Fish final statistical synthesis

This synthesis is generated only from frozen outer-evaluation JSON/CSV
outputs. No model was trained or re-inferred during finalization.

## Main interpretation

- ResNet18 foreground-sufficiency training is a context-robustness
  intervention, not a clean-accuracy improvement method.
- The frozen ResNet18 cell means are: F0 original macro-F1 0.957721, F1
  0.955602 (−0.21 pp); F0 cross-swap macro-F1 0.518933, F1 0.591293
  (+7.24 pp); F0 DAR-flip 0.214595, F1 0.179226 (−3.54 pp).
- The pooled species-stratified group bootstrap gives a 95% interval of
  [0.0624, 0.0859] for the ResNet18 cross-swap difference F1−F0.
- MobileNetV3 is an architecture-sensitivity supplement: original macro-F1
  changes from 0.958009 to 0.928605 (−2.94 pp), while cross-swap changes from
  0.517315 to 0.565552 (+4.82 pp).
- Route C is a fixed two-stage mechanism control, not an exhaustive or
  faithful CLIB reproduction.

## Bootstrap protocol

All 5,000-replicate intervals use ground-truth species stratification and
`(fold, group_id)` clustering. For ResNet18, all registered seeds belonging to
the same sampled group are combined before resampling; seeds are not treated
as independent clusters. Official TEST remains locked.

## Generated files

- `experiments/cxt_fish_final_method_summary.csv`
- `experiments/cxt_fish_resnet_outer_bootstrap_5000.json`
- `experiments/cxt_fish_mobilenet_per_fold_results.csv`
- `experiments/cxt_fish_mobilenet_bootstrap_5000.json`
- `experiments/cxt_fish_view_gap_summary.csv`
