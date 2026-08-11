# F0-2RGB statistical repair

This is an exploratory post-hoc mechanism-control reanalysis from frozen per-image predictions. No model was trained or re-inferred.

The superseded JSON used the bootstrap distribution mean as `point_estimate`. The observed point estimate is instead the equal-weight mean of the nine paired cell differences.

- Observed CXT-Fish minus F0-2RGB cross-composite effect: **+6.83 pp**.
- Bootstrap mean: +6.83 pp.
- 5,000-replicate percentile interval: [+5.71, +7.87] pp.
- Resampling: ground-truth-species-stratified group clusters within fold; paired methods and all three seeds share each fold draw.
- `official_test_accessed`: false.

The former `experiments/cxt_fish_rgb2_vs_f1_bootstrap_5000.json` is retained as a superseded statistical artifact and is not used for manuscript numbers.
