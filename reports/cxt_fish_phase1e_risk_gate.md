# TRAFS-Fish Phase 1E risk Gate

## Frozen evaluation

The F0/A1 checkpoint was used without training on the locked validation split.
Risk was computed from the original-versus-fixed-foreground margin gap, with
`image_risk = 0` whenever the original prediction was incorrect.  Track risk
was the median image risk; within-class track risk was its average-tie rank.
The F0 validation context-swap predictions were read only for this Gate.

## Result

| Gate quantity | Value |
| --- | ---: |
| Validation images | 4,124 |
| Validation tracks | 1,302 |
| Image-risk AUROC for swap failure | 0.4748 |
| Track-risk AUROC | 0.4686 |
| Within-class track-risk AUROC | 0.4686 |
| Low-risk swap failure | 0.2851 |
| High-risk swap failure | 0.2371 |
| Low-risk DARflip | 0.2102 |
| High-risk DARflip | 0.1811 |

The high-risk group is not worse: its swap-failure and DARflip rates are both
lower. Image-risk AUROC is below the preregistered 0.60 threshold, and the
overall high-minus-low swap-failure difference is -4.80 percentage points.

## Decision

The zero-training Gate fails.  The original-versus-foreground margin gap is
not a usable predictor of the frozen model's cross-class context failure on
this validation split.  Therefore E1 and E2 are not trained, no additional
seeds are run, and no risk-weight or loss search is permitted.  TRAFS-Fish is
closed as a method attempt; the repository retains F1 and the Phase 1C
trajectory/context evaluation as the final evidence base.
