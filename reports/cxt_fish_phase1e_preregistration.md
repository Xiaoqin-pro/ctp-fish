# TRAFS-Fish Phase 1E

Phase 1D donor-aware background intervention is permanently closed at commit
`6bda46e` after its data-semantic Gate failed.  Phase 1E does not use donor
backgrounds or extra annotations.

The frozen F0/A1 model computes, on train only,
`image_risk = I(margin_orig > 0) * max(0, margin_orig - margin_fg)`.  A track
risk is the median over its images.  Within each species, positive-risk tracks
receive average-tie percentile weights; zero-risk tracks receive zero, and
positive weights are normalized to mean one over training tracks.

The zero-training validation Gate compares image risk, track risk, and
within-species track risk against cross-class swap failure, DARflip, and
cross-swap margin drop.  It requires high-risk groups to be worse than low-risk
groups, image-risk AUROC at least 0.60, and directionality across multiple
species.  If the Gate fails, no E1/E2 training is authorized.

If it passes, E1 is F1 plus uniform asymmetric margin transfer and E2 is the
same model with the normalized track-risk weight.  The original margin is
detached, the loss has no feature consistency, C2 contrast, donor background,
or inference-time component.  E2 is compared with E1; only a meaningful,
multi-class improvement authorizes additional seeds.
