# CXT-Fish construct-validity and statistics-repair protocol v1

This is a post-hoc, reviewer-motivated, inference-and-reanalysis-only package.
It does not authorize training, checkpoint selection, method selection, new
seeds, new backbones, new datasets, or access to the official TEST split.

The frozen F0/F1 checkpoints, the primary seed-3407 donor manifest, the five
donor-sensitivity manifests, and the nine F0-2RGB training cells are immutable.
The package repairs the F0-2RGB observed-point-estimate/statistical reporting,
audits donor-species weighting and donor foreground residuals, evaluates one
fish-suppressed donor sensitivity using the exact primary pairings, audits
Gate-0 split comparability, and records manuscript terminology/novelty limits.

## Fish-suppressed donor sensitivity

This analysis is named `post-hoc donor-subject-suppressed construct-validity
sensitivity`. It reuses each original recipient, donor, checkpoint, mask,
pairing, resize rule, and recipient feathering rule. The sole intervention is
to suppress the donor subject with the already frozen context-sanity rule:
binary official donor mask, one iteration of an elliptical 3x3 dilation,
OpenCV Telea inpainting with radius 3. No alternative inpainting rule or
parameter search is allowed.

All outcomes are reported. A positive result may support only that the effect
is not confined to visible donor-subject evidence; it does not prove natural
background robustness. A null or negative result narrows the claim to the
original conflicting donor-composite stress test.

## Statistical identity

The primary F0-2RGB observed effect is the equal-weight mean of the nine
paired cell differences, CXT-Fish minus F0-2RGB. The bootstrap mean is reported
separately from this observed point estimate. Bootstrap draws are paired,
ground-truth-species-stratified within fold, cluster-resampled by group, and
shared across the three seeds in each fold.

## Evidence wording

The main ResNet result is a `frozen post-development group-disjoint
evaluation`, not an independent confirmation or external validation. Frozen,
pre-specified seeds, group-disjoint, post-hoc, exploratory, and preregistered
endpoint are distinct labels and must not be conflated.
