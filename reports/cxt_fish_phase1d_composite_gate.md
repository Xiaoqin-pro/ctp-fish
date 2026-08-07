# CXT-Fish Phase 1D donor-composite Gate

## Frozen automatic checks

The training-only donor manifest contains 18,681 recipient--donor pairs and
has SHA-256 `6343d370dfea280839ece65d54f655a92d30366f2411c34fa45a3574695555d2`.
The stratified Gate selected 50 pairs. All 50 had matching image/mask sizes,
and none had a same-species or same-trajectory donor.

## Visual Gate result: FAIL

The automatic checks are not sufficient. Visual inspection found that official
Fish4Knowledge masks can cover one annotated donor fish while leaving another
visible fish outside that mask. For example, donor
`fish_03/fish_003388315223_26053.png` contains a smaller fish to the left of the
masked central fish. Telea inpainting correctly removes the masked fish but
cannot remove the unannotated fish, which remains in the composite.

This violates the preregistered requirement that the donor contribution be a
fish-free background. Therefore the generated manifest is **not authorized
for D1 or D2 training**. No donor-aware model was trained, no validation
metric was inspected, and no locked partition was accessed.

## Consequence

The current Fish4Knowledge single-fish masks do not establish a reliable
fish-free donor-background pool. Proceeding would turn the proposed
background-conflict intervention into an uncontrolled multi-fish intervention,
invalidating the donor-label interpretation. This Phase 1D method attempt is
closed under the frozen Gate rule; it may only be reconsidered with new,
complete multi-fish annotations or an independently validated fish-removal
source, not by tuning the present inpainting parameters.
