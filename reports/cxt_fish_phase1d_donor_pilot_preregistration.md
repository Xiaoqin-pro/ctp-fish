# CXT-Fish Phase 1D: Donor-aware counterfactual pilot

## Scope

Phase 1D tests one mechanism suggested by the frozen Phase 1C diagnosis:
whether a donor-class-specific ranking constraint adds value beyond ordinary
cross-class background swapping.  Internal test and outer folds remain locked.

## Frozen data construction

Every train recipient receives one static donor selected uniformly by donor
class, then uniformly by donor trajectory, then uniformly by frame.  Donor and
recipient must have different species and different trajectory IDs.  The donor
fish is removed by the Gate-0 frozen Telea procedure (3x3 elliptical dilation,
one iteration, radius 3) before resize.  The recipient is composited with its
fixed feathered official mask (radius 3).  Pairing and image synthesis occur
before normalization; the manifest SHA-256 is recorded before training.

## Models

`D0` is the frozen F1 checkpoint, re-evaluated with the Phase 1D evaluator.
`D1` and `D2` are independently trained from the same ImageNet initialization,
seed, sampler, optimizer, augmentations, epochs, and stopping rule as F1.

\[L_{D1}=L_{CE}(x_r,y_r)+L_{CE}(x_r^{fg},y_r)+L_{CE}(x_{r\leftarrow d},y_r)\]

\[L_{D2}=L_{D1}+0.1\max(0,s_{y_d}(x^{swap})-s_{y_r}(x^{swap}))\]

The margin is zero.  D2 must be compared with D1, not D0, to establish a
donor-specific effect.

## Gate and decision

Before training, 50 stratified train composites will be checked for donor-fish
residuals, recipient truncation, mask seams, and path/size mismatches.  This
gate cannot inspect model scores or alter synthesis parameters.

Single-seed order: D1 then D2 with seed 3407.  D1 must improve cross-class swap
macro F1 and reduce DARflip without material clean/tail harm to continue.  D2
earns the two additional frozen seeds only if versus D1 it improves cross-swap
macro F1 by about one percentage point, reduces DARflip by about one point,
does not reduce clean or tail macro F1 by more than 0.5 points, improves several
classes/tracks, and has a nonzero interpretable activation rate.  Otherwise the
donor-aware route stops with no loss, margin, sampler, or backbone search.
