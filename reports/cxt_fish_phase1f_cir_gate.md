# CIR-Fish Phase 1F zero-training Gate

## Decision

**FAIL — I1/I2 training is not authorized.** The frozen F1 checkpoint does
contain a weak donor-residual signal, but it does not reliably predict the
actual cross-class intervention and does not survive the fixed class and
shuffle controls.

## Frozen protocol

- checkpoint: `outputs/cxt_fish/phase1c_pilot/F1_seed3407/best.pt`
- partition: validation only (`4,124` recipients, `1,302` recipient tracks)
- pairs: one fixed cross-class, cross-track donor per recipient
- bootstrap: 1,000 recipient-track clusters, seed `3407`
- training, calibration, internal test and outer folds: not accessed
- raw RGB context swaps are interpreted as non-target context/interference
  exchange; they are not claimed to be pure fish-free backgrounds.

## Gate measurements

| Check | Result | Frozen criterion | Status |
|---|---:|---:|---|
| Overall residual donor-class AUROC | 0.6047 | >= 0.60 | pass |
| CIR score AUROC for real DAR-flip | 0.5649 | >= 0.60 | fail |
| q-CIR vs q-real Spearman | 0.1268 | > 0 | pass |
| True minus shuffle donor attraction | +0.00048 | CI lower > 0 | fail |
| Bootstrap 95% CI | [-0.00071, +0.00170] | lower > 0 | fail |
| Classes with residual AUROC >= 0.60 | 5/16 | >= 8/16 | fail |
| Valid pair fraction | 1.0000 | >= 0.95 | pass |

The injected recipient accuracy remained 0.9939 (shuffle 0.9942), while donor
attraction was only 0.00218 (shuffle 0.00170). The weak aggregate residual
signal therefore does not translate into a reliable, class-general or
shuffle-separated intervention mechanism.

## Interpretation and stop rule

The residual is not sufficiently specific to the donor's causal context. The
real-intervention alignment is weak and the paired bootstrap cannot distinguish
true residual pairing from a deterministic shuffle. Under the preregistered
Gate-0/Gate-3 rule, CIR-Fish is closed without training I1 or I2, without
adding another loss, and without opening any locked partition.

This does **not** invalidate the frozen CXT-Fish findings. The defensible final
paper route is the trajectory-aware reliability/context audit with F1 as the
main mitigation baseline, F2 as a negative ablation, F3 as an aggressive
robustness variant, and the failed TRAFS, donor-aware composite, and CIR gates
reported as bounded negative method experiments.
