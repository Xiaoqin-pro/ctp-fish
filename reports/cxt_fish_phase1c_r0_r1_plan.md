# CXT-Fish Phase 1C R0/R1 execution plan

## Fixed evidence boundary

The seed-3407 pilot is frozen at `c19c3d3`: F3 improved cross-class context-swap macro F1 and reduced donor attraction, while reducing original-view macro F1. This is a single-seed signal, not a final method result. No frozen Phase 1A/1B/1C result will be overwritten or reinterpreted.

## R0: registered-seed replication

Train F1, F2, and F3 sequentially for the already registered seeds `2026` and `17`, using exactly `configs/cxt_fish_phase1c.yaml`. The queue order is seed 2026 F1/F2/F3, then seed 17 F1/F2/F3. It must not access internal test or outer folds. Report all original/foreground/same-swap/cross-swap metrics, DAR, DAR_flip, tail F1, and track-balanced accuracy, with three-seed means, standard deviations, and positive-seed counts versus F0.

## R1: BN-only diagnostic, only after R0

After R0, run original-domain BatchNorm recalibration for F3 seed 3407 without updating weights, labels, or non-BN model parameters. Reset BN running statistics, use cumulative statistics (`momentum=None`), feed a deterministic original-RGB train pass under `no_grad`, restore module momentum, and write a new independent F3-BNR checkpoint. Evaluate it with the same frozen validation manifest. If this diagnostic materially recovers original-view performance while preserving the robustness signal, apply the identical fixed procedure to both replication F3 checkpoints; otherwise do not select a favorable seed.

## Prohibitions

Do not modify alpha, mu, blur, mask feathering, sampler, model architecture, donor manifest, test/outer-fold access, or the Phase 1C evaluator during R0/R1. Do not begin the targeted R2 repair until R0 and R1 are frozen.
