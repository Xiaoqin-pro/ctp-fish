# CXT-Fish Phase 1C: context-controlled learning pilot

## Status and scope

Phase 1B is frozen at commit `7943716`. Its C2 cross-track contrast mechanism is retained only as an auxiliary candidate: it consistently improves track-balanced accuracy, but it does not improve macro or tail F1. Phase 1C therefore uses the frozen A1 track-uniform CE model as its primary basis. Internal test and outer folds remain locked.

## Fixed primary experiment

All newly trained variants use ResNet-18 with ImageNet initialization, A1 global track-uniform sampling, unchanged augmentation/optimizer/learning-rate/epochs/early stopping, and seed 3407. Inference uses ordinary RGB only.

| Variant | RGB CE | Foreground-view CE | Context consistency |
|---|---:|---:|---:|
| F0 | yes; frozen A1 checkpoint reused | no | no |
| F1 | yes | yes | no |
| F2 | yes | no | yes |
| F3 | yes | yes | yes |

The foreground view is generated online as `mask * image + (1-mask) * GaussianBlur(image)`, using the official mask, fixed Gaussian parameters, and fixed mask-edge feathering. `foreground_ce_weight=1.0`; `context_consistency_weight=0.1`. Consistency uses the L2-normalized ResNet-18 512-dimensional pooled feature, with foreground feature stop-gradient: `1 - cosine(z_original, stop_gradient(z_foreground))`. No projection head is added.

F1/F2/F3 concatenate original and foreground views into one forward pass, so their BatchNorm treatment is identical. F1 vs F0 evaluates the full foreground-view training mechanism; F3 vs F1 is the clean comparison for the additional consistency loss.

## Frozen validation-only context evaluation

The deterministic manifest at `splits/f4k16t_phase1c_context_swap_val.csv` contains at most one same-class/cross-track and one cross-class/cross-track donor per validation recipient. Composition happens after fixed resize on the recipient canvas, before normalization, retains the recipient fish with its feathered official mask, and fills its background with resized donor RGB. Donors never share the recipient trajectory; cross-class donors also differ in species.

Report original, foreground-only, same-class-swap, and cross-class-swap macro F1; head/mid/tail F1; track-balanced accuracy; per-class and per-track values; prediction agreement; donor attraction rate (DAR); and conditional donor attraction rate among originally correct recipient predictions (DAR_flip). Report all absolute metrics and deltas.

## Decision sequence

Run F1, F2, F3 sequentially for seed 3407 only. Then stop and report: F1 vs F0, F2 vs F0, F3 vs F0, and F3 vs F1. Only if the frozen context mechanism shows an interpretable robustness signal will the two remaining registered seeds be considered. Only after that decision may `C2 + selected context configuration` be tested for complementarity.

## Prohibitions

Do not alter Phase 1A/B results or C2 parameters. Do not access internal test or outer folds. Do not implement prototypes, memory banks, new backbones, multiple foreground generators, or train-time context-swap selection.
