# CXT-Fish CLIB-style fairness baseline protocol v1

## Purpose

This is a post-confirmation external baseline, not a new CXT-Fish variant.
The CXT-Fish F0/F1 outer results are already frozen. This run asks whether a
paper-faithful subject--background contrastive baseline changes the same
track-disjoint and context-swap metrics under the identical outer protocol.

Until every implementation detail is verified against the published method and
its official code, the result must be named **Subject--Background Contrastive
Baseline (CLIB-style)**, not a claim of exact CLIB reproduction.

Reference: Liu et al., *CLIB: Contrastive Learning of Ignoring Background for
Underwater Fish Image Classification*, Frontiers in Neurorobotics (2024),
doi:10.3389/fnbot.2024.1423848.

## Frozen data and evaluation boundary

- Dataset: the frozen F4K-16T metadata and three outer, group-disjoint folds.
- Runs: 3 folds × seeds {3407, 2026, 17} = 9 runs.
- ResNet18, ImageNet initialization, 224×224 input.
- The fold's inner development split is used only for checkpoint selection with
  the already frozen patience-7 rule. It is not a configuration search set.
- Each outer-test fold is read once, after all training for that fold is done.
- Calibration and official TEST remain locked; no output is overwritten.

## View construction

The baseline uses the published CLIB-style image views rather than the
official fish masks used by CXT-Fish:

1. original RGB image;
2. subject view: deterministic center crop with ratio 0.25;
3. background view: deterministic four-corner mosaic with ratio 0.25.

The term *background* means the non-central-crop context supplied by this
construction; it does not assert that every pixel is fish-free. No mask is
used to construct training views, and no random crop or flip is introduced by
the view builder beyond the frozen base augmentation.

## Training and inference

The contrastive stage uses a 512→256→ReLU→128 projection head with L2
normalization, temperature 0.07, and the fixed multi-view InfoNCE objective.
The original and same-sample subject views form the positive pair; all other
views in the batch are negatives. The classifier stage uses the frozen global
pooling representation and cross-entropy. Exact optimizer, epoch and
checkpoint settings are in `configs/cxt_fish_clib_style_outer_v1.yaml`.

At evaluation, only the ordinary original RGB image is provided to the
classifier. The projection head and training-only views are not part of the
inference path.

## Required implementation checks before training

1. View generation is deterministic and has no mask, validation or test read.
2. Positive/negative masks match the declared same-sample rule and contain no
   empty-positive anchors.
3. Train, inner-dev and outer-test group IDs are disjoint.
4. A checkpoint cannot be selected from outer-test or current validation.
5. A dry-run proves that all nine run manifests contain the protocol SHA,
   split SHA, view-construction SHA and initialization SHA.
6. No CLIB result is inspected to change the protocol, loss, temperature,
   learning rate, epoch count or seed list.

## Reporting

The baseline must use the same evaluator and report original, foreground,
same-class-swap and cross-class-swap macro-F1, tail-F1, track-balanced
accuracy, DAR, DAR-flip and prediction agreement, with paired trajectory/group
bootstrap intervals. Results are reported even if they are weaker than
CXT-Fish. This experiment cannot reopen method search.

