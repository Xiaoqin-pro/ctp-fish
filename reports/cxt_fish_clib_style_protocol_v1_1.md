# Protocol-adapted Subject--Background Contrastive Baseline v1.1

This document supersedes the training instructions in v1 while preserving v1
as a historical record. The baseline is intentionally called **CLIB-style**,
not an exact CLIB reproduction: it matches CXT-Fish's ResNet18, image size,
track-disjoint outer folds, labeled-data budget, optimizer budget and
checkpoint rule.

The published CLIB method is used only as the design reference for three
views. The exact crop formula and official implementation are not available
in this repository, so the fixed rule here is explicitly a deterministic
CLIB-inspired interpretation and must not be presented as author-code
reproduction.

## Frozen runs and data boundary

- 3 outer folds × seeds {3407, 2026, 17} = 9 runs.
- ResNet18 ImageNet1K_V1, 224×224, same frozen F4K-16T manifests as CXT-Fish.
- Outer-train labels are fully matched to CXT-Fish.
- Inner-dev is used only for patience-7 checkpoint selection.
- Outer-test is read once after all training; calibration and official TEST stay locked.
- No method, augmentation, temperature, learning rate or seed search is allowed.

## Fixed views

Each RGB image creates three deterministic pre-transform views:

1. original image;
2. central subject crop;
3. four-corner context mosaic.

The working interpretation is that `ratio=0.25` retains 25% of each crop's
width and height. This semantic choice is frozen for this baseline and is
subject to a train-only geometry audit; it is not selected by classification
performance. Official fish masks are not used to create these views.

## Separate training stages

Pretraining uses the explicitly listed CLIB-style augmentation with crop scale
0.2--1.0 and temperature 0.07. The projection head is 512→256→ReLU→128 with
L2 normalization. Original and same-sample subject views are the positive pair;
all other views in the batch are negatives. A pretrain checkpoint is written
atomically after every epoch.

The classifier stage uses the separately listed supervised augmentation,
freezes the encoder and BatchNorm statistics, trains the classifier with
cross-entropy on all outer-train labels, and writes resumable checkpoints.
The final classifier checkpoint is selected only on inner-dev accuracy.

## Required train-only geometry audit

Before any formal run, audit the first 32 image paths in the fixed fold-1
outer-train manifest. Report subject-box mask recall, subject-box/mask IoU,
mask-centroid inclusion, corner-mosaic mask inclusion, per-class counts and a
fixed contact sheet. The audit cannot inspect model predictions or any outer
test record. It cannot change the ratio after seeing classification results.

## Required reporting

Use the same outer evaluator and paired trajectory bootstrap as CXT-Fish:
original, foreground, same-class-swap and cross-class-swap macro-F1, tail-F1,
track-balanced accuracy, DAR, DAR-flip and agreement. Also report total
training epochs, GPU time, peak memory, views per sample, projection-head
presence and inference forward count.
