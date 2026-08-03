# CXT-Fish Phase 1B: cross-trajectory representation and context control

## Entry condition and scope

Phase 1A selected **A1** (uniform-over-track sampling plus cross entropy) as
the replicated development base by tail F1, then track-balanced accuracy, then
macro F1. Phase 1B keeps the same ResNet-18, image size, optimizer, schedule,
track-level split, and frozen seeds `3407`, `2026`, and `17`.

Only the fixed `track/train` and `track/val` partitions may be read. The
internal test and outer folds remain locked. No prototype module, memory bank,
momentum teacher, alternate backbone, or class-prior loss is introduced.

## Fixed batch construction

Every contrastive batch contains `P=6` species, `Q=2` distinct recorded tracks
per species, and `K=2` frames per selected track (`24` RGB images per batch).
When a selected class has fewer than two eligible tracks, sampling with
replacement is disallowed and that class is skipped for that batch draw. The
sampler is seeded deterministically by `(seed, epoch)`.

## Models

All models retain ordinary RGB-only inference.

| ID | Training objective |
|---|---|
| C0 | A1 classification training plus ordinary supervised contrastive learning. Same-class positives may come from the same or different recorded tracks. |
| C1 | A1 classification training plus cross-trajectory supervised contrastive learning. A positive pair must have the same class and different recorded tracks. |
| C2 | C1 plus foreground-context consistency and foreground-only classification during training. |

The projection head is a two-layer MLP attached to the final ResNet-18 feature:
`512 -> 512 -> 128`, with ReLU between layers and L2-normalized 128-D output.
It is discarded at inference.

## Losses

For C0 and C1, the total loss is

`L = L_CE(original) + 0.1 * L_contrast`.

The contrastive temperature is fixed to `0.07`. C0 uses all same-class positives
in the batch. C1 uses only same-class, different-track positives; anchors with
no eligible positive are excluded from the contrastive average and their count
is logged.

For C2, the foreground-only image replaces pixels outside the official fish mask
with RGB `(128, 128, 128)`. Its loss is

`L = L_CE(original) + 0.1 * L_cross_track + 0.1 * L_context + 1.0 * L_CE(foreground)`.

`L_context = 1 - cosine(z_original, stop_gradient(z_foreground))`.

Masks are used only during C2 training. They are not used at validation or
inference, and they are never supplied as model inputs.

## Training and evaluation

The epoch ceiling and early-stopping rule remain the frozen Gate-0 settings.
All three models use the same seed set and track-level validation partition.
Primary development metrics are tail F1, track-balanced accuracy, and macro F1.
Context diagnostics are foreground-only retention and the fixed online
context-swap sensitivity defined in the next implementation commit.

This is a controlled development comparison, not an outer-fold claim. The
internal test may be read only after C0/C1/C2 and all associated diagnostics are
frozen.
