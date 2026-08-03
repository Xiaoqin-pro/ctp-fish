# CXT-Fish Phase 1B v1.1: Cross-Trajectory Contrast Only

## Revision status

This transparent training-before-results revision supersedes Phase 1B v1.0 for
implementation. The original v1.0 document remains unchanged as an audit
record. The revision separates cross-trajectory contrast from all context
control mechanisms, which move to a future Phase 1C.

## Frozen reference and scope

Phase 1A A1 (global track-uniform sampling plus CE) is the frozen natural
training reference and is **not** retrained here. C0 is a structured-batch
control for C1/C2, not a strict single-variable comparison with A1: structured
batching changes class and within-class trajectory composition.

Only fixed `track/train` and `track/val` may be read. Internal test and outer
folds remain locked. The development pilot seed is `3407`, already recorded in
the repository configuration. Seeds `2026` and `17` are reserved for a later
replication only if the v1.1 pilot provides positive, stable mechanism evidence.

## Shared structured batch and model

All C0/C1/C2 training batches use `P=4` classes, `Q=3` distinct tracks per
class, and `K=2` frames per track (`batch_size=24`). The same deterministic
sampler, ResNet-18 initialization, augmentations, CE classification branch,
optimizer, schedule, epoch ceiling, early stopping rule, and seed are shared.

All three instantiate the same projection head:

`512 -> 256 -> ReLU -> 128 -> L2 normalization`.

The head is retained in the optimizer and checkpoint for C0 even though its
contrastive weight is zero. It is discarded at RGB-only inference.

## Controlled comparison

| ID | Classification loss | Contrastive weight | Positive set |
|---|---|---:|---|
| C0 | CE | 0.0 | none; structured-batch control |
| C1 | CE | 0.1 | same class, non-self |
| C2 | CE | 0.1 | same class, different `group_id` |

Contrastive temperature is fixed to `0.1`. No foreground-only classification,
context consistency, context-swap training, prototype loss, memory bank,
momentum teacher, new backbone, class reweighting, or additional sampler search
is permitted in Phase 1B.

With `P=4, Q=3, K=2`, each anchor has one same-track and four cross-track
same-class positives. Thus C1 has five positives and C2 has four; C2 differs
from C1 only by exclusion of the same-track positive.

## Required implementation checks

Before any real pilot run, tests must show:

1. exactly `P` classes, `Q` distinct tracks per class, and `K` frames per track;
2. C1 positives equal same-class non-self samples;
3. C2 positives equal same-class different-track samples and every anchor has a
   positive;
4. samplers read only training records and never validation/test records;
5. identical seeds reproduce the batch sequence;
6. C0/C1/C2 differ only in positive-mask/loss activation;
7. contrastive loss handles no-positive anchors safely and remains finite.

Training logs additionally record CE, contrastive loss, mean same-track and
cross-track positives per anchor, eligible-anchor count, embedding norms, and
per-epoch observed class/track counts.

## Pilot decision

Run C0, then C1, then C2 with seed `3407`; read only fixed validation metrics.
Implementation validity precedes interpretation. The scientific comparisons are
`C1-C0` (ordinary SupCon) and `C2-C1` (cross-trajectory positive definition).
Primary order is tail F1, track-balanced accuracy, then macro F1. Do not tune
P/Q/K, temperature, projection size, or loss weight after this pilot.

If C2 exceeds C1 without clear macro-F1 damage and gains are distributed beyond
a single class/track, run the reserved seeds. If C1 helps but C2 does not,
remove the cross-trajectory innovation claim. If neither helps, stop contrastive
tuning and evaluate context control separately in Phase 1C.
