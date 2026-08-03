# DLE-Fish constrained pilot protocol

## Scope

DLE-Fish is a final, limited method pilot following the frozen F1 result and
the corrected CT-DFS/linear-probe audit. It is not a continuation of donor,
TRAFS, CIR or TAP-Fish. Those results and reports remain unchanged.

The method tests whether official fish masks can supervise where the target
class activation is concentrated in the original RGB image, without changing
the inference input or sampling distribution.

## Evidence terminology

The spatial quantity is called **positive class-evidence concentration**. It
is derived from the GAP-compatible class activation map and softplus; it is
not claimed to be an exact decomposition of the classifier's full evidence,
and masked pooling is not called a pure fish-only representation because the
deep feature cells have large receptive fields.

## Fixed mechanism

The ResNet18 layer-4 pre-pooling feature map is used. Global pooling and
mask-weighted pooling share one classifier. Q0 is the F1-compatible control;
Q1 adds masked-original CE; Q2a/Q2b add the fixed positive-evidence
concentration loss with lambda 0.05 or 0.10. Inference uses only original RGB
and never reads a mask.

Augmentation geometry is applied identically to RGB, foreground view and mask.
Masks are area-resampled to the feature map and clipped to [0,1]. Samples
with post-transform mask mass below one complete feature cell skip masked and
localization auxiliary terms while retaining the two F1 CE terms; the skip
rate is logged by class and mask-area bin.

## Development protocol

Two deterministic, group-disjoint inner folds are built from train only in
`splits/f4k16t_phase2_inner_2fold.json`. Current validation, internal test and
outer folds are not used for method selection. Q0, Q1, Q2a and Q2b share the
same ResNet18 initialization, sampler, augmentation, optimizer, schedule and
seed. No additional backbone, head, BN, sampler or evidence weight is allowed.

Selection is Pareto-based across both inner folds: clean and tail macro-F1
losses are each at most 0.5pp, track-balanced accuracy loss at most 0.3pp,
and at least one of cross-swap macro-F1 or DAR-flip improves without a clear
loss in the other; evidence concentration must improve in both folds and
across multiple classes. Current validation is used once only after the
variant is frozen.

`internal_test_accessed=false`, `outer_folds_accessed=false`, and
`official_test_accessed=false` are mandatory metadata fields.
