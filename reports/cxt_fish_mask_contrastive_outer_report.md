# Route C outer evaluation report

## Scope

This report freezes the protocol-matched mask-guided subject--non-primary
contrastive baseline (route C). It is an external baseline for CXT-Fish, not a
new CXT-Fish variant and not a CLIB reproduction. All 3 folds and 3 registered
seeds were trained before outer-test access. The official test set remains
locked.

## Evaluation integrity

- Training cells: 9/9 complete and provenance-validated.
- Outer evaluation cells: 9/9 complete.
- Evaluation used the frozen `best.pt` selected by inner-dev accuracy.
- `outer_test_accessed`: true for this report; `official_test_accessed`: false.
- No checkpoint, loss, view, augmentation, or parameter changes were made
  after outer evaluation began.
- The route-C outer evaluator used strict checkpoint loading and the explicit
  outer-test unlock gate.

## Route-C outer metrics

| fold | seed | original macro-F1 | foreground macro-F1 | same-swap macro-F1 | cross-swap macro-F1 | DAR-flip |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 0.696307 | 0.686857 | 0.673722 | 0.413088 | 0.142661 |
| 1 | 2026 | 0.698398 | 0.704419 | 0.675445 | 0.364900 | 0.188219 |
| 1 | 3407 | 0.738548 | 0.740911 | 0.708286 | 0.399608 | 0.156548 |
| 2 | 17 | 0.705007 | 0.718216 | 0.719879 | 0.467252 | 0.134715 |
| 2 | 2026 | 0.644334 | 0.638065 | 0.643149 | 0.383331 | 0.158279 |
| 2 | 3407 | 0.656627 | 0.653501 | 0.663223 | 0.404195 | 0.150467 |
| 3 | 17 | 0.709260 | 0.709040 | 0.738448 | 0.466015 | 0.163446 |
| 3 | 2026 | 0.714898 | 0.714637 | 0.746146 | 0.460121 | 0.176180 |
| 3 | 3407 | 0.722562 | 0.717365 | 0.744538 | 0.443065 | 0.178176 |
| **mean** |  | **0.698438** | **0.698112** | **0.701426** | **0.422397** | **0.161410** |

## Paired comparison with frozen CXT-Fish F1

The same outer-test images and group IDs were paired with the previously
frozen F1 outputs. A 1,000-replicate group bootstrap was computed from
group-level confusion matrices; route C minus F1 is reported below. The point
estimate is the mean over the nine fold--seed cells.

| metric | mean route-C minus F1 | cells with bootstrap CI lower bound > 0 |
|---|---:|---:|
| original macro-F1 | -0.257164 | 0/9 |
| foreground macro-F1 | -0.256893 | 0/9 |
| same-swap macro-F1 | -0.227720 | 0/9 |
| cross-swap macro-F1 | -0.168896 | 0/9 |

Every original-macro-F1 cell had a strictly negative bootstrap interval. For
cross-swap, eight cells were strictly negative; the remaining interval still
did not establish a positive route-C advantage.

## Decision

Route C is a valid, reproducible external baseline, but it does not provide a
better clean--context-robustness trade-off than the frozen label-level
foreground-sufficiency training. Its outer performance is substantially lower
on both ordinary recognition and cross-class context swap. No route-C tuning,
new loss, new view, or new training run is authorized after this report.

The main CXT-Fish result remains the F1 outer result; route C is retained as a
negative but fair comparison.
