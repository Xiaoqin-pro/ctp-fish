# CTP-Fish Gate-0: Data and Baseline Audit

## Decision: PROCEED_WITH_CAUTION

Gate-0 provides structural evidence to continue **only with a track-aware,
context-controlled protocol**. This is not a method result and does not open
the locked outer folds.

## Dataset and protocol

- Official Fish4Knowledge records audited: **27,370** images, **8,725** species-track groups, **23** species.
- F4K-16T development subset: **27,133** images across 16 retained species.
- Image-level split permits trajectory crossing; track-level split keeps every `group_id` in exactly one partition.
- All models: ImageNet-initialized ResNet-18, 224 px, AdamW, fixed 40-epoch ceiling and early stopping; seeds [3407, 2026, 17].

## Baseline comparison (mean +/- sample SD across three seeds)

| Split protocol | Macro F1 | Track-balanced accuracy |
|---|---:|---:|
| Image-level | 0.9917 +/- 0.0039 | 0.9960 +/- 0.0003 |
| Track-level | 0.9669 +/- 0.0084 | 0.9887 +/- 0.0027 |

The macro-F1 gap is 2.48 percentage points. It is evidence that the conventional image-level protocol is optimistic for class-balanced recognition, even though aggregate image accuracy remains high.

### Head/mid/tail class summary

Classes are assigned once by descending track-level training-image count and split into three near-equal groups (head/mid/tail); values are unweighted mean per-class F1 across three seeds.

| Class-frequency tier | Image-level F1 | Track-level F1 |
|---|---:|---:|
| Head | 0.9976 | 0.9949 |
| Mid | 0.9907 | 0.9745 |
| Tail | 0.9856 | 0.9257 |

## Cross-partition similarity audit

| Protocol | Mean pHash distance | Mean feature cosine | pHash nearest-neighbor same-track fraction | Feature nearest-neighbor same-track fraction |
|---|---:|---:|---:|---:|
| Image-level | 9.5568 | 0.9183 | 0.2550 | 0.4821 |
| Track-level | 10.8383 | 0.9005 | 0.0000 | 0.0000 |

Track-level splitting removes direct crossing of the recorded fish trajectories. It does not establish independence across site, camera, date, or unrecorded scene factors.

## Online mask-view background audit (fixed track-level seed 407)

| Train view -> test view | Accuracy | Balanced accuracy | Macro F1 | Track-balanced accuracy |
|---|---:|---:|---:|---:|
| original -> original | 0.9919 | 0.9317 | 0.9367 | 0.9852 |
| foreground_only -> foreground_only | 0.9928 | 0.9310 | 0.9380 | 0.9893 |
| background_only -> background_only | 0.9783 | 0.7744 | 0.7995 | 0.9702 |
| original -> foreground_only | 0.9508 | 0.8142 | 0.8002 | 0.9443 |
| original -> background_only | 0.6451 | 0.3996 | 0.3652 | 0.6305 |


The background-only matched-view result is materially above chance for 16 classes. This establishes available contextual signal, not a claim that background is the sole causal shortcut. The original-view model's performance under foreground-only and background-only test shifts should be interpreted as sensitivity to information removal rather than as an in-distribution accuracy estimate.

## Research room and boundary

The observed combination of (i) track-crossing similarity under image-level splitting, (ii) a track-level macro-F1 decrease, and (iii) nontrivial background-only recognition supports a constrained next phase: methods must be evaluated with the frozen track-aware protocol and must directly test whether improvements remain after controlling contextual cues. No trajectory sampler, prototype learner, contrastive objective, or outer-fold result is included in Gate-0.

## Provenance

- Git commit: `38d35b2a366bffc057d3792ee96fad8f7df36580`
- Config SHA-256: `bda8d22d0244331ddb9f3cdd6368902654cba4d5ef07884f50e190a08cba0339`
- Image-level split SHA-256: `73e68a81d8e0a13b99034ee0f337e31a34419794a47f984d055e4acb1585250e`
- Track-level split SHA-256: `556650e75c641ae696f14345687a15a1afe67fe53ef9d1422db9db98adfed790`
- Per-class table: `experiments\ctp_fish_gate0_per_class_summary.csv`
- Track-error table: `experiments\ctp_fish_gate0_track_error_summary.csv`
- Official outer folds remain locked and were not evaluated.
