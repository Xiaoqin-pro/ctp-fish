# Route C outer evaluation report

This report is regenerated from the frozen route-C metrics JSON files. Route C is a two-stage mechanism control, not a faithful CLIB reproduction or an exhaustive contrastive-learning comparison.

## Integrity

- Training and outer evaluation cells: 9/9.
- `official_test_accessed`: false in all nine metrics files.
- No new route-C training, tuning, or checkpoint selection was performed during this regeneration.

## Cell results

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
| **mean** |  | **0.698438** | **0.698112** | **0.701426** | **0.422397** | **0.160966** |

## Interpretation

Route C remains substantially below the frozen label-level CXT-Fish baseline on ordinary recognition and cross-class context-swap performance. The corrected DAR-flip mean above is computed directly from all nine raw JSON files.
