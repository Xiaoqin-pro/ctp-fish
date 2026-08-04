# CXT-Fish frozen outer confirmation

## Protocol

This report summarizes the pre-registered three-fold, three-seed outer confirmation after all 18 F0/F1 cells were trained and frozen. Checkpoints were selected only on the fold-specific inner development split. The outer-test partition was read only in this final evaluation; official TEST was not accessed.

- Cells: 3 folds x 3 registered seeds (3407, 2026, 17) x F0/F1
- Primary views: original RGB, foreground-only, same-class cross-track swap, cross-class swap
- Unit of generalization: group/trajectory-disjoint outer folds
- Outer-test access before this evaluation: false
- Official TEST accessed: false

## Mean +/- SD across 9 fold-seed pairs

| Method | Original macro-F1 | Original tail-F1 | Track-balanced accuracy | Cross-swap macro-F1 | DAR-flip |
|---|---:|---:|---:|---:|---:|
| F0 | 0.9577 +/- 0.0086 | 0.9233 +/- 0.0209 | 0.9887 +/- 0.0016 | 0.5189 +/- 0.0251 | 0.2146 +/- 0.0164 |
| F1 | 0.9556 +/- 0.0193 | 0.9260 +/- 0.0249 | 0.9895 +/- 0.0025 | 0.5913 +/- 0.0437 | 0.1792 +/- 0.0166 |

## F1 - F0 paired deltas

| Metric | Mean delta | SD | Paired group bootstrap 95% CI | Favorable pairs |
|---|---:|---:|---:|---:|
| Original macro-F1 | -0.0021 | 0.0186 | [-0.0116, +0.0071] | 6/9 |
| Original tail-F1 | +0.0026 | 0.0318 | not computed | 5/9 |
| Track-balanced accuracy | +0.0009 | 0.0024 | [-0.0002, +0.0019] | 7/9 |
| Cross-swap macro-F1 | +0.0724 | 0.0405 | [+0.0601, +0.0817] | 8/9 |
| DAR-flip (lower is better) | -0.0354 | 0.0200 | not computed | 9/9 |

## Interpretation

F1 is not an overall clean-accuracy improvement method in this outer confirmation: its mean original macro-F1 is slightly below F0. Its consistent signal is context robustness: cross-class context-swap macro-F1 is higher on average and DAR-flip is lower on average, while tail-F1 and track-balanced accuracy are approximately preserved or slightly improved. The appropriate claim is therefore foreground-sufficiency training as a simple robustness intervention, not a universally superior classifier.

The complete per-cell table is in `experiments/cxt_fish_outer_results.csv`. Raw per-image outputs remain local under `outputs/cxt_fish/final_outer_evaluation/` and are not part of the tracked result table.
