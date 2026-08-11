# DLE-Fish constrained inner-fold pilot

## Scope

This report records the completed two-fold, train-only pilot for the constrained
DLE-Fish candidates. The inner folds are group-disjoint and use only the frozen
training partition. Internal test, current validation, calibration, outer folds,
and official test were not read.

The pilot is a mechanism and implementation screen, not a final confirmation.
The current result is based on inner-fold validation accuracy; macro-F1, tail-F1,
track-balanced metrics, and context-swap metrics require the separate frozen
evaluation pass before selecting a candidate.

## Variants

- Q0: F1-compatible control.
- Q1: Q0 plus masked pooled-feature classification loss.
- Q2a: Q1 plus positive class-evidence concentration, weight 0.05.
- Q2b: Q1 plus positive class-evidence concentration, weight 0.10.

All variants use the same ResNet-18 initialization, sampler, optimizer,
augmentation, inner folds, seed (3407), and early-stopping rule. The only
intended difference is the DLE objective component.

## Inner-fold results

| Variant | Fold 0 best accuracy | Fold 1 best accuracy | Mean best accuracy |
|---|---:|---:|---:|
| Q0 | 0.991543 | 0.993362 | 0.992452 |
| Q1 | 0.988973 | 0.989400 | 0.989187 |
| Q2a | 0.990044 | 0.991756 | 0.990900 |
| Q2b | 0.990151 | 0.993255 | 0.991703 |

The auxiliary DLE terms were finite in both folds. Q2a and Q2b also produced
non-zero valid-mask fractions, confirming that the localization mechanism was
activated rather than silently receiving empty masks.

## Interpretation boundary

On this preliminary inner-fold accuracy screen, Q0 has the highest mean score;
Q2b is closest among the DLE variants, while Q1 and Q2a are lower. This is not
yet a scientific PASS/FAIL decision because the pre-registered selection
requires clean/tail/track-balanced and context-swap metrics. No candidate is
promoted or tuned from these numbers.

## Provenance and access audit

- Variant: DLE-Fish constrained pilot.
- Seed: 3407.
- Folds: frozen group-disjoint inner folds 0 and 1.
- Checkpoints: local output directories under `outputs/cxt_fish/phase2_dle/`.
- Internal test accessed: false.
- Current validation accessed: false.
- Calibration accessed: false.
- Outer folds accessed: false.
- Official test accessed: false.

The next authorized step is one deterministic frozen evaluation pass on the two
inner held-out folds, followed by the pre-registered Pareto comparison. No new
loss, sampler, backbone, or hyperparameter search is authorized.
