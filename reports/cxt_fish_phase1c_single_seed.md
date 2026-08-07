# CXT-Fish Phase 1C single-seed pilot

## Scope

This report compares F0–F3 at registered seed 3407 on the frozen track-level validation partition only. F0 reuses the frozen A1 checkpoint; F1/F2/F3 are sequential new runs. Internal test and outer folds were not accessed.

## Context metrics

| Variant | Original macro F1 | Foreground macro F1 | Same-class swap F1 | Cross-class swap F1 | DAR_flip |
|---|---:|---:|---:|---:|---:|
| F0 | 0.9747 | 0.8798 | 0.9258 | 0.5433 | 0.2011 |
| F1 | 0.9749 | 0.9769 | 0.9518 | 0.5786 | 0.2106 |
| F2 | 0.9656 | 0.8056 | 0.9316 | 0.5106 | 0.1859 |
| F3 | 0.9598 | 0.9619 | 0.9163 | 0.5991 | 0.1410 |

## Fixed comparisons

- F1−F0: cross-class swap macro F1 +0.0353; DAR_flip +0.0095; original macro F1 +0.0002.
- F2−F0: cross-class swap macro F1 -0.0327; DAR_flip -0.0152; original macro F1 -0.0091.
- F3−F1: cross-class swap macro F1 +0.0205; DAR_flip -0.0696; original macro F1 -0.0151.

## Interpretation boundary

F3 provides a positive single-seed robustness signal: lower donor attraction and higher cross-class-swap F1 than F0/F1. It also has lower original-image macro F1 than F0/F1. This pilot therefore supports only a decision about whether to run the two registered replication seeds; it is not a final method claim, and it must not be used to change foreground construction, loss weights, sampling, or evaluation rules.
