# CXT-Fish Phase 1C R0 three-seed replication

## Scope

F0–F3 are compared on the frozen validation partition for seeds 3407, 2026, and 17. F0 reuses the corresponding frozen A1 checkpoint. Internal test and outer folds were not accessed.

## Three-seed absolute metrics

| Variant | Original macro F1 | Tail F1 | Track-balanced accuracy | Cross-class swap macro F1 | DAR_flip |
|---|---:|---:|---:|---:|---:|
| F0 | 0.9619 ± 0.0111 | 0.9427 ± 0.0198 | 0.9866 ± 0.0019 | 0.5402 ± 0.0027 | 0.1995 ± 0.0047 |
| F1 | 0.9733 ± 0.0055 | 0.9622 ± 0.0229 | 0.9887 ± 0.0042 | 0.5844 ± 0.0426 | 0.1790 ± 0.0332 |
| F2 | 0.9598 ± 0.0053 | 0.9168 ± 0.0197 | 0.9864 ± 0.0011 | 0.4991 ± 0.0231 | 0.2089 ± 0.0219 |
| F3 | 0.9649 ± 0.0097 | 0.9541 ± 0.0104 | 0.9892 ± 0.0011 | 0.5941 ± 0.0298 | 0.1772 ± 0.0330 |

## F3 relative to matched F0

- Cross-class swap macro F1: +0.0539 ± 0.0289; positive in 3/3 seeds.
- Conditional donor attraction (DAR_flip): -0.0223 ± 0.0332; lower is favorable, so it is lower in 2/3 seeds.
- Original macro F1: +0.0029 ± 0.0176; negative in 1/3 seeds.

## Interpretation boundary

R0 determines whether the seed-3407 context-robustness signal replicates. It does not select a new method or alter alpha, mu, blur, masks, sampler, architecture, or evaluation. R1 is limited to the predeclared original-domain BatchNorm-only diagnostic for F3.
