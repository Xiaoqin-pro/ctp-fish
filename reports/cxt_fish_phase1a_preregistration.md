# CXT-Fish Phase 1A: orthogonal sampling and long-tail baselines

## Purpose

Phase 1A separates class-frequency correction from trajectory-length
correction. It does not train cross-track contrast, foreground-context losses,
prototypes, memory banks, or alternative backbones.

Only frozen track-level `train` and `val` partitions are permitted. The
internal test and outer folds remain unread during every Phase 1A comparison
and selection step.

## Four samplers

| ID | Sampling distribution |
|---|---|
| S0 | Uniform over frames; long tracks have proportionally higher total probability. |
| S1 | Uniform over all tracks, then uniform over a frame within the selected track. |
| S2 | Uniform over classes, then uniform over all frames within the selected class. |
| S3 | Uniform over classes, then tracks within class, then frames within track. |

All samplers produce the same number of optimization steps and use the same
batch size, augmentation family, ResNet-18, optimizer, schedule, epoch ceiling
and fixed screening seed `407`.

## Fixed screening matrix

| ID | Sampler | Classification objective |
|---|---|---|
| A0 | S0 | Cross entropy |
| A1 | S1 | Cross entropy |
| A2 | S2 | Cross entropy |
| A3 | S3 | Cross entropy |
| A4-I | S0 | Balanced Softmax using image-count priors from train only |
| A4-T | S0 | Balanced Softmax using track-count priors from train only |
| A5-I | S1 | Balanced Softmax using image-count priors from train only |
| A5-T | S1 | Balanced Softmax using track-count priors from train only |

No S2/S3 plus Balanced-Softmax run is assumed necessary or beneficial in this
screening matrix: class-uniform sampling already changes the effective class
frequency and would confound the mechanism comparison.

## Metrics and selection

For each configuration report validation macro F1, balanced accuracy,
track-balanced accuracy, head/mid/tail per-class F1 and per-track error
distribution. All eight configurations are reported.

Select two or three configurations for three-seed replication in this fixed
order: tail F1, then track-balanced accuracy, then macro F1; an exact tie uses
the simpler sampler/objective. Selection occurs on validation only. The
internal test is read once only after Phase 1A and later method components are
fully frozen.

## Interpretation

Phase 1A asks whether long-track dominance, class imbalance, and the choice of
image-count versus track-count priors each affect cross-track generalization.
It does not ask which system can maximize a near-ceiling aggregate accuracy.
Any result is described as a development finding, not an outer-fold claim.
