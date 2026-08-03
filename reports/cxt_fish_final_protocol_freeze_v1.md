# CXT-Fish final protocol freeze v1

## Decision

The method search is closed. CXT-Fish is frozen as a two-part contribution:

1. track-uniform sampling, which gives each trajectory equal training weight;
2. foreground-sufficiency training, which adds a fixed classification loss on a
   mask-based context-suppressed view during training only.

At inference the model receives one ordinary RGB image, performs one ResNet-18
forward pass, and requires neither a mask nor a trajectory identifier.

F0 (track-uniform CE) is the primary control. F1 is the primary method. F3 is
retained as a diagnostic ablation showing that stronger feature consistency is
not automatically beneficial. Donor-aware, TRAFS, CIR, CT-DFS, TAP, DLE and
CXT-Select are closed negative extensions and cannot be revived or tuned.

## Frozen evidence boundary

The current track-level validation results are development evidence. They were
used to select and freeze F1 before this document. The official test remains
locked. The previously named internal-test trajectories are released only as
part of the fixed outer confirmation below: they are assigned by the frozen
outer manifest and are never used for checkpoint or configuration selection.
They were not accessed before this freeze. The outer manifest must not be used
to change the method, seed list, checkpoint rule or metrics.

The CXT-Select negative evaluation was frozen at commit
`d387b858983dccef57d8a3eea93f274b69c4e133`. No locked partition was accessed
by that evaluation.

## Fixed outer confirmation

Run only the following 18 cells:

```text
3 outer folds × 3 registered seeds × {F0, F1}
```

For each fold, the outer-test trajectories are held out. A fixed
track-disjoint development subset is created inside outer-train and is shared
by F0 and F1. Early stopping is allowed only on that inner development subset.
The outer-test trajectories are read exactly once after training and are never
used for checkpoint or configuration selection.

The complete configuration is in
`configs/cxt_fish_outer_validation_v1.yaml`. The referenced split and manifest
hashes are recorded there and must match before any outer inference.

Primary metrics are macro-F1, tail-F1 and track-balanced accuracy. Secondary
metrics are balanced accuracy, foreground macro-F1, same-class and cross-class
context-swap macro-F1, DAR, DAR-flip and prediction agreement. All paired
intervals use group/trajectory bootstrap. Results must be reported per class,
per trajectory and by head/mid/tail groups.

## Explicit non-goals

No new loss, sampler, backbone, context generator, risk score, selective
prediction rule or post-hoc calibration is permitted after this freeze. The
outer results are confirmation, not a tuning set. A negative outer result is a
valid result and closes the method claim; it does not authorize another method
search cycle.

## Reproducibility manifest

| Asset | SHA-256 |
|---|---|
| `splits/f4k16t_class_ids.json` | `75fc443707303d902d0e92d17ef93cff587a3bb4b16179537bd2aba5e44affe6` |
| `splits/f4k16t_track_level_dev.csv` | `556650e75c641ae696f14345687a15a1afe67fe53ef9d1422db9db98adfed790` |
| `splits/f4k16t_outer_3fold.json` | `7bd0e25b30ddca42589ef33a9acb29131fb70396d8ba16357881b719ea093521` |
| `splits/f4k16t_phase1c_context_swap_val.manifest.json` | `4ae63a6cd2bcdd7ff7e310b8bdfa82d29351fa460b959803277d51cf51be86c7` |

The repository commit containing this freeze is the implementation provenance;
weights and generated outputs remain local and are not committed.

## Publication framing

The paper should present CXT-Fish as a track-aware evaluation protocol plus a
simple, no-inference-overhead foreground-sufficiency training principle. The
negative extensions belong in the supplement as boundary evidence, not as a
failure diary. Claims of priority (“first”) require a separate literature
audit before submission.
