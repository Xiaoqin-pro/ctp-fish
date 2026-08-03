# CT-DFS Fish: final method pilot

CT-DFS (Context- and Track-balanced Dual-Flow Supervision) is the last
authorized method pilot in this repository. It combines the frozen F1
foreground-supervision result with a controlled redistribution of foreground
classification supervision.

## Fixed variants

- **P0:** frozen F1 checkpoint; original and foreground view use the same
  sample (reference only; no retraining in this pilot).
- **P1:** original stream uses S1 track-uniform sampling; foreground stream
  independently uses S1 track-uniform sampling.
- **P2:** original stream uses S1; foreground stream independently uses S3
  class-and-track-uniform sampling.

P1 and P2 use one shared ResNet18 classifier, one concatenated forward pass for
both streams, the same augmentation family, and the same CE objective. The
only planned change is the foreground index distribution. Internal test and
outer folds remain locked; validation is used only for the pilot comparison.

## Pilot decision

The primary comparison is **P2 minus P1**. A single seed (`3407`) is run first.
The engineering authorization targets are: tail F1 about +1 percentage point,
cross-class swap macro-F1 about +1 point, clean macro-F1 decline no more than
0.5 points, no clear track-balanced accuracy decline, at least half of tail
classes improving, and gains spread across multiple tracks. These are pilot
authorization targets rather than claims of statistical significance.

If P2 is directionally better, two preregistered seeds (`2026`, `17`) may be
run and the S2 class-balanced foreground ablation is added only for the final
paper. If P2 is not better than P1, CT-DFS is closed and no further method
variant is introduced.
