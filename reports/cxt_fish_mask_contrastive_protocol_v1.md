# Mask-Guided Subject--Non-Primary Contrastive Baseline v1

This is route C: a semantic, mask-guided external baseline for the frozen
CXT-Fish F0/F1 comparison. It is not CLIB reproduction and it is not a new
CXT-Fish method variant.

## Fixed views

For every outer-train image, the official binary mask creates three views:

1. `original`: unchanged RGB;
2. `foreground_subject`: retain the annotated primary fish and fill all
   non-primary pixels with RGB value 128;
3. `non_primary_context`: retain the non-primary region and fill annotated
   fish pixels with RGB value 128.

All three views receive the same random crop, flip and color-jitter parameters
so that the mask semantics remain aligned. The mask is used only during
training-view construction; inference remains ordinary RGB only.

## Fixed objective and budget

The shared ResNet18 encoder and 512→256→ReLU→128 L2-normalized projection head
use temperature 0.07. Original and foreground views of the same sample are a
positive pair. The same-sample non-primary view is an explicit hard negative;
all other candidate views are negatives. Pretraining runs 40 epochs, followed
by a frozen-encoder 40-epoch classifier stage using the same labeled
outer-train records as CXT-Fish. Both stages use the fixed augmentation and
Cosine schedule in the YAML. No parameter search is allowed.

## Data and evaluation boundary

There are 3 outer folds × seeds {3407, 2026, 17} = 9 runs. The inner-dev split
is used only for checkpoint selection. Each outer-test fold is read once after
all training. Calibration, current validation and official TEST remain locked.

## Scientific comparison

The primary comparison is not “which method wins clean accuracy.” The frozen
questions are whether a feature-level subject/non-primary separation baseline
offers a different clean--context-robustness trade-off from the label-level
foreground-sufficiency training in CXT-Fish, under identical trajectories,
backbone, seeds, data budget and outer evaluator.

All results, including a weaker baseline, must be retained. No tuning is
allowed after any outer result is observed.
