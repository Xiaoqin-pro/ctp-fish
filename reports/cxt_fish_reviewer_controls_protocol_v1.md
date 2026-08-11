# CXT-Fish reviewer-risk control package v1

## Status and scope

This package is a post-hoc, reviewer-motivated supplement after the frozen
outer results. It does not reopen CXT-Fish method development, select a new
method, or alter the frozen F0/F1, MobileNet, Route C, donor manifest, point
estimate, or primary bootstrap output. The parent result commit is
`0332cd9bf580cde12e5f82587dd7cae9dcba3b6b`.

The package has three parts:

1. protocol and artifact audits that do not train or access official TEST;
2. donor-assignment sensitivity using frozen checkpoints only;
3. one compute-, supervision-, and BatchNorm-exposure-matched ordinary-RGB
   control (`F0_2RGB`) trained on the same nine fold/seed cells.

All new outputs are supplementary. The frozen primary donor realization uses
seed 3407 and remains unchanged.

## A. Audits

The audit records the pre-specified 23-to-16 class rule (`num_groups >= 15`),
the exact donor selection semantics, the context-diagnostic implementation,
fold-3 per-class heterogeneity, and deterministic examples. The audits must
describe what is present in the frozen artifacts; missing historical details
are reported as `not recoverable from frozen artifacts` rather than inferred.

## B. Donor-assignment sensitivity

Five alternative deterministic manifests use seeds 4101--4105. They reuse the
frozen candidate pool, ordering, hash rule, and composite construction; only
the seed changes. The original seed-3407 manifest is never overwritten.

For each realization, the same manifest is used for F0 and F1 and all nine
fold/seed cells are evaluated with frozen checkpoints. The descriptive primary
quantity is the equal-weight mean of the nine paired cross-class donor-context
composite macro-F1 differences, together with its cell SD and favourable-cell
count. All five realizations are reported; no realization is selected or added
because of its result. This is a post-hoc sensitivity analysis, not a new
confirmatory endpoint.

## C. Two-RGB matched control

`F0_2RGB` uses the same ResNet18, ImageNet initialization, frozen class list,
track-uniform sampler, 64-recipient batches, concatenated 2B forward, AdamW
configuration, cosine schedule, 40-epoch budget, inner-development clean
accuracy checkpointing, and three fixed seeds as F1. It differs only in the
second view: both views are independently augmented ordinary RGB images; no
mask, foreground blur, or background editing is used.

The loss is:

```text
CE(f(view1), y) + CE(f(view2), y)
```

All nine cells must finish before any outer pixel is read. The only focused
comparison is F1 versus F0-2RGB on cross-class donor-context-composite
macro-F1. Clean, foreground, same-composite, DAR-flip, and agreement are
descriptive. Any interval is labelled an exploratory post-hoc control
interval and is not a new confirmatory endpoint.

## Fixed boundaries

- Official Fish4Knowledge TEST remains locked and unaccessed.
- No second dataset, third backbone, new loss, augmentation search, seed
  search, donor-seed expansion, CLIB re-training, or Route C re-training.
- Existing primary manifests and outputs are immutable.
- Any implementation bug is fixed in a separate commit without changing the
  scientific protocol.
- After this closed package is complete, model experimentation stops again.

## Configuration

The machine-readable version is
`configs/cxt_fish_reviewer_controls_v1.yaml`.
