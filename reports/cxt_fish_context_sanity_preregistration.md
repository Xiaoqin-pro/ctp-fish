# CXT-Fish context-shortcut sanity audit

## Purpose and boundary

This is a fixed, single-seed diagnostic that refines the interpretation of the
Gate-0 background-only result. It does not reopen Gate-0, choose a final
method, train a contrastive model, or serve as a new pass/fail gate.

Only the frozen track-level `train` and `val` partitions are used. The existing
track-level internal `test` partition and locked outer three folds are not
read. All diagnostic output is written under `outputs/cxt_fish/context_sanity`
and is excluded from Git.

## Fixed seed and common model protocol

- Seed: `407`.
- Image classifier: the same ImageNet-initialized ResNet-18 family as Gate-0.
- Data protocol: frozen F4K-16T track-level train/val manifests.
- Training schedule: the Gate-0 optimizer, augmentation family, epoch ceiling,
  scheduler and early-stopping rule.
- No architecture replacement, class reweighting, contrastive loss, prototype,
  memory bank, momentum teacher, or outer-fold access.

## Five visual diagnostic conditions

| Condition | Input to classifier | Diagnostic question |
|---|---|---|
| `mask_only` | Official binary fish mask copied to three channels | Are silhouette, area and location discriminative? |
| `constant_fill_background` | Recipient RGB with its fish mask filled with RGB `(128,128,128)` | Reproduce the Gate-0 hole-bearing background view. |
| `inpainted_background` | Recipient RGB after Telea removal of the dilated fish mask | Does contextual signal remain after suppressing the fish hole and edge? |
| `shuffled_mask_background` | Recipient RGB with a deterministically assigned donor mask filled neutral gray | Does a mask-shaped hole itself transmit class information? |
| `geometry_only` | Six normalized mask-box features into multinomial logistic regression | Can geometry and composition alone explain signal? |

`mask_only`, `constant_fill_background`, `inpainted_background`, and
`shuffled_mask_background` each train and validate their own ResNet-18. The
geometry condition uses a non-visual classifier and never receives RGB pixels.

## Fixed image transformations

### Inpainting

The implementation uses OpenCV Telea inpainting with radius `3` pixels. Before
inpainting, the foreground mask is dilated with a `3 x 3` elliptical kernel
for one iteration. This can produce unnatural regions for large fish and is a
diagnostic transformation only, not an attempt to recover a real underwater
background.

### Shuffled masks

For each recipient image, assign one donor mask from the same partition using
a stable SHA-256 ordering of `cxt_fish_mask_shuffle_v1|recipient_image_path`.
The donor must have a different `group_id`; its nearest-neighbor mask is
resized to the recipient image dimensions. The assignment is generated once
from split metadata, stored as an index CSV with its SHA-256, and reused for
train and validation. The donor species is not used to choose a favorable
mask.

### Geometry-only features

For the smallest foreground bounding box `(x, y, w, h)` in a width `W` and
height `H` image, fit a multinomial logistic regression on:

`w/W, h/H, wh/(WH), (x+w/2)/W, (y+h/2)/H, w/h`.

Standardization parameters are fit on train only. Logistic-regression regular-
ization strength is fixed to `C=1.0`, solver `lbfgs`, `max_iter=2000`, and
seed `407`; no CNN is used in this diagnostic.

## Outputs and interpretation

Report validation accuracy, balanced accuracy, macro F1, tail-class F1 and
per-class F1 for every condition. The report distinguishes observations from
causal claims:

- high inpainted-background performance supports residual environmental or
  camera-context signal;
- high mask-only or geometry-only performance supports shape, scale or
  composition shortcuts;
- a large constant-fill to inpainted-background decrease implicates the
  recipient fish-shaped hole as part of the Gate-0 signal.

Regardless of outcome, the paper refers to **contextual shortcuts** unless a
more specific mechanism is directly supported. No finding is described as data
leakage or as an external critique of prior work.

## Next-step guard

After this audit is frozen, the next authorized work is Phase 1A's orthogonal
sampling/class-prior baselines. Cross-track contrast and foreground-context
losses remain prohibited until Phase 1A has selected a base setting on
validation.
