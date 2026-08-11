# CXT-Fish context-shortcut sanity audit

## Scope

This fixed-seed diagnostic uses only the frozen F4K-16T track-level `train`
and `val` partitions. It does not read the internal test partition or locked
outer folds. The visual conditions each use a separately trained ResNet-18;
geometry-only uses the preregistered six-feature logistic regression.

## Validation results

| Condition | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| mask_only | 0.9049 | 0.6933 | 0.7127 | 0.9031 |
| geometry_only | 0.4406 | 0.0850 | 0.0768 | 0.2911 |
| constant_fill_background | 0.9811 | 0.8677 | 0.8955 | 0.9804 |
| inpainted_background | 0.9690 | 0.7754 | 0.8053 | 0.9679 |
| shuffled_mask_background | 0.9799 | 0.8753 | 0.8968 | 0.9792 |


## Interpretation fixed by this audit

- `mask_only` reaches macro F1 0.7127, whereas
  `geometry_only` reaches 0.0768. Thus,
  detailed silhouette/shape information is highly discriminative, while the
  predefined box-scale-position features alone are not a sufficient account.
- `constant_fill_background` reaches 0.8955;
  `shuffled_mask_background` reaches 0.8968 (difference
  +0.0013). Replacing the fish-hole shape with a
  cross-track donor mask does not materially remove the available signal in
  this fixed diagnostic.
- `inpainted_background` remains high at 0.8053, but is
  0.0902 below the constant-fill condition. Therefore the
  original background-only signal is consistent with both residual
  environmental/camera context and some fish-removal artifact. Telea
  inpainting is itself a diagnostic transformation and not a reconstruction of
  the true seabed.

The manuscript terminology is therefore fixed to **contextual shortcuts**:
the evidence does not justify attributing the effect solely to background,
nor does it establish causal environmental influence or scene independence.

## Consequence

The next authorized stage is Phase 1A's orthogonal sampling and class-prior
baselines. Cross-track contrast, foreground context loss, prototype learning,
memory banks, internal-test access and outer-fold access remain out of scope
until Phase 1A is frozen.

## Provenance

- Seed: `407`
- Shuffled-mask index SHA-256: `ebba4ba4f7d6bd3013b86af48db78e77e24c6035f2c7aa189cebaf2f33cd7681`
- Index rows: `22805` (train + validation only)
- Internal test read: `False`
- Outer folds read: `False`
- Report input summary SHA-256: `a1e6560bfcbb5e6ece7de0cb76644834707d65ceddcb0c922123d31ab7acd39f`
