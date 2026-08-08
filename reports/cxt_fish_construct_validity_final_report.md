# CXT-Fish construct-validity final report

This package is a closed, reviewer-motivated, inference-only and reanalysis-only extension. No model was trained, no checkpoint was selected, and official Fish4Knowledge TEST was not accessed.

## 1. Protocol

- Protocol commit: `7471c0cb64305a811231755b033fd14a3199300f`.
- Final package head recorded by Git: `dae183ae43e412ef03a18ac2e083e17b17b670c5`.
- Role: post-hoc construct-validity and statistics repair.

## 2. F0-2RGB statistical repair

- Correct observed equal-weight 9-cell effect: **+6.83 pp**.
- Bootstrap mean: +6.83 pp.
- Exploratory percentile 95% interval: [+5.71, +7.87] pp.
- The former JSON point estimate was the bootstrap distribution mean and is superseded.

## 3. Donor construct audits

- Donor-species rows: 16; natural-manifest-weight effect and donor-species-equal-weight sensitivity are reported separately.
- Donor residual pairs: 27133; mean hard-mask residual fraction: 0.0301.
- Residual/effect bins use pre-specified cutoffs and are descriptive, not causal.

## 4. Fish-suppressed donor sensitivity

- Frozen-pair observed effect: **+10.48 pp**.
- Exploratory percentile 95% interval: [+9.38, +11.61] pp.
- Same recipient, same donor, same pairing, same checkpoint; only donor subject suppression changed.
- This analysis may support a narrower statement that the effect is not confined to visible donor-subject evidence; it does not prove natural-background robustness.

## 5. Gate-0 comparability

- Existing image-level and group-disjoint manifests audited: 96 partition rows.
- Splits were not changed; the observed difference is not interpreted as a causal decomposition of same-group crossing alone.

## 6. Final limits

The study does not claim external-dataset or cross-camera generalization, backbone-agnostic behavior, clean-accuracy improvement, independent confirmation, causal proof of foreground sufficiency, superiority over CLIB, or inferiority of contrastive learning.

## 7. Integrity

- Frozen F0/F1 checkpoints present: 18.
- Frozen F0-2RGB checkpoints present: 9.
- No new checkpoint was created by this package.
- Official TEST accessed: false.
- Primary and five donor-sensitivity manifests remain unchanged.
