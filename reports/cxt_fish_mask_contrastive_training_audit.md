# Route C training audit

This is a no-training audit of the already frozen route-C outer baseline.
It does not reopen outer-test data or change any checkpoint, metric, or
protocol decision.

## Scope and access

- Cells audited: 9/9 (folds 1--3; seeds 17, 2026, and 3407).
- Each cell contains `pretrain_last.pt`, `best.pt`, `last.pt`,
  `training_curve.csv`, and `run_metadata.json`.
- All metadata records the frozen
  `mask_guided_subject_nonprimary` variant.
- `outer_test_accessed=false` and `official_test_accessed=false` in every
  training metadata file. This audit reads only training curves and metadata.

## Training integrity

- All nine pretraining stages completed 40 epochs.
- Pretraining contrastive loss was finite in every row and decreased from the
  first to the final logged epoch in all nine cells (first: 0.0904--0.1152;
  final: 0.0014--0.0022).
- Classifier curves were finite in every row. Classifier early stopping ended
  at epochs 31--40, with the best inner-development epoch in 24--38.
- No cell selected a checkpoint using outer-test data. The best checkpoint was
  selected by the frozen inner-development accuracy rule.
- No cell was still improving at the final logged epoch according to the
  recorded best checkpoint; the best epoch was earlier than the 40-epoch
  budget in all nine cells.

## Interpretation

The route-C implementation completed its fixed two-stage budget without
non-finite values, provenance failures, or an obvious collapsed pretraining
run. This supports treating route C as a valid reproducible mechanism control.
It does **not** turn route C into a faithful reproduction of the original CLIB
training recipe, nor does it justify another route-C training sweep.

The outer report remains the scientific result; this document only records the
engineering audit requested before moving to the fixed MobileNetV3 transfer
check.
