# CTP-Fish

Gate-0 feasibility audit for Fish4Knowledge trajectory-aware fish classification.

This repository is deliberately independent from `underwater-calibration`. Gate-0 audits data structure, trajectory grouping, image-level versus trajectory-level splits, nearest-neighbour leakage risk, a plain ResNet18 baseline, and background correlation. It does not implement any proposed trajectory method.

## Data

Fish4Knowledge is not stored in Git. Set `data_root` in `configs/f4k_audit.yaml` to a manually downloaded local copy, inspect its real file layout, and then set the filename parser regex in that same file. The audit refuses unparsed files rather than silently guessing metadata.

## Initial commands

```powershell
python scripts/check_environment.py
python scripts/audit_f4k.py --config configs/f4k_audit.yaml
python scripts/build_f4k_splits.py --config configs/f4k_split.yaml
pytest -q
```
