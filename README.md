# CXT-Fish

Track-aware context-robust fish recognition study on the leakage-audited
F4K-16T subset of Fish4Knowledge.

The frozen study asks whether fish classifiers rely on correlated trajectory
context and whether foreground-sufficiency training improves robustness to
fixed cross-group context swaps. The primary method is a two-view supervised
objective:

```text
L = CE(f(x), y) + CE(f(x_fg), y)
```

The mask is used only to construct the training view. Inference uses ordinary
RGB input and one forward pass. The main confirmation is the 3-fold × 3-seed
ResNet18 outer evaluation; MobileNetV3-Large is a one-seed architecture-
sensitivity supplement. Route C is retained as a fixed two-stage mechanism
control, not as a faithful CLIB reproduction.

The official Fish4Knowledge test set was not accessed.

## Data

Fish4Knowledge is not stored in Git. Set `data_root` in `configs/f4k_audit.yaml` to a manually downloaded local copy, inspect its real file layout, and then set the filename parser regex in that same file. The audit refuses unparsed files rather than silently guessing metadata.

## Reproduction commands

```powershell
python scripts/check_environment.py
python scripts/audit_f4k.py --config configs/f4k_audit.yaml
python scripts/build_f4k_splits.py --config configs/f4k_split.yaml
pytest -q
```

To regenerate only the final statistics from frozen outputs (without
training or inference):

```powershell
python scripts/finalize_cxt_fish_statistics.py
```

See `reports/cxt_fish_claims_and_limits.md` for permitted claims and explicit
limitations.
