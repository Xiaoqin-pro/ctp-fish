"""Gate-0 training entry point. It explicitly rejects locked outer folds."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from tools.reproducibility import seed_everything

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--split",choices=["image","track"],required=True); parser.add_argument("--outer-folds"); parser.add_argument("--dry-run",action="store_true"); args=parser.parse_args()
    if args.outer_folds: raise ValueError("Outer evaluation folds are locked during Gate-0.")
    cfg=yaml.safe_load(Path(args.config).read_text()); seed_everything(int(cfg["seed"]))
    if args.dry_run: print({"model":"torchvision ResNet18 ImageNet", "split":args.split, "epochs":cfg["epochs"], "forbidden_methods":"no reweighting, no trajectory sampler, no contrastive/prototype method"}); return
    raise RuntimeError("Training is intentionally not started until data audit and split audit have completed.")
if __name__=="__main__": main()
