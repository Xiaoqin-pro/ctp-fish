"""Evaluate the six-cell MobileNetV3 transfer check after explicit unlock."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.mobilenet_context import MobileNetV3LargeContext
from scripts.evaluate_cxt_fish_outer import OuterContextDataset, metric_block, sha256, tiers

ROOT = Path(__file__).resolve().parents[1]


def assert_six_training_cells(root: Path, cfg: dict) -> None:
    for fold in ("1", "2", "3"):
        for method in ("MV0", "MV1"):
            run = root / f"fold_{fold}" / f"{method}_seed3407"
            required = ("best.pt", "last.pt", "training_curve.csv", "run_metadata.json")
            missing = [name for name in required if not (run / name).is_file()]
            if missing:
                raise RuntimeError(f"six-cell training gate failed for {run}: missing {missing}")
            metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
            if metadata.get("outer_test_accessed") is not False or metadata.get("official_test_accessed") is not False:
                raise RuntimeError(f"training access flags invalid: {run}")
            if metadata.get("protocol_variant") != cfg["protocol_variant"]:
                raise RuntimeError(f"protocol variant mismatch: {run}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_mobilenet_outer_v1.yaml")
    parser.add_argument("--fold", choices=["1", "2", "3"], required=True)
    parser.add_argument("--method", choices=["MV0", "MV1"], required=True)
    parser.add_argument("--seed", type=int, choices=[3407], default=3407)
    parser.add_argument("--output-root", default="outputs/cxt_fish/mobilenet_outer_evaluation")
    parser.add_argument("--unlock-outer-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    train_root = ROOT / "outputs/cxt_fish/mobilenet_outer"
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "outer_test_loaded": False, "official_test_accessed": False, "six_cell_gate_checked": False}, indent=2)); return
    if not args.unlock_outer_test:
        raise PermissionError("MobileNet outer evaluation requires --unlock-outer-test")
    assert_six_training_cells(train_root, cfg)
    fold_root = ROOT / cfg["outer_manifest_output_root"] / f"fold_{args.fold}"
    test_path, swap_path = fold_root / "outer_test.csv", fold_root / "outer_context_swap.csv"
    checkpoint = train_root / f"fold_{args.fold}" / f"{args.method}_seed{args.seed}" / "best.pt"
    output = ROOT / args.output_root / f"fold_{args.fold}" / f"{args.method}_seed{args.seed}"
    if (output / "metrics.json").is_file():
        print(json.dumps({"status": "already_complete", "output": str(output)})); return
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Refusing to overwrite {output}")
    records = pd.read_csv(test_path); manifest = pd.read_csv(swap_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_ids = [str(value) for value in state["class_ids"]]
    model = MobileNetV3LargeContext(len(class_ids)); model.load_state_dict(state["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    train_records = pd.read_csv(fold_root / "outer_train.csv"); tier_map = tiers(train_records, class_ids)
    dataset = OuterContextDataset(records, manifest, cfg, class_ids)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    values = {name: [] for name in ("original", "foreground", "same_swap", "cross_swap")}; targets=[]; donors=[]; paths=[]; groups=[]
    with torch.no_grad():
        for original, foreground, same, cross, target, donor, path, group in loader:
            for name, images in zip(values, (original, foreground, same, cross)):
                logits, _ = model(images.to(device, non_blocking=True)); values[name].extend(logits.argmax(1).cpu().tolist())
            targets.extend(target.tolist()); donors.extend(donor.tolist()); paths.extend(path); groups.extend(group)
    frame = pd.DataFrame({"image_path": paths, "group_id": groups, "target": targets, "cross_donor_target": donors, **values})
    metrics = {name: metric_block(frame, name, class_ids, tier_map) for name in values}; original_correct = frame.original.eq(frame.target); attraction = frame.cross_swap.eq(frame.cross_donor_target)
    metrics["context"] = {"prediction_agreement": float(frame.original.eq(frame.cross_swap).mean()), "dar": float(attraction.mean()), "dar_flip": None if not bool(original_correct.any()) else float(attraction[original_correct].mean()), "delta_foreground_macro_f1": metrics["original"]["macro_f1"] - metrics["foreground"]["macro_f1"], "delta_same_swap_macro_f1": metrics["original"]["macro_f1"] - metrics["same_swap"]["macro_f1"], "delta_cross_swap_macro_f1": metrics["original"]["macro_f1"] - metrics["cross_swap"]["macro_f1"]}
    output.mkdir(parents=True, exist_ok=True); frame.to_csv(output / "per_image.csv", index=False)
    (output / "metrics.json").write_text(json.dumps({"status": "complete", "fold": args.fold, "method": args.method, "seed": args.seed, "partition": "outer_test", "outer_test_accessed": True, "official_test_accessed": False, "checkpoint_sha256": sha256(checkpoint), "config_sha256": sha256(cfg_path), "outer_test_sha256": sha256(test_path), "context_swap_sha256": sha256(swap_path), "metrics": metrics, "tier_map": tier_map, "n_images": len(frame), "n_groups": int(frame.group_id.nunique())}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "fold": args.fold, "method": args.method, "seed": args.seed, "original_macro_f1": metrics["original"]["macro_f1"], "cross_swap_macro_f1": metrics["cross_swap"]["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
