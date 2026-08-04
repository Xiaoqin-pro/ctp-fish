"""Evaluate one frozen route-C mask-guided outer-test cell."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.resnet_contrastive import ResNet18Contrastive
from scripts.evaluate_cxt_fish_outer import OuterContextDataset, metric_block, sha256, tiers


ROOT = Path(__file__).resolve().parents[1]


def validate_state(state: dict, *, fold: str, seed: int, cfg_path: Path, train_path: Path, dev_path: Path, class_ids: list[str]) -> None:
    expected = {"stage": "classifier", "fold": fold, "seed": seed, "config_sha256": sha256(cfg_path), "outer_train_sha256": sha256(train_path), "inner_dev_sha256": sha256(dev_path), "class_ids": class_ids, "outer_test_accessed": False, "official_test_accessed": False}
    for key, value in expected.items():
        if state.get(key) != value: raise ValueError(f"checkpoint provenance mismatch for {key}: {state.get(key)!r} != {value!r}")
    view_schema_sha = sha256(ROOT / "datasets" / "mask_contrastive_dataset.py")
    if state.get("view_schema_sha256") != view_schema_sha: raise ValueError("checkpoint view-schema hash mismatch")
    if not isinstance(state.get("initialization_sha256"), str) or len(state["initialization_sha256"]) != 64: raise ValueError("checkpoint initialization hash is missing or malformed")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/cxt_fish_mask_contrastive_outer_v1.yaml"); parser.add_argument("--fold", choices=["1", "2", "3"], required=True); parser.add_argument("--seed", type=int, choices=[3407, 2026, 17], required=True); parser.add_argument("--output-root", default="outputs/cxt_fish/mask_contrastive_outer_v1_evaluation"); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    cfg_path = ROOT / args.config; cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")); fold_root = ROOT / cfg["outer_manifest_root"] / f"fold_{args.fold}"; test_path, swap_path, train_path, dev_path = (fold_root / name for name in ("outer_test.csv", "outer_context_swap.csv", "outer_train.csv", "inner_dev.csv")); checkpoint = ROOT / cfg["output_root"] / f"fold_{args.fold}" / f"seed{args.seed}" / "best.pt"
    if not all(path.is_file() for path in (test_path, swap_path, train_path, dev_path)): raise FileNotFoundError("frozen fold manifest is incomplete")
    records, manifest = pd.read_csv(test_path), pd.read_csv(swap_path)
    if args.dry_run: print(json.dumps({"status": "dry_run", "fold": args.fold, "seed": args.seed, "outer_test_rows": len(records), "checkpoint": str(checkpoint), "official_test_accessed": False}, indent=2)); return
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    frozen = json.loads((ROOT / cfg["class_ids_path"]).read_text(encoding="utf-8")); class_ids = [str(value) for value in frozen["class_ids"]]; state = torch.load(checkpoint, map_location="cpu", weights_only=False); validate_state(state, fold=args.fold, seed=args.seed, cfg_path=cfg_path, train_path=train_path, dev_path=dev_path, class_ids=class_ids)
    model = ResNet18Contrastive(len(class_ids)); model.load_state_dict(state["model"], strict=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    if set(records.split.astype(str)) != {"test"} or set(manifest.recipient_image_path.astype(str)) != set(records.image_path.astype(str)): raise ValueError("outer-test/context-swap manifest mismatch")
    train_records = pd.read_csv(train_path); tier_map = tiers(train_records, class_ids); loader = DataLoader(OuterContextDataset(records, manifest, cfg, class_ids), batch_size=64, shuffle=False, num_workers=2, pin_memory=device.type == "cuda"); values = {name: [] for name in ("original", "foreground", "same_swap", "cross_swap")}; targets, donors, paths, groups = [], [], [], []
    with torch.no_grad():
        for original, foreground, same, cross, target, donor, path, group in loader:
            for name, images in zip(values, (original, foreground, same, cross)):
                logits, _ = model(images.to(device, non_blocking=True)); values[name].extend(logits.argmax(1).cpu().tolist())
            targets.extend(target.tolist()); donors.extend(donor.tolist()); paths.extend(path); groups.extend(group)
    frame = pd.DataFrame({"image_path": paths, "group_id": groups, "target": targets, "cross_donor_target": donors, **values}); metrics = {name: metric_block(frame, name, class_ids, tier_map) for name in values}; correct = frame.original.eq(frame.target); attracted = frame.cross_swap.eq(frame.cross_donor_target); metrics["context"] = {"prediction_agreement": float(frame.original.eq(frame.cross_swap).mean()), "dar": float(attracted.mean()), "dar_flip": None if not bool(correct.any()) else float(attracted[correct].mean()), "delta_foreground_macro_f1": metrics["original"]["macro_f1"] - metrics["foreground"]["macro_f1"], "delta_same_swap_macro_f1": metrics["original"]["macro_f1"] - metrics["same_swap"]["macro_f1"], "delta_cross_swap_macro_f1": metrics["original"]["macro_f1"] - metrics["cross_swap"]["macro_f1"]}
    output = ROOT / args.output_root / f"fold_{args.fold}" / f"seed{args.seed}"; output.mkdir(parents=True, exist_ok=True); frame.to_csv(output / "per_image.csv", index=False); (output / "metrics.json").write_text(json.dumps({"status": "complete", "fold": args.fold, "seed": args.seed, "partition": "outer_test", "outer_test_accessed": True, "official_test_accessed": False, "checkpoint_sha256": sha256(checkpoint), "config_sha256": sha256(cfg_path), "outer_test_sha256": sha256(test_path), "context_swap_sha256": sha256(swap_path), "metrics": metrics, "tier_map": tier_map, "n_images": len(frame), "n_groups": int(frame.group_id.nunique())}, indent=2), encoding="utf-8"); print(json.dumps({"status": "complete", "fold": args.fold, "seed": args.seed, "original_macro_f1": metrics["original"]["macro_f1"], "cross_swap_macro_f1": metrics["cross_swap"]["macro_f1"]}, indent=2))


if __name__ == "__main__": main()
