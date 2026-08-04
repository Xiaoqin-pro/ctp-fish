"""Evaluate one frozen CLIB-style outer-test cell with the CXT evaluator."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_clib_style_outer_v1_1.yaml")
    parser.add_argument("--fold", choices=["1", "2", "3"], required=True)
    parser.add_argument("--seed", type=int, choices=[3407, 2026, 17], required=True)
    parser.add_argument("--output-root", default="outputs/cxt_fish/clib_style_outer_v1_1_evaluation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    fold_root = ROOT / cfg["outer_manifest_root"] / f"fold_{args.fold}"
    test_path = fold_root / "outer_test.csv"
    swap_path = fold_root / "outer_context_swap.csv"
    checkpoint = ROOT / cfg["output_root"] / f"fold_{args.fold}" / f"seed{args.seed}" / "best.pt"
    for path in (test_path, swap_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = ROOT / args.output_root / f"fold_{args.fold}" / f"seed{args.seed}"
    if (output / "metrics.json").is_file():
        print(json.dumps({"status": "already_complete", "output": str(output)}))
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty evaluation output: {output}")
    records = pd.read_csv(test_path)
    manifest = pd.read_csv(swap_path)
    if set(records.split.astype(str)) != {"test"}:
        raise ValueError("outer_test.csv must contain only test records")
    if set(manifest.recipient_image_path.astype(str)) != set(records.image_path.astype(str)):
        raise ValueError("context-swap manifest recipients do not equal outer-test images")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "fold": args.fold, "seed": args.seed,
                          "outer_test_rows": len(records), "checkpoint": str(checkpoint),
                          "official_test_accessed": False}, indent=2))
        return
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_ids = [str(value) for value in state["class_ids"]]
    model = ResNet18Contrastive(len(class_ids))
    model.load_state_dict(state["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    train_records = pd.read_csv(fold_root / "outer_train.csv")
    tier_map = tiers(train_records, class_ids)
    dataset = OuterContextDataset(records, manifest, cfg, class_ids)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    values = {name: [] for name in ("original", "foreground", "same_swap", "cross_swap")}
    targets, donors, paths, groups = [], [], [], []
    with torch.no_grad():
        for original, foreground, same, cross, target, donor, path, group in loader:
            for name, images in zip(values, (original, foreground, same, cross)):
                logits, _ = model(images.to(device, non_blocking=True))
                values[name].extend(logits.argmax(1).cpu().tolist())
            targets.extend(target.tolist()); donors.extend(donor.tolist()); paths.extend(path); groups.extend(group)
    frame = pd.DataFrame({"image_path": paths, "group_id": groups, "target": targets, "cross_donor_target": donors, **values})
    metrics = {name: metric_block(frame, name, class_ids, tier_map) for name in values}
    original_correct = frame.original.eq(frame.target)
    donor_attraction = frame.cross_swap.eq(frame.cross_donor_target)
    metrics["context"] = {
        "prediction_agreement": float(frame.original.eq(frame.cross_swap).mean()),
        "dar": float(donor_attraction.mean()),
        "dar_flip": None if not bool(original_correct.any()) else float(donor_attraction[original_correct].mean()),
        "delta_foreground_macro_f1": metrics["original"]["macro_f1"] - metrics["foreground"]["macro_f1"],
        "delta_same_swap_macro_f1": metrics["original"]["macro_f1"] - metrics["same_swap"]["macro_f1"],
        "delta_cross_swap_macro_f1": metrics["original"]["macro_f1"] - metrics["cross_swap"]["macro_f1"],
    }
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_image.csv", index=False)
    (output / "metrics.json").write_text(json.dumps({
        "status": "complete", "fold": args.fold, "seed": args.seed, "partition": "outer_test",
        "outer_test_accessed": True, "official_test_accessed": False,
        "checkpoint_sha256": sha256(checkpoint), "config_sha256": sha256(cfg_path),
        "outer_test_sha256": sha256(test_path), "context_swap_sha256": sha256(swap_path),
        "metrics": metrics, "tier_map": tier_map, "n_images": len(frame),
        "n_groups": int(frame.group_id.nunique()),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "fold": args.fold, "seed": args.seed,
                      "outer_test_rows": len(frame), "original_macro_f1": metrics["original"]["macro_f1"],
                      "cross_swap_macro_f1": metrics["cross_swap"]["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
