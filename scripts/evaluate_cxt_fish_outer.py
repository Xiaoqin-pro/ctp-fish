"""Evaluate one frozen CXT-Fish outer-test cell.

This is the single post-freeze read of one outer-test fold.  It never reads
outer-train or inner-dev pixels and never selects a checkpoint.  Context
swaps come only from the fold-specific outer-test manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.context_views import context_swap_view, foreground_blur_view
from metrics.classification import classification_metrics
from metrics.track_metrics import cluster_bootstrap_mean, track_balanced_accuracy
from models.resnet_context import ResNet18Context

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eval_transform(size: int):
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


class OuterContextDataset(Dataset):
    def __init__(self, records: pd.DataFrame, manifest: pd.DataFrame, cfg: dict, class_ids: list[str]):
        self.records = records.sort_values("image_path").reset_index(drop=True)
        self.transform = eval_transform(int(cfg["image_size"]))
        self.cfg = cfg
        self.class_to_index = {str(value): i for i, value in enumerate(class_ids)}
        self.donors = {key: group.set_index("swap_type").to_dict("index") for key, group in manifest.groupby("recipient_image_id")}
        missing = [str(value) for value in self.records.image_path if str(value) not in self.donors]
        if missing:
            raise ValueError(f"Missing context-swap rows for {len(missing)} outer-test images")
        for key, donor in self.donors.items():
            if set(donor) != {"same_class_cross_track", "cross_class"}:
                raise ValueError(f"Recipient does not have exactly two swap rows: {key}")
            if not all(bool(donor[name].get("supported", False)) for name in donor):
                raise ValueError(f"Unsupported swap row for recipient: {key}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        original = Image.open(row.image_path).convert("RGB")
        mask = Image.open(row.mask_path).convert("L")
        foreground = foreground_blur_view(
            original, mask, blur_kernel=21, blur_sigma=5.0, feather_radius=3
        )
        donor = self.donors[str(row.image_path)]
        same = context_swap_view(
            original, mask, Image.open(donor["same_class_cross_track"]["donor_image_path"]).convert("RGB"),
            feather_radius=3,
        )
        cross_row = donor["cross_class"]
        cross = context_swap_view(
            original, mask, Image.open(cross_row["donor_image_path"]).convert("RGB"), feather_radius=3
        )
        return tuple(self.transform(view) for view in (original, foreground, same, cross)) + (
            self.class_to_index[str(row.species_id)],
            self.class_to_index[str(cross_row["donor_species_id"])],
            str(row.image_path), str(row.group_id),
        )


def tiers(train: pd.DataFrame, class_ids: list[str]) -> dict[str, str]:
    counts = train.species_id.astype(str).value_counts()
    ordered = sorted([str(value) for value in class_ids], key=lambda value: (-int(counts.get(value, 0)), value))
    return {value: tier for tier, group in zip(("head", "mid", "tail"), np.array_split(np.asarray(ordered, dtype=object), 3)) for value in group.tolist()}


def metric_block(frame: pd.DataFrame, column: str, class_ids: list[str], tier_map: dict[str, str]) -> dict:
    metric = classification_metrics(frame.target.to_numpy(), frame[column].to_numpy(), list(range(len(class_ids))))
    per_class = dict(zip(map(str, class_ids), metric["per_class_f1"]))
    for tier in ("head", "mid", "tail"):
        metric[f"{tier}_f1"] = float(np.mean([per_class[key] for key, value in tier_map.items() if value == tier]))
    track = frame[["group_id", "target", column]].rename(columns={column: "prediction"}).copy()
    track["correct"] = track.target.eq(track.prediction)
    value, per_track = track_balanced_accuracy(track)
    metric["track_balanced_accuracy"] = value
    metric["track_balanced_accuracy_ci95"] = list(cluster_bootstrap_mean(per_track, seed=3407))
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_outer_validation_v1.yaml")
    parser.add_argument("--fold", choices=["1", "2", "3"], required=True)
    parser.add_argument("--method", choices=["F0", "F1"], required=True)
    parser.add_argument("--seed", type=int, choices=[3407, 2026, 17], required=True)
    parser.add_argument("--output-root", default="outputs/cxt_fish/final_outer_evaluation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    fold_root = ROOT / cfg["outer_manifest_output_root"] / f"fold_{args.fold}"
    test_path = fold_root / "outer_test.csv"
    swap_path = fold_root / "outer_context_swap.csv"
    checkpoint = ROOT / "outputs/cxt_fish/final_outer" / f"fold_{args.fold}" / f"{args.method}_seed{args.seed}" / "best.pt"
    for path in (test_path, swap_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = ROOT / args.output_root / f"fold_{args.fold}" / f"{args.method}_seed{args.seed}"
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
        print(json.dumps({"status": "dry_run", "fold": args.fold, "method": args.method, "seed": args.seed,
                          "outer_test_rows": len(records), "checkpoint": str(checkpoint),
                          "official_test_accessed": False}, indent=2))
        return

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_ids = [str(value) for value in state["class_ids"]]
    model = ResNet18Context(len(class_ids))
    model.load_state_dict(state["model"])
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
        "status": "complete", "fold": args.fold, "method": args.method, "seed": args.seed,
        "partition": "outer_test", "outer_test_accessed": True, "internal_test_accessed_before_outer_confirmation": False,
        "official_test_accessed": False, "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(cfg_path), "outer_test_sha256": sha256(test_path), "context_swap_sha256": sha256(swap_path),
        "metrics": metrics, "tier_map": tier_map, "n_images": len(frame), "n_groups": int(frame.group_id.nunique()),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "fold": args.fold, "method": args.method, "seed": args.seed,
                      "outer_test_rows": len(frame), "original_macro_f1": metrics["original"]["macro_f1"],
                      "cross_swap_macro_f1": metrics["cross_swap"]["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
