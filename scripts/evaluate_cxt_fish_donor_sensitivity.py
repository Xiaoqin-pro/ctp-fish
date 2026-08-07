"""Evaluate alternative donor realizations with frozen F0/F1 checkpoints only."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.context_views import context_swap_view  # noqa: E402
from metrics.classification import classification_metrics  # noqa: E402
from metrics.track_metrics import track_balanced_accuracy  # noqa: E402
from models.resnet_context import ResNet18Context  # noqa: E402
from scripts.evaluate_cxt_fish_outer import sha256, tiers  # noqa: E402


def eval_transform(size: int):
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


class CrossCompositeDataset(Dataset):
    def __init__(self, records: pd.DataFrame, manifest: pd.DataFrame, image_size: int, class_ids: list[str]):
        self.records = records.sort_values("image_path").reset_index(drop=True)
        self.transform = eval_transform(image_size)
        self.class_to_index = {str(value): i for i, value in enumerate(class_ids)}
        self.donors = {
            key: group.set_index("swap_type").to_dict("index")
            for key, group in manifest.groupby("recipient_image_path")
        }
        if set(self.donors) != set(self.records.image_path.astype(str)):
            raise ValueError("Donor sensitivity recipients do not equal outer-test images.")
        for key, donor in self.donors.items():
            if set(donor) != {"same_class_cross_track", "cross_class"}:
                raise ValueError(f"Invalid donor rows for {key}")
            if any(str(row.get("supported", "")).lower() != "true" for row in donor.values()):
                raise ValueError(f"Unsupported donor row for {key}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        original = Image.open(row.image_path).convert("RGB")
        mask = Image.open(row.mask_path).convert("L")
        donor = self.donors[str(row.image_path)]["cross_class"]
        cross = context_swap_view(
            original, mask,
            Image.open(donor["donor_image_path"]).convert("RGB"),
            feather_radius=3,
        )
        return self.transform(cross), self.class_to_index[str(row.species_id)], str(row.image_path), str(row.group_id), int(donor["donor_species_id"])


def metric_block(frame: pd.DataFrame, class_ids: list[str], tier_map: dict[str, str]) -> dict:
    metric = classification_metrics(frame.target.to_numpy(), frame.cross_swap.to_numpy(), list(range(len(class_ids))))
    per_class = dict(zip(map(str, class_ids), metric["per_class_f1"]))
    for tier in ("head", "mid", "tail"):
        metric[f"{tier}_f1"] = float(np.mean([per_class[key] for key, value in tier_map.items() if value == tier]))
    grouped = frame[["group_id", "target", "cross_swap"]].copy()
    grouped["correct"] = grouped.target.eq(grouped.cross_swap)
    metric["group_balanced_accuracy"], _ = track_balanced_accuracy(grouped)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_outer_validation_v1.yaml")
    parser.add_argument("--donor-seed", type=int, choices=[4101, 4102, 4103, 4104, 4105])
    parser.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--method", choices=["F0", "F1"], required=True)
    parser.add_argument("--model-seed", type=int, choices=[3407, 2026, 17], required=True)
    parser.add_argument("--output-root", default="outputs/cxt_fish/donor_sensitivity/evaluation")
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    fold_root = ROOT / cfg["outer_manifest_output_root"] / f"fold_{args.fold}"
    test_path = fold_root / "outer_test.csv"
    manifest_path = ROOT / "outputs/cxt_fish/donor_sensitivity/manifests" / f"seed_{args.donor_seed}" / f"fold_{args.fold}" / "outer_context_swap.csv"
    checkpoint = ROOT / "outputs/cxt_fish/final_outer" / f"fold_{args.fold}" / f"{args.method}_seed{args.model_seed}" / "best.pt"
    frozen_original = ROOT / "outputs/cxt_fish/final_outer_evaluation" / f"fold_{args.fold}" / f"{args.method}_seed{args.model_seed}" / "per_image.csv"
    for path in (test_path, manifest_path, checkpoint, frozen_original):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = ROOT / args.output_root / f"seed_{args.donor_seed}" / f"fold_{args.fold}" / f"{args.method}_seed{args.model_seed}"
    if (output / "metrics.json").is_file():
        print(json.dumps({"status": "already_complete", "output": str(output)}))
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")

    records = pd.read_csv(test_path)
    manifest = pd.read_csv(manifest_path)
    original = pd.read_csv(frozen_original)
    if set(records["image_path"].astype(str)) != set(original["image_path"].astype(str)):
        raise ValueError("Frozen original predictions do not pair one-to-one with outer-test.")
    group_check = records[["image_path", "group_id"]].merge(
        original[["image_path", "group_id"]], on="image_path", how="inner",
        suffixes=("_outer", "_frozen"), validate="one_to_one"
    )
    if len(group_check) != len(records) or not group_check["group_id_outer"].astype(str).equals(group_check["group_id_frozen"].astype(str)):
        raise ValueError("Frozen original group IDs do not match outer-test.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_ids = [str(value) for value in state["class_ids"]]
    model = ResNet18Context(len(class_ids))
    model.load_state_dict(state["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    train_records = pd.read_csv(fold_root / "outer_train.csv")
    tier_map = tiers(train_records, class_ids)
    dataset = CrossCompositeDataset(records, manifest, int(cfg["image_size"]), class_ids)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    paths, groups, targets, donors, predictions = [], [], [], [], []
    with torch.no_grad():
        for images, target, path, group, donor in loader:
            logits, _ = model(images.to(device, non_blocking=True))
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(target.tolist()); paths.extend(path); groups.extend(group); donors.extend(donor.tolist())
    frame = pd.DataFrame({"image_path": paths, "group_id": groups, "target": targets,
                          "cross_donor_target": donors, "cross_swap": predictions})
    frame = frame.merge(original[["image_path", "original"]], on="image_path", how="left", validate="one_to_one")
    if frame["original"].isna().any():
        raise ValueError("Original predictions missing after join.")
    metrics = metric_block(frame, class_ids, tier_map)
    donor_attraction = frame.cross_swap.eq(frame.cross_donor_target)
    original_correct = frame.original.eq(frame.target)
    context = {
        "prediction_agreement": float(frame.original.eq(frame.cross_swap).mean()),
        "dar": float(donor_attraction.mean()),
        "dar_flip": None if not bool(original_correct.any()) else float(donor_attraction[original_correct].mean()),
        "cross_swap_macro_f1": metrics["macro_f1"],
    }
    metrics_payload = {
        "status": "complete", "fold": args.fold, "method": args.method,
        "model_seed": args.model_seed, "donor_seed": args.donor_seed,
        "partition": "outer_test", "outer_test_accessed": True,
        "official_test_accessed": False, "post_hoc_sensitivity": True,
        "checkpoint_sha256": sha256(checkpoint), "config_sha256": sha256(cfg_path),
        "outer_test_sha256": sha256(test_path), "donor_manifest_sha256": sha256(manifest_path),
        "frozen_original_predictions_sha256": sha256(frozen_original),
        "metrics": {"cross_composite": metrics, "context": context},
        "tier_map": tier_map, "n_images": len(frame), "n_groups": int(frame.group_id.nunique()),
    }
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_image.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "fold": args.fold, "method": args.method,
                      "model_seed": args.model_seed, "donor_seed": args.donor_seed,
                      "cross_composite_macro_f1": metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
