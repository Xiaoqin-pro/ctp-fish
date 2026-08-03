"""Validation-only context evaluation for frozen CXT-Fish Phase 1C checkpoints."""
from __future__ import annotations

import argparse
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
from models.resnet_classifier import build_resnet18
from models.resnet_context import ResNet18Context
from tools.io_utils import atomic_csv_dump, atomic_json_dump


def evaluation_transform(size: int):
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


class ContextEvaluationDataset(Dataset):
    def __init__(self, records: pd.DataFrame, manifest: pd.DataFrame, transform, cfg: dict, class_ids: list[str]) -> None:
        if set(records.split.unique()) != {"val"}: raise ValueError("Phase 1C evaluation only accepts validation records")
        if set(manifest.swap_type.unique()) != {"same_class_cross_track", "cross_class"}: raise ValueError("invalid swap manifest types")
        if manifest.recipient_image_id.nunique() != len(records) or len(manifest) != 2 * len(records): raise ValueError("swap manifest must contain exactly two rows per validation recipient")
        self.records = records.sort_values("image_path").reset_index(drop=True); self.transform = transform; self.class_to_index = {v: i for i, v in enumerate(class_ids)}; self.cfg = cfg
        self.donors = {key: group.set_index("swap_type").to_dict("index") for key, group in manifest.groupby("recipient_image_id")}

    def __len__(self): return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]; donor = self.donors[str(row.image_path)]
        original = Image.open(row.image_path).convert("RGB"); mask = Image.open(row.mask_path).convert("L")
        foreground = foreground_blur_view(original, mask, blur_kernel=int(self.cfg["foreground_blur_kernel"]), blur_sigma=float(self.cfg["foreground_blur_sigma"]), feather_radius=int(self.cfg["mask_feather_radius"]))
        same = context_swap_view(original, mask, Image.open(donor["same_class_cross_track"]["donor_image_path"]).convert("RGB"), feather_radius=int(self.cfg["mask_feather_radius"]))
        cross_row = donor["cross_class"]
        cross = context_swap_view(original, mask, Image.open(cross_row["donor_image_path"]).convert("RGB"), feather_radius=int(self.cfg["mask_feather_radius"]))
        return tuple(self.transform(view) for view in (original, foreground, same, cross)) + (self.class_to_index[str(row.species_id)], self.class_to_index[str(cross_row["donor_species_id"])], str(row.image_path), str(row.group_id))


def load_checkpoint(path: Path, device: torch.device):
    state = torch.load(path, map_location=device, weights_only=False); class_ids = state["class_ids"]
    keys = state["model"].keys()
    model = ResNet18Context(len(class_ids)) if any(key.startswith("encoder.") for key in keys) else build_resnet18(len(class_ids))
    model.load_state_dict(state["model"]); return model.to(device).eval(), state, class_ids


def predict(model, images: torch.Tensor) -> torch.Tensor:
    result = model(images)
    return result[0] if isinstance(result, tuple) else result


def metric_block(frame: pd.DataFrame, column: str, class_ids: list[str]) -> tuple[dict, pd.DataFrame]:
    values = classification_metrics(frame.target.to_numpy(), frame[column].to_numpy(), list(range(len(class_ids))))
    track_frame = frame[["group_id", "target", column]].rename(columns={column: "prediction"}).copy()
    track_frame["correct"] = track_frame.target.eq(track_frame.prediction)
    track_value, tracks = track_balanced_accuracy(track_frame)
    values |= {"track_balanced_accuracy": track_value, "track_balanced_accuracy_ci95": list(cluster_bootstrap_mean(tracks, seed=3407)), "class_ids": class_ids}
    return values, tracks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_phase1c.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluation-name", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if Path(args.evaluation_name).name != args.evaluation_name: raise ValueError("evaluation-name must be a simple directory name")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); split_path = Path(cfg["track_split_path"])
    if split_path.resolve() == Path(cfg["outer_folds_path"]).resolve(): raise ValueError("outer folds are locked")
    metadata, split, manifest = pd.read_csv(cfg["metadata_path"]), pd.read_csv(split_path), pd.read_csv(cfg["swap_manifest_path"])
    records = metadata.merge(split.loc[split.split == "val", ["image_path", "split"]], on="image_path", validate="one_to_one")
    if records.empty or set(records.split) != {"val"}: raise ValueError("validation records are required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, state, class_ids = load_checkpoint(Path(args.checkpoint), device)
    dataset = ContextEvaluationDataset(records, manifest, evaluation_transform(int(cfg["image_size"])), cfg, class_ids)
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    values = {name: [] for name in ("original", "foreground", "same_swap", "cross_swap")}; targets=[]; donors=[]; paths=[]; groups=[]
    with torch.no_grad():
        for original, foreground, same, cross, target, donor, path, group in loader:
            for name, image in zip(values, (original, foreground, same, cross)):
                values[name].extend(predict(model, image.to(device)).argmax(1).cpu().tolist())
            targets.extend(target.tolist()); donors.extend(donor.tolist()); paths.extend(path); groups.extend(group)
    frame = pd.DataFrame({"image_path": paths, "group_id": groups, "target": targets, "cross_donor_target": donors, **values})
    metrics: dict[str, dict] = {}; tracks: dict[str, pd.DataFrame] = {}
    for name in values:
        metrics[name], tracks[name] = metric_block(frame, name, class_ids)
    original_correct = frame.original.eq(frame.target)
    cross_donor = frame.cross_swap.eq(frame.cross_donor_target)
    metrics["context"] = {
        "prediction_agreement": float(frame.original.eq(frame.cross_swap).mean()),
        "dar": float(cross_donor.mean()),
        "dar_flip": None if not bool(original_correct.any()) else float(cross_donor[original_correct].mean()),
        "delta_foreground_macro_f1": metrics["original"]["macro_f1"] - metrics["foreground"]["macro_f1"],
        "delta_same_swap_macro_f1": metrics["original"]["macro_f1"] - metrics["same_swap"]["macro_f1"],
        "delta_cross_swap_macro_f1": metrics["original"]["macro_f1"] - metrics["cross_swap"]["macro_f1"],
    }
    metrics |= {"partition": "val", "checkpoint": str(args.checkpoint), "checkpoint_variant": state.get("variant", "f0_a1"), "internal_test_accessed": False, "outer_folds_accessed": False}
    output = Path(args.checkpoint).parent / f"evaluation_{args.evaluation_name}"
    atomic_csv_dump(frame, output / "per_image_val.csv", args.overwrite)
    atomic_json_dump(metrics, output / "metrics_val.json", args.overwrite)
    for name, value in tracks.items(): atomic_csv_dump(value, output / f"per_track_{name}_val.csv", args.overwrite)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
