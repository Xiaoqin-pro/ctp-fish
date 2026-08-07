"""Evaluate frozen DLE checkpoints on the train-only, group-disjoint inner folds."""
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
from datasets.phase1c_dataset import MEAN, STD
from metrics.classification import classification_metrics
from metrics.track_metrics import cluster_bootstrap_mean, track_balanced_accuracy
from models.dle_resnet import DLEResNet18
from scripts.build_phase1c_context_swap_manifest import build_manifest
from tools.io_utils import atomic_csv_dump, atomic_json_dump


VARIANTS = ("Q0", "Q1", "Q2a", "Q2b")


def eval_transform(size: int):
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class InnerEvaluationDataset(Dataset):
    def __init__(self, records: pd.DataFrame, manifest: pd.DataFrame, cfg: dict, class_ids: list[str]):
        self.records = records.sort_values("image_path").reset_index(drop=True)
        self.class_to_index = {str(value): i for i, value in enumerate(class_ids)}
        self.transform = eval_transform(int(cfg["image_size"]))
        self.cfg = cfg
        self.donors = {key: group.set_index("swap_type").to_dict("index") for key, group in manifest.groupby("recipient_image_id")}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        image = Image.open(row.image_path).convert("RGB")
        mask = Image.open(row.mask_path).convert("L")
        foreground = foreground_blur_view(image, mask, blur_kernel=int(self.cfg["foreground_blur_kernel"]), blur_sigma=float(self.cfg["foreground_blur_sigma"]), feather_radius=int(self.cfg["mask_feather_radius"]))
        donor = self.donors[str(row.image_path)]
        same = context_swap_view(image, mask, Image.open(donor["same_class_cross_track"]["donor_image_path"]).convert("RGB"), feather_radius=int(self.cfg["mask_feather_radius"]))
        cross = context_swap_view(image, mask, Image.open(donor["cross_class"]["donor_image_path"]).convert("RGB"), feather_radius=int(self.cfg["mask_feather_radius"]))
        image = transforms.functional.center_crop(transforms.functional.resize(image, 256), (int(self.cfg["image_size"]), int(self.cfg["image_size"])))
        mask = transforms.functional.center_crop(transforms.functional.resize(mask, 256, Image.Resampling.NEAREST), (int(self.cfg["image_size"]), int(self.cfg["image_size"])))
        return (self.transform(image), self.transform(foreground), self.transform(same), self.transform(cross), transforms.functional.to_tensor(mask).clamp(0, 1), self.class_to_index[str(row.species_id)], self.class_to_index[str(donor["cross_class"]["donor_species_id"])], str(row.image_path), str(row.group_id))


def tiers(train: pd.DataFrame, class_ids: list[str]) -> dict[str, str]:
    counts = train.species_id.astype(str).value_counts()
    ordered = sorted([str(value) for value in class_ids], key=lambda value: (-int(counts.get(value, 0)), value))
    groups = np.array_split(np.asarray(ordered, dtype=object), 3)
    return {value: tier for tier, group in zip(("head", "mid", "tail"), groups) for value in group.tolist()}


def metric_block(frame: pd.DataFrame, column: str, class_ids: list[str], tier_map: dict[str, str]) -> dict:
    metric = classification_metrics(frame.target.to_numpy(), frame[column].to_numpy(), list(range(len(class_ids))))
    per_class = dict(zip(map(str, class_ids), metric["per_class_f1"]))
    metric["head_f1"] = float(np.mean([per_class[key] for key, tier in tier_map.items() if tier == "head"]))
    metric["mid_f1"] = float(np.mean([per_class[key] for key, tier in tier_map.items() if tier == "mid"]))
    metric["tail_f1"] = float(np.mean([per_class[key] for key, tier in tier_map.items() if tier == "tail"]))
    track_frame = frame[["group_id", "target", column]].rename(columns={column: "prediction"}).copy()
    track_frame["correct"] = track_frame.target.eq(track_frame.prediction)
    value, per_track = track_balanced_accuracy(track_frame)
    metric["track_balanced_accuracy"] = value
    metric["track_balanced_accuracy_ci95"] = list(cluster_bootstrap_mean(per_track, seed=3407))
    return metric


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/cxt_fish_phase2_dle.yaml"); parser.add_argument("--root", default="outputs/cxt_fish/phase2_dle"); parser.add_argument("--output", default="experiments/dle_inner_evaluation.json")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); class_ids = [str(value) for value in json.loads(Path(cfg["class_ids_path"]).read_text(encoding="utf-8"))["class_ids"]]
    inner = json.loads(Path(cfg["inner_folds_path"]).read_text(encoding="utf-8")); metadata = pd.read_csv(cfg["metadata_path"]); split = pd.read_csv(cfg["track_split_path"])
    train_all = metadata.merge(split.loc[split.split == "train", ["image_path", "split"]], on="image_path", validate="one_to_one")
    root = Path(args.root); records_out = []; summary = []
    for fold in (0, 1):
        heldout = set(inner["folds"][fold]["heldout_group_ids"]); heldout_records = train_all[train_all.group_id.astype(str).isin(heldout)].reset_index(drop=True); train_records = train_all[~train_all.group_id.astype(str).isin(heldout)]
        if heldout_records.empty or set(heldout_records.split) != {"train"}: raise ValueError("inner evaluation records must come only from frozen train partition")
        manifest_records = heldout_records.copy(); manifest_records["split"] = "val"; manifest = build_manifest(manifest_records, seed=3407)
        dataset = InnerEvaluationDataset(heldout_records, manifest, cfg, class_ids); loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=2)
        for variant in VARIANTS:
            checkpoint = root / f"fold{fold}_{variant}_seed{int(cfg['pilot_seed'])}" / "best.pt"
            if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); state = torch.load(checkpoint, map_location=device, weights_only=False); model = DLEResNet18(len(class_ids)).to(device); model.load_state_dict(state["model"]); model.eval()
            values = {name: [] for name in ("original", "foreground", "masked", "same_swap", "cross_swap")}; targets = []; donors = []; paths = []; groups = []
            with torch.no_grad():
                for original, foreground, same, cross, mask, target, donor, path, group in loader:
                    for name, image in (("original", original), ("foreground", foreground), ("same_swap", same), ("cross_swap", cross)):
                        values[name].extend(model(image.to(device))["global_logits"].argmax(1).cpu().tolist())
                    values["masked"].extend(model(original.to(device), mask.to(device))["masked_logits"].argmax(1).cpu().tolist()); targets.extend(target.tolist()); donors.extend(donor.tolist()); paths.extend(path); groups.extend(group)
            frame = pd.DataFrame({"image_path": paths, "group_id": groups, "target": targets, "cross_donor_target": donors, **values}); tier_map = tiers(train_records, class_ids); metrics = {name: metric_block(frame, name, class_ids, tier_map) for name in values}
            original_correct = frame.original.eq(frame.target); donor_attracted = frame.cross_swap.eq(frame.cross_donor_target)
            metrics["context"] = {"prediction_agreement": float(frame.original.eq(frame.cross_swap).mean()), "dar": float(donor_attracted.mean()), "dar_flip": float(donor_attracted[original_correct].mean()) if bool(original_correct.any()) else None, "delta_cross_swap_macro_f1": metrics["original"]["macro_f1"] - metrics["cross_swap"]["macro_f1"]}
            summary.append({"fold": fold, "variant": variant, "checkpoint": str(checkpoint), "best_epoch": int(state.get("history", [{}])[-1].get("epoch", -1)), "metrics": metrics}); records_out.append((fold, variant, frame))
    payload = {"schema_version": "dle_inner_evaluation_v1", "partition": "train_only_inner_heldout", "folds": summary, "current_val_accessed": False, "internal_test_accessed": False, "outer_folds_accessed": False, "official_test_accessed": False}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); atomic_json_dump(payload, output, overwrite=True)
    for fold, variant, frame in records_out: atomic_csv_dump(frame, output.parent / f"dle_inner_fold{fold}_{variant}_per_image.csv", overwrite=True)
    print(json.dumps({"folds": 2, "variants": len(VARIANTS), "output": str(output), "current_val_accessed": False}, indent=2))


if __name__ == "__main__": main()
