"""Compute frozen F0 original-vs-foreground margin risk on one locked split."""
from __future__ import annotations

import argparse
import hashlib
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
from datasets.context_views import foreground_blur_view
from models.resnet_classifier import build_resnet18


class RiskDataset(Dataset):
    def __init__(self, records: pd.DataFrame, size: int, blur_kernel: int, blur_sigma: float, feather_radius: int):
        self.records = records.reset_index(drop=True); self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])]); self.kwargs = {"blur_kernel": blur_kernel, "blur_sigma": blur_sigma, "feather_radius": feather_radius}
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        row = self.records.iloc[index]; image = Image.open(row.image_path).convert("RGB"); mask = Image.open(row.mask_path).convert("L"); fg = foreground_blur_view(image, mask, **self.kwargs)
        return self.transform(image), self.transform(fg), int(row.target), row.image_path, str(row.species_id), str(row.group_id)


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def run(model, loader, device):
    rows = []
    for original, foreground, target, paths, species, groups in loader:
        a, b = model(original.to(device)), model(foreground.to(device)); target = target.to(device)
        def margin(logits):
            correct = logits.gather(1, target[:, None]).squeeze(1); other = logits.clone(); other.scatter_(1, target[:, None], float("-inf")); return correct - other.max(1).values
        mo, mf = margin(a).cpu().numpy(), margin(b).cpu().numpy(); correct = mo > 0
        for path, cls, group, orig, fg, valid in zip(paths, species, groups, mo, mf, correct): rows.append({"image_path": path, "species_id": cls, "group_id": group, "margin_orig": float(orig), "margin_fg": float(fg), "orig_correct": bool(valid), "image_risk": float(max(0.0, orig - fg) if valid else 0.0)})
    return pd.DataFrame(rows)


def add_track_fields(frame: pd.DataFrame, normalize_weights: bool) -> pd.DataFrame:
    track = frame.groupby(["species_id", "group_id"], as_index=False).agg(track_risk=("image_risk", "median"), track_positive_fraction=("orig_correct", "mean")); track["raw_weight"] = 0.0
    for _, index in track.groupby("species_id").groups.items():
        idx = list(index); positive = track.loc[idx, "track_risk"] > 1e-8; positive_idx = track.loc[idx].loc[positive].index; n = len(positive_idx)
        if n: track.loc[positive_idx, "raw_weight"] = (track.loc[positive_idx, "track_risk"].rank(method="average") - 1) / max(n - 1, 1)
    if normalize_weights:
        mean = float(track.loc[track.raw_weight.gt(0), "raw_weight"].mean()) if track.raw_weight.gt(0).any() else 1.0; track["track_weight"] = track.raw_weight / max(mean, 1e-8)
    else: track["track_weight"] = track.raw_weight
    track["track_rank_within_species"] = track.groupby("species_id")["track_risk"].rank(method="average")
    return frame.merge(track[["species_id", "group_id", "track_risk", "track_weight", "track_rank_within_species"]], on=["species_id", "group_id"], validate="many_to_one")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/cxt_fish_phase1e.yaml"); parser.add_argument("--split", choices=["train", "val"], required=True); parser.add_argument("--output", required=True); args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    split = pd.read_csv(cfg["track_split_path"]); metadata = pd.read_csv(cfg["metadata_path"]); records = metadata.merge(split[["image_path", "split"]], on="image_path", validate="one_to_one"); records = records.loc[records.split.eq(args.split)].copy(); records["target"] = records.species_id.astype(str).map({v: i for i, v in enumerate(sorted(metadata.species_id.astype(str).unique()))})
    state = torch.load(cfg["base_checkpoint"], map_location="cpu", weights_only=False); class_ids = [str(value) for value in state["class_ids"]]; records = records.loc[records.species_id.astype(str).isin(class_ids)].copy(); records["target"] = records.species_id.astype(str).map({value: index for index, value in enumerate(class_ids)}); model = build_resnet18(len(class_ids)); model.load_state_dict(state["model"]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    loader = DataLoader(RiskDataset(records, int(cfg["image_size"]), int(cfg["foreground_blur_kernel"]), float(cfg["foreground_blur_sigma"]), int(cfg["mask_feather_radius"])), batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=2, pin_memory=device.type == "cuda"); result = add_track_fields(run(model, loader, device), normalize_weights=args.split == "train"); result["source_checkpoint_sha256"] = sha256(Path(cfg["base_checkpoint"])); output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); result.to_csv(output, index=False); print({"split": args.split, "rows": len(result), "tracks": result[["species_id", "group_id"]].drop_duplicates().shape[0], "sha256": sha256(output)})


if __name__ == "__main__": main()
