"""Validation-only evaluator for frozen CXT-Fish Phase 1B checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.f4k_dataset import F4KDataset
from metrics.classification import classification_metrics
from metrics.track_metrics import cluster_bootstrap_mean, track_balanced_accuracy
from models.resnet_contrastive import ResNet18Contrastive
from tools.io_utils import atomic_csv_dump, atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_phase1b.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", choices=["c0", "c1", "c2"], required=True)
    parser.add_argument("--evaluation-name", default="phase1b_val")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if Path(args.evaluation_name).name != args.evaluation_name:
        raise ValueError("evaluation-name must be a simple directory name.")
    cfg = yaml.safe_load(Path(args.config).read_text())
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("variant") != args.variant:
        raise ValueError("Checkpoint variant does not match requested evaluation variant.")
    metadata, split = pd.read_csv(cfg["metadata_path"]), pd.read_csv(cfg["track_split_path"])
    records = metadata.merge(split[["image_path", "split"]], on="image_path", validate="one_to_one")
    records = records[records.split == "val"].reset_index(drop=True)
    if records.empty: raise ValueError("Validation partition is empty.")
    class_ids = state["class_ids"]
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(int(cfg["image_size"])), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    dataset = F4KDataset(records, transform, class_ids=class_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=int(cfg["p"]) * int(cfg["q"]) * int(cfg["k"]), shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    model = ResNet18Contrastive(len(class_ids)); model.load_state_dict(state["model"]); model.to(device).eval()
    targets: list[int] = []; predictions: list[int] = []; paths: list[str] = []; groups: list[str] = []
    with torch.no_grad():
        for images, labels, image_paths, group_ids in loader:
            logits, _ = model(images.to(device)); targets.extend(labels.tolist()); predictions.extend(logits.argmax(1).cpu().tolist()); paths.extend(image_paths); groups.extend(group_ids)
    frame = pd.DataFrame({"image_path": paths, "group_id": groups, "target": targets, "prediction": predictions})
    frame["correct"] = frame.target.eq(frame.prediction)
    track_accuracy, per_track = track_balanced_accuracy(frame)
    metrics = classification_metrics(np.asarray(targets), np.asarray(predictions), list(range(len(class_ids)))) | {
        "track_balanced_accuracy": track_accuracy,
        "track_balanced_accuracy_ci95": list(cluster_bootstrap_mean(per_track, seed=3407)),
        "class_ids": class_ids, "partition": "val", "variant": args.variant,
        "internal_test_accessed": False, "outer_folds_accessed": False,
    }
    output = Path(args.checkpoint).parent / f"evaluation_{args.evaluation_name}"
    atomic_csv_dump(frame, output / "per_image_val.csv", args.overwrite)
    atomic_csv_dump(per_track, output / "per_track_val.csv", args.overwrite)
    atomic_json_dump(metrics, output / "metrics_val.json", args.overwrite)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
