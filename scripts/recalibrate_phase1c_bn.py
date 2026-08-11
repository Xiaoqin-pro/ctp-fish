"""Original-domain BatchNorm-only recalibration for frozen Phase 1C F3 checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.f4k_dataset import F4KDataset
from models.resnet_context import ResNet18Context


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_save(value: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent, suffix=".tmp") as handle: temporary = Path(handle.name)
    torch.save(value, temporary); os.replace(temporary, destination)


def deterministic_transform(size: int):
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def recalibrate(model: nn.Module, loader: DataLoader, device: torch.device) -> int:
    """Update only BN running statistics using a cumulative original-RGB train pass."""
    model.eval()
    batchnorms = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not batchnorms: raise ValueError("checkpoint model has no BatchNorm layers")
    momenta = [module.momentum for module in batchnorms]
    for module in batchnorms:
        module.reset_running_stats(); module.momentum = None; module.train()
    batches = 0
    try:
        with torch.no_grad():
            for image, _, _, _ in loader:
                model(image.to(device, non_blocking=True)); batches += 1
    finally:
        for module, momentum in zip(batchnorms, momenta): module.momentum = momentum
    return batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_phase1c.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source, destination = Path(args.checkpoint), Path(args.output)
    if destination.exists(): raise FileExistsError(f"refusing to overwrite {destination}")
    state = torch.load(source, map_location="cpu", weights_only=False)
    if state.get("variant") != "f3": raise ValueError("R1 accepts frozen F3 checkpoints only")
    split_path = Path(cfg["track_split_path"])
    if split_path.resolve() == Path(cfg["outer_folds_path"]).resolve(): raise ValueError("outer folds are locked")
    metadata, split = pd.read_csv(cfg["metadata_path"]), pd.read_csv(split_path)
    records = metadata.merge(split.loc[split.split == "train", ["image_path", "split"]], on="image_path", validate="one_to_one")
    if records.empty or set(records.split) != {"train"}: raise ValueError("R1 must use original train records only")
    dataset = F4KDataset(records, deterministic_transform(int(cfg["image_size"])), class_ids=state["class_ids"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    model = ResNet18Context(len(state["class_ids"])); model.load_state_dict(state["model"]); model.to(device)
    batches = recalibrate(model, loader, device)
    payload = state | {"model": model.state_dict(), "bn_recalibration": {"method": "original_domain_cumulative_batchnorm", "source_checkpoint": str(source), "source_checkpoint_sha256": sha256(source), "train_records": len(records), "batches": batches, "weights_updated": False, "labels_used": False, "internal_test_accessed": False, "outer_folds_accessed": False}}
    atomic_save(payload, destination)
    destination.with_suffix(".json").write_text(json.dumps(payload["bn_recalibration"], indent=2), encoding="utf-8")
    print(json.dumps(payload["bn_recalibration"], indent=2))


if __name__ == "__main__": main()
