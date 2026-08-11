"""Frozen-encoder linear-probe audit for the corrected 16-class CT-DFS pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.f4k_dataset import F4KDataset
from models.resnet_context import ResNet18Context
from tools.reproducibility import seed_everything


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eval_transform(size: int):
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


@torch.no_grad()
def extract(model: ResNet18Context, loader: DataLoader, device: torch.device):
    model.eval(); features = []; labels = []
    for images, targets, *_ in loader:
        _, pooled = model(images.to(device))
        features.append(pooled.float().cpu()); labels.append(targets.long().cpu())
    return torch.cat(features), torch.cat(labels)


def train_probe(train_x, train_y, val_x, val_y, seed: int, epochs: int, device: torch.device):
    seed_everything(seed)
    probe = nn.Linear(train_x.shape[1], int(train_y.max()) + 1).to(device)
    optimizer = AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        probe.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(features.to(device)), labels.to(device))
            if not torch.isfinite(loss): raise FloatingPointError("non-finite linear-probe loss")
            loss.backward(); optimizer.step()
    probe.eval()
    with torch.no_grad():
        predictions = probe(val_x.to(device)).argmax(1).cpu().numpy()
    truth = val_y.numpy()
    return {"macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(truth, predictions))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_ctdfs.yaml")
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH; repeat for each frozen encoder")
    parser.add_argument("--output", default="outputs/cxt_fish/ctdfs_fixed16/linear_probe_metrics.json")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    protocol = json.loads(Path(cfg["class_ids_path"]).read_text(encoding="utf-8")); class_ids = [str(v) for v in protocol["class_ids"]]
    if len(class_ids) != 16: raise ValueError("linear probe requires the frozen 16-class protocol")
    metadata = pd.read_csv(cfg["metadata_path"]); split = pd.read_csv(cfg["track_split_path"])
    train = metadata.merge(split.loc[split.split == "train", ["image_path", "split"]], on="image_path", validate="one_to_one")
    val = metadata.merge(split.loc[split.split == "val", ["image_path", "split"]], on="image_path", validate="one_to_one")
    if set(train.species_id.astype(str)) != set(class_ids) or set(val.species_id.astype(str)) != set(class_ids): raise ValueError("train/val class set mismatch")
    transform = eval_transform(int(cfg["image_size"]))
    train_loader = DataLoader(F4KDataset(train, transform, class_ids=class_ids), batch_size=256, shuffle=False, num_workers=2)
    val_loader = DataLoader(F4KDataset(val, transform, class_ids=class_ids), batch_size=256, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_features = train_labels = val_features = val_labels = None
    results = []
    for spec in args.checkpoint:
        name, raw_path = spec.split("=", 1); checkpoint_path = Path(raw_path); checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if [str(v) for v in checkpoint.get("class_ids", [])] != class_ids: raise ValueError(f"{name} class protocol mismatch")
        model = ResNet18Context(len(class_ids)); model.load_state_dict(checkpoint["model"]); model.to(device)
        for parameter in model.encoder.parameters(): parameter.requires_grad_(False)
        if train_features is None: train_features, train_labels = extract(model, train_loader, device); val_features, val_labels = extract(model, val_loader, device)
        else:
            train_features, train_labels = extract(model, train_loader, device); val_features, val_labels = extract(model, val_loader, device)
        metrics = train_probe(train_features, train_labels, val_features, val_labels, args.seed, args.epochs, device)
        results.append({"name": name, "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path), "metrics": metrics, "feature_dim": int(train_features.shape[1]), "train_samples": len(train_labels), "val_samples": len(val_labels)})
    payload = {"schema_version": "ctdfs_fixed16_linear_probe_v1", "class_ids": class_ids, "seed": args.seed, "epochs": args.epochs, "official_test_accessed": False, "outer_folds_accessed": False, "results": results}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
