"""Train one frozen CLIB-style outer-fold baseline cell.

The driver has two fixed stages: three-view subject/context contrastive
pretraining, followed by a frozen-encoder classifier stage. It never loads
outer_test.csv; only the inner development split is read for checkpoint
selection during the classifier stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.clib_style_dataset import CLIBStyleDataset
from datasets.f4k_dataset import F4KDataset
from datasets.phase1a_samplers import Phase1ASampler
from losses.clib_style_contrastive import paired_subject_infonce
from models.resnet_contrastive import ResNet18Contrastive
from tools.reproducibility import seed_everything


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_dict_sha256(state_dict: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii")); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as handle:
        temporary = Path(handle.name)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def worker_init(worker_id: int, seed: int) -> None:
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def eval_transform(size: int):
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


def train_transform(size: int):
    return transforms.Compose([
        transforms.RandomResizedCrop(size), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.1, .1, .1, .05), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


@torch.no_grad()
def validate(model, loader, device, amp: bool):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = correct = count = 0
    for images, targets, _, _ in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits, _ = model(images)
            loss = criterion(logits, targets)
        loss_sum += float(loss) * len(targets)
        correct += int(logits.argmax(1).eq(targets).sum())
        count += len(targets)
    return loss_sum / count, correct / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_clib_style_outer_v1.yaml")
    parser.add_argument("--fold", choices=["1", "2", "3"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if args.seed not in cfg["seeds"]:
        raise ValueError("Only registered CLIB-style seeds are allowed.")
    fold_root = ROOT / cfg["outer_manifest_root"] / f"fold_{args.fold}"
    train_path, dev_path = fold_root / "outer_train.csv", fold_root / "inner_dev.csv"
    outer_test_path = fold_root / "outer_test.csv"
    if not train_path.is_file() or not dev_path.is_file() or not outer_test_path.is_file():
        raise FileNotFoundError("Build the frozen outer manifests before training.")
    if args.dry_run:
        print(json.dumps({
            "fold": args.fold, "seed": args.seed,
            "outer_train": str(train_path), "inner_dev": str(dev_path),
            "outer_test_loaded": False, "official_test_accessed": False,
            "views": cfg["views"], "temperature": cfg["contrastive"]["temperature"],
        }, indent=2))
        return

    seed_everything(args.seed)
    train_records = pd.read_csv(train_path)
    dev_records = pd.read_csv(dev_path)
    if set(train_records.group_id) & set(dev_records.group_id):
        raise ValueError("Outer train and inner-dev groups overlap.")
    class_ids = sorted(set(train_records.species_id.astype(str)) | set(dev_records.species_id.astype(str)))
    if len(class_ids) != 16 or set(train_records.species_id.astype(str)) != set(class_ids) or set(dev_records.species_id.astype(str)) != set(class_ids):
        raise ValueError("Both outer-train and inner-dev must contain all 16 frozen classes.")
    for relative, expected in cfg["frozen_asset_sha256"].items():
        asset = {"class_ids": cfg["class_ids_path"], "metadata": cfg["metadata_path"], "outer_folds": cfg["outer_folds_path"]}[relative]
        actual = sha256(ROOT / asset)
        if actual != expected:
            raise ValueError(f"Frozen asset hash mismatch for {relative}: {actual} != {expected}")

    image_size = int(cfg["image_size"])
    train_set = CLIBStyleDataset(train_records, train_transform(image_size), float(cfg["views"]["ratio"]), class_ids)
    dev_set = F4KDataset(dev_records, eval_transform(image_size), class_ids=class_ids)
    sampler = Phase1ASampler(train_set.records, "s1_track_uniform", args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = {"num_workers": int(cfg["training"]["num_workers"]), "pin_memory": device.type == "cuda", "worker_init_fn": partial(worker_init, seed=args.seed)}
    train_loader = DataLoader(train_set, batch_size=int(cfg["training"]["batch_size"]), sampler=sampler, **common)
    dev_loader = DataLoader(dev_set, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, **common)
    model = ResNet18Contrastive(len(class_ids)).to(device)
    initialization_sha = state_dict_sha256(model.state_dict())
    view_schema_sha = sha256(ROOT / "datasets" / "clib_style_views.py")
    amp = bool(cfg["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    pretrain_optimizer = AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    output = ROOT / (args.output_root or cfg["output_root"]) / f"fold_{args.fold}" / f"seed{args.seed}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, int(cfg["training"]["pretrain_epochs"]) + 1):
        model.train(); sampler.set_epoch(epoch); loss_sum = 0.0; batches = 0
        for original, subject, context, _, _, _ in train_loader:
            views = torch.stack((original, subject, context), dim=1).to(device, non_blocking=True)
            flat = views.flatten(0, 1)
            pretrain_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                _, embedding = model(flat)
                loss = paired_subject_infonce(embedding.view(views.shape[0], 3, -1), float(cfg["contrastive"]["temperature"]))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite CLIB-style contrastive loss")
            scaler.scale(loss).backward(); scaler.step(pretrain_optimizer); scaler.update()
            loss_sum += float(loss.detach()); batches += 1
        history.append({"stage": "pretrain", "epoch": epoch, "contrastive_loss": loss_sum / max(1, batches), "batches": batches})
        print(json.dumps(history[-1]), flush=True)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()
    classifier_optimizer = AdamW(model.classifier.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    criterion = nn.CrossEntropyLoss(); best = -1.0; patience = 0
    for epoch in range(1, int(cfg["training"]["classifier_epochs"]) + 1):
        model.classifier.train(); sampler.set_epoch(epoch + int(cfg["training"]["pretrain_epochs"]))
        loss_sum = 0.0; batches = 0
        for original, _, _, targets, _, _ in train_loader:
            original, targets = original.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            classifier_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                feature = model.encoder(original).flatten(1)
            logits = model.classifier(feature); loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite CLIB-style classifier loss")
            loss.backward(); classifier_optimizer.step(); loss_sum += float(loss.detach()); batches += 1
        dev_loss, dev_accuracy = validate(model, dev_loader, device, amp)
        row = {"stage": "classifier", "epoch": epoch, "train_ce": loss_sum / max(1, batches), "inner_dev_loss": dev_loss, "inner_dev_accuracy": dev_accuracy, "batches": batches}
        history.append(row); print(json.dumps(row), flush=True)
        state = {"model": model.state_dict(), "class_ids": class_ids, "fold": args.fold, "seed": args.seed, "epoch": epoch, "stage": "classifier", "history": history, "best_inner_dev_accuracy": best, "config_sha256": sha256(cfg_path), "outer_train_sha256": sha256(train_path), "inner_dev_sha256": sha256(dev_path), "initialization_sha256": initialization_sha, "view_schema_sha256": view_schema_sha, "outer_test_accessed": False, "official_test_accessed": False}
        if dev_accuracy > best:
            best, patience = dev_accuracy, 0; state["best_inner_dev_accuracy"] = best; atomic_save(state, output / "best.pt")
        else:
            patience += 1
        atomic_save(state | {"patience": patience}, output / "last.pt")
        pd.DataFrame(history).to_csv(output / "training_curve.csv", index=False)
        if patience >= 7:
            break
    (output / "run_metadata.json").write_text(json.dumps({"fold": args.fold, "seed": args.seed, "config_sha256": sha256(cfg_path), "outer_train_sha256": sha256(train_path), "inner_dev_sha256": sha256(dev_path), "initialization_sha256": initialization_sha, "view_schema_sha256": view_schema_sha, "outer_test_accessed": False, "official_test_accessed": False, "checkpoint_selection": "inner_dev_accuracy_with_patience_7"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
