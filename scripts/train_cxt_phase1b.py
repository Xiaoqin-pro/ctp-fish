"""CXT-Fish Phase 1B contrast-only pilot; internal test and outer folds stay locked."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.f4k_dataset import F4KDataset
from datasets.phase1b_sampler import StructuredTrackBatchSampler
from losses.supervised_contrastive import supervised_contrastive_loss
from models.resnet_contrastive import ResNet18Contrastive
from tools.reproducibility import seed_everything


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as handle:
        temp = Path(handle.name)
    torch.save(payload, temp); os.replace(temp, path)


def _worker_init(worker_id: int, seed: int) -> None:
    random.seed(seed + worker_id); np.random.seed(seed + worker_id)


def transforms_for(image_size: int):
    normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    train = transforms.Compose([transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip(), transforms.ColorJitter(.1, .1, .1, .05), transforms.ToTensor(), normalize])
    evaluate = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(image_size), transforms.ToTensor(), normalize])
    return train, evaluate


def _group_tensor(groups: list[str], device: torch.device) -> torch.Tensor:
    mapping = {group: index for index, group in enumerate(sorted(set(groups)))}
    return torch.tensor([mapping[group] for group in groups], dtype=torch.long, device=device)


def variant_spec(variant: str, configured_weight: float) -> dict:
    """The only variant-dependent behavior in the Phase 1B pilot."""
    if variant not in {"c0", "c1", "c2"}:
        raise ValueError(f"Unknown Phase 1B variant: {variant}")
    return {
        "classification_loss": "ce",
        "projection_head": "512-256-128",
        "positive_mode": {"c0": None, "c1": "standard", "c2": "cross_track"}[variant],
        "contrastive_weight": 0.0 if variant == "c0" else float(configured_weight),
    }


def run_train_epoch(model, loader, optimizer, scaler, device, amp, variant, temperature, weight):
    model.train(); ce_loss = nn.CrossEntropyLoss(); totals = {"ce": 0.0, "contrast": 0.0, "loss": 0.0, "embedding_norm": 0.0, "eligible": 0.0, "positive": 0.0, "batches": 0}
    specification = variant_spec(variant, weight)
    for images, targets, _, group_ids in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits, embedding = model(images)
            ce = ce_loss(logits, targets)
            if specification["positive_mode"] is None:
                contrast = embedding.sum() * 0.0; stats = {"eligible_anchors": 0, "mean_positive_count": 0.0}
            else:
                contrast, stats = supervised_contrastive_loss(embedding, targets, _group_tensor(list(group_ids), device), temperature, specification["positive_mode"])
            loss = ce + specification["contrastive_weight"] * contrast
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite Phase 1B loss.")
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        totals["ce"] += float(ce.detach()); totals["contrast"] += float(contrast.detach()); totals["loss"] += float(loss.detach())
        totals["embedding_norm"] += float(embedding.detach().norm(dim=1).mean()); totals["eligible"] += stats["eligible_anchors"]; totals["positive"] += stats["mean_positive_count"]; totals["batches"] += 1
    count = totals.pop("batches")
    return {key: value / count for key, value in totals.items()} | {"batches": count}


@torch.no_grad()
def run_val_epoch(model, loader, device, amp):
    model.eval(); criterion = nn.CrossEntropyLoss(); loss_sum = correct = count = 0
    for images, targets, _, _ in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits, _ = model(images); loss = criterion(logits, targets)
        loss_sum += float(loss) * len(targets); correct += int(logits.argmax(1).eq(targets).sum()); count += len(targets)
    return loss_sum / count, correct / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_phase1b.yaml")
    parser.add_argument("--variant", choices=["c0", "c1", "c2"], required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", default="outputs/cxt_fish/phase1b_pilot")
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(args.seed if args.seed is not None else cfg["pilot_seed"]); seed_everything(seed)
    if args.dry_run:
        print(json.dumps({"variant": args.variant, "seed": seed, "p": cfg["p"], "q": cfg["q"], "k": cfg["k"], "internal_test_accessed": False, "outer_folds_accessed": False})); return
    split_path = Path(cfg["track_split_path"])
    if split_path.resolve() == Path(cfg["outer_folds_path"]).resolve(): raise ValueError("Outer folds are locked.")
    metadata, split = pd.read_csv(cfg["metadata_path"]), pd.read_csv(split_path)
    records = metadata.merge(split[["image_path", "split"]], on="image_path", validate="one_to_one")
    if set(records.split.unique()) - {"train", "val", "test"}: raise ValueError("Unexpected split labels.")
    train_records, val_records = records[records.split == "train"], records[records.split == "val"]
    if train_records.empty or val_records.empty: raise ValueError("Phase 1B requires train and val records.")
    class_ids = sorted(records.species_id.astype(str).unique()); train_tf, val_tf = transforms_for(int(cfg["image_size"]))
    train_set = F4KDataset(train_records, train_tf, class_ids=class_ids); val_set = F4KDataset(val_records, val_tf, class_ids=class_ids)
    batch_sampler = StructuredTrackBatchSampler(train_set.records, cfg["p"], cfg["q"], cfg["k"], seed, len(train_set) // (int(cfg["p"]) * int(cfg["q"]) * int(cfg["k"])))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = {"num_workers": 2, "pin_memory": device.type == "cuda", "worker_init_fn": partial(_worker_init, seed=seed)}
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler, **common)
    val_loader = DataLoader(val_set, batch_size=int(cfg["p"]) * int(cfg["q"]) * int(cfg["k"]), shuffle=False, **common)
    model = ResNet18Contrastive(len(class_ids)).to(device); optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])); scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"])); scaler = torch.amp.GradScaler(device.type, enabled=bool(cfg["amp"]) and device.type == "cuda")
    run_name = args.run_name or f"{args.variant}_seed{seed}"; output = Path(args.output_root) / run_name
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True); history = []; best = -1.0; patience = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        batch_sampler.set_epoch(epoch)
        train = run_train_epoch(model, train_loader, optimizer, scaler, device, bool(cfg["amp"]), args.variant, float(cfg["temperature"]), float(cfg["contrastive_weight"]))
        val_loss, val_accuracy = run_val_epoch(model, val_loader, device, bool(cfg["amp"])); scheduler.step()
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train.items()}, "val_loss": val_loss, "val_accuracy": val_accuracy, "lr": optimizer.param_groups[0]["lr"]}; history.append(row)
        payload = {"model": model.state_dict(), "class_ids": class_ids, "epoch": epoch, "variant": args.variant, "config": cfg, "seed": seed, "internal_test_accessed": False, "outer_folds_accessed": False}
        if val_accuracy > best:
            best = val_accuracy; patience = 0; _atomic_save(payload, output / "best.pt")
        else: patience += 1
        _atomic_save(payload | {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history, "best_val_accuracy": best}, output / "last.pt")
        pd.DataFrame(history).to_csv(output / "training_curve.csv", index=False); print(json.dumps(row))
        if patience >= int(cfg["early_stopping_patience"]): break
    (output / "run_metadata.json").write_text(json.dumps({"variant": args.variant, "seed": seed, "p": cfg["p"], "q": cfg["q"], "k": cfg["k"], "temperature": cfg["temperature"], "contrastive_weight": cfg["contrastive_weight"], "internal_test_accessed": False, "outer_folds_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
