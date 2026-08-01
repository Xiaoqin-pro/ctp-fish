"""Train only F1/F2/F3 of the frozen CXT-Fish Phase 1C pilot."""
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
from datasets.phase1a_samplers import Phase1ASampler
from datasets.phase1c_dataset import PairedTrainTransform, Phase1CDataset
from losses.context_consistency import context_consistency_loss, variant_losses
from models.resnet_context import ResNet18Context
from tools.reproducibility import seed_everything


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as handle:
        temporary = Path(handle.name)
    torch.save(payload, temporary); os.replace(temporary, path)


def seed_worker(worker_id: int, seed: int) -> None:
    random.seed(seed + worker_id); np.random.seed(seed + worker_id)


def capture_rng() -> dict:
    payload = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng(payload: dict) -> None:
    random.setstate(payload["python"]); np.random.set_state(payload["numpy"]); torch.set_rng_state(payload["torch"])
    if "cuda" in payload and torch.cuda.is_available(): torch.cuda.set_rng_state_all(payload["cuda"])


def eval_transform(size: int):
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


@torch.no_grad()
def validate(model, loader, device, amp) -> tuple[float, float]:
    model.eval(); criterion = nn.CrossEntropyLoss(); total = correct = count = 0
    for images, targets, _, _ in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits, _ = model(images); loss = criterion(logits, targets)
        total += float(loss) * len(targets); correct += int(logits.argmax(1).eq(targets).sum()); count += len(targets)
    return total / count, correct / count


def train_epoch(model, loader, optimizer, scaler, device, amp, variant, cfg) -> dict[str, float]:
    model.train(); criterion = nn.CrossEntropyLoss(); enabled = variant_losses(variant)
    sums = {"original_ce": 0.0, "foreground_ce": 0.0, "context": 0.0, "loss": 0.0, "batches": 0}
    for original, foreground, targets, _, _ in loader:
        original, foreground, targets = original.to(device, non_blocking=True), foreground.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # One concatenated forward pass fixes the BatchNorm handling across F1/F2/F3.
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits, feature = model(torch.cat([original, foreground], dim=0))
            batch = len(targets); original_logits, foreground_logits = logits[:batch], logits[batch:]
            original_feature, foreground_feature = feature[:batch], feature[batch:]
            original_ce = criterion(original_logits, targets)
            foreground_ce = criterion(foreground_logits, targets) if enabled["foreground_ce"] else original_ce.detach() * 0.0
            context = context_consistency_loss(original_feature, foreground_feature) if enabled["consistency"] else original_ce.detach() * 0.0
            loss = original_ce + float(cfg["foreground_ce_weight"]) * foreground_ce + float(cfg["context_consistency_weight"]) * context
        if not torch.isfinite(loss): raise FloatingPointError("non-finite Phase 1C loss")
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        for key, value in (("original_ce", original_ce), ("foreground_ce", foreground_ce), ("context", context), ("loss", loss)):
            sums[key] += float(value.detach())
        sums["batches"] += 1
    batches = sums.pop("batches")
    return {key: value / batches for key, value in sums.items()} | {"batches": batches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_phase1c.yaml")
    parser.add_argument("--variant", choices=["f1", "f2", "f3"], required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", default="outputs/cxt_fish/phase1c_pilot")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", help="Resume only from this variant's completed-epoch last.pt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else cfg["pilot_seed"]); seed_everything(seed)
    if args.dry_run:
        print(json.dumps({"variant": args.variant, "seed": seed, "sampler": "s1_track_uniform", "internal_test_accessed": False, "outer_folds_accessed": False})); return
    split_path = Path(cfg["track_split_path"])
    if split_path.resolve() == Path(cfg["outer_folds_path"]).resolve(): raise ValueError("outer folds are locked")
    metadata, split = pd.read_csv(cfg["metadata_path"]), pd.read_csv(split_path)
    records = metadata.merge(split[["image_path", "split"]], on="image_path", validate="one_to_one")
    if set(records.split.unique()) - {"train", "val", "test"}: raise ValueError("unexpected split labels")
    train_records, val_records = records.query("split == 'train'"), records.query("split == 'val'")
    if train_records.empty or val_records.empty: raise ValueError("train/validation partitions are required")
    class_ids = sorted(records.species_id.astype(str).unique())
    train_set = Phase1CDataset(train_records, PairedTrainTransform(int(cfg["image_size"])), class_ids, blur_kernel=int(cfg["foreground_blur_kernel"]), blur_sigma=float(cfg["foreground_blur_sigma"]), feather_radius=int(cfg["mask_feather_radius"]))
    val_set = F4KDataset(val_records, eval_transform(int(cfg["image_size"])), class_ids=class_ids)
    sampler = Phase1ASampler(train_set.records, "s1_track_uniform", seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = {"num_workers": 2, "pin_memory": device.type == "cuda", "worker_init_fn": partial(seed_worker, seed=seed)}
    train_loader = DataLoader(train_set, batch_size=int(cfg["batch_size"]), sampler=sampler, **common)
    val_loader = DataLoader(val_set, batch_size=int(cfg["batch_size"]), shuffle=False, **common)
    model = ResNet18Context(len(class_ids)).to(device); optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])); scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"])); scaler = torch.amp.GradScaler(device.type, enabled=bool(cfg["amp"]) and device.type == "cuda")
    name = args.run_name or f"{args.variant.upper()}_seed{seed}"; output = Path(args.output_root) / name
    history: list[dict] = []; best = -1.0; patience = 0; start_epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("variant") != args.variant or int(state.get("seed", -1)) != seed: raise ValueError("resume checkpoint variant or seed mismatch")
        if state.get("config") != cfg: raise ValueError("resume checkpoint config mismatch")
        if not all(key in state for key in ("optimizer", "scheduler", "scaler", "history", "rng_state")): raise ValueError("resume requires a complete-epoch last.pt")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        history, best, start_epoch = list(state["history"]), float(state["best_val_accuracy"]), int(state["epoch"]) + 1
        patience = 0 if not history else sum(1 for row in reversed(history) if float(row["val_accuracy"]) < best)
        restore_rng(state["rng_state"])
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    if start_epoch > int(cfg["epochs"]):
        print(json.dumps({"status": "already_complete", "epoch": start_epoch - 1})); return
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        sampler.set_epoch(epoch); train = train_epoch(model, train_loader, optimizer, scaler, device, bool(cfg["amp"]), args.variant, cfg); val_loss, val_accuracy = validate(model, val_loader, device, bool(cfg["amp"])); scheduler.step()
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train.items()}, "val_loss": val_loss, "val_accuracy": val_accuracy, "lr": optimizer.param_groups[0]["lr"]}; history.append(row)
        payload = {"model": model.state_dict(), "class_ids": class_ids, "epoch": epoch, "variant": args.variant, "config": cfg, "seed": seed, "sampler": "s1_track_uniform", "rng_state": capture_rng(), "internal_test_accessed": False, "outer_folds_accessed": False}
        if val_accuracy > best: best, patience = val_accuracy, 0; atomic_save(payload, output / "best.pt")
        else: patience += 1
        atomic_save(payload | {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history, "best_val_accuracy": best}, output / "last.pt")
        pd.DataFrame(history).to_csv(output / "training_curve.csv", index=False); print(json.dumps(row))
        if patience >= int(cfg["early_stopping_patience"]): break
    (output / "run_metadata.json").write_text(json.dumps({"variant": args.variant, "seed": seed, "sampler": "s1_track_uniform", "foreground_ce": variant_losses(args.variant)["foreground_ce"], "consistency": variant_losses(args.variant)["consistency"], "internal_test_accessed": False, "outer_folds_accessed": False}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
