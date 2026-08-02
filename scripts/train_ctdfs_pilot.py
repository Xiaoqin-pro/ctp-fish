"""Train one fixed CT-DFS pilot variant on train and frozen validation only."""
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
from datasets.ctdfs_dataset import CTDFSDataset
from datasets.ctdfs_sampler import CTDFSDualStreamSampler
from datasets.f4k_dataset import F4KDataset
from datasets.phase1c_dataset import PairedTrainTransform
from models.resnet_context import ResNet18Context
from tools.reproducibility import seed_everything


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as handle:
        temporary = Path(handle.name)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_worker(worker_id: int, seed: int) -> None:
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def capture_rng() -> dict:
    result = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def eval_transform(size: int):
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


@torch.no_grad()
def validate(model, loader, device, amp):
    model.eval(); criterion = nn.CrossEntropyLoss(); total = correct = count = 0
    for images, targets, _, _ in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits, _ = model(images); loss = criterion(logits, targets)
        total += float(loss) * len(targets); correct += int(logits.argmax(1).eq(targets).sum()); count += len(targets)
    return total / count, correct / count


def train_epoch(model, loader, optimizer, scaler, device, amp):
    model.train(); criterion = nn.CrossEntropyLoss(); sums = {"original_ce": 0.0, "foreground_ce": 0.0, "loss": 0.0}; batches = 0
    for original, foreground, original_target, foreground_target, *_ in loader:
        original, foreground = original.to(device), foreground.to(device)
        original_target, foreground_target = original_target.to(device), foreground_target.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits, _ = model(torch.cat([original, foreground], dim=0))
            batch = len(original_target)
            original_logits, foreground_logits = logits[:batch], logits[batch:]
            original_ce = criterion(original_logits, original_target)
            foreground_ce = criterion(foreground_logits, foreground_target)
            loss = original_ce + foreground_ce
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite CT-DFS loss")
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        sums["original_ce"] += float(original_ce.detach()); sums["foreground_ce"] += float(foreground_ce.detach()); sums["loss"] += float(loss.detach()); batches += 1
    return {key: value / batches for key, value in sums.items()} | {"batches": batches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_ctdfs.yaml")
    parser.add_argument("--variant", choices=["P1", "P2"], required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", default="outputs/cxt_fish/ctdfs_pilot")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", help="resume from a complete epoch checkpoint")
    args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else cfg["pilot_seed"]); seed_everything(seed)
    split_path = Path(cfg["track_split_path"])
    if split_path.resolve() == Path(cfg["outer_folds_path"]).resolve(): raise ValueError("outer folds are locked")
    metadata, split = pd.read_csv(cfg["metadata_path"]), pd.read_csv(split_path)
    train = metadata.merge(split.loc[split.split == "train", ["image_path", "split"]], on="image_path", validate="one_to_one")
    val = metadata.merge(split.loc[split.split == "val", ["image_path", "split"]], on="image_path", validate="one_to_one")
    if train.empty or val.empty or set(train.split) != {"train"} or set(val.split) != {"val"}: raise ValueError("CT-DFS requires train and val only")
    class_protocol = json.loads(Path(cfg["class_ids_path"]).read_text(encoding="utf-8"))
    class_ids = [str(value) for value in class_protocol["class_ids"]]
    if len(class_ids) != 16 or int(class_protocol["class_count"]) != 16:
        raise ValueError("F4K-16T class protocol must contain exactly 16 classes")
    if set(train.species_id.astype(str)) != set(class_ids) or set(val.species_id.astype(str)) != set(class_ids):
        raise ValueError("train/val species set does not match frozen F4K-16T class protocol")
    train_set = CTDFSDataset(train, PairedTrainTransform(int(cfg["image_size"])), class_ids, blur_kernel=int(cfg["foreground_blur_kernel"]), blur_sigma=float(cfg["foreground_blur_sigma"]), feather_radius=int(cfg["mask_feather_radius"]))
    val_set = F4KDataset(val, eval_transform(int(cfg["image_size"])), class_ids=class_ids)
    foreground_mode = str(cfg["p1_foreground_sampler"] if args.variant == "P1" else cfg["p2_foreground_sampler"])
    sampler = CTDFSDualStreamSampler(train, str(cfg["original_sampler"]), foreground_mode, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = {"num_workers": 2, "pin_memory": device.type == "cuda", "worker_init_fn": partial(seed_worker, seed=seed)}
    train_loader = DataLoader(train_set, batch_size=int(cfg["batch_size"]), sampler=sampler, **common)
    val_loader = DataLoader(val_set, batch_size=int(cfg["batch_size"]), shuffle=False, **common)
    model = ResNet18Context(len(class_ids)).to(device); optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"])); scaler = torch.amp.GradScaler(device.type, enabled=bool(cfg["amp"]) and device.type == "cuda")
    name = args.run_name or f"{args.variant}_seed{seed}"; output = Path(args.output_root) / name
    history = []; best = -1.0; patience = 0; start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file(): raise FileNotFoundError(resume_path)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("variant") != args.variant.lower(): raise ValueError("resume checkpoint variant mismatch")
        if [str(v) for v in checkpoint.get("class_ids", [])] != class_ids: raise ValueError("resume checkpoint class protocol mismatch")
        if checkpoint.get("epoch", 0) < 1: raise ValueError("resume checkpoint must be a completed epoch")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"]); scaler.load_state_dict(checkpoint["scaler"]); restore_rng(checkpoint["rng_state"])
        history = list(checkpoint.get("history", [])); best = float(checkpoint.get("best_val_accuracy", max((float(row["val_accuracy"]) for row in history), default=-1.0)))
        last_best = max((idx for idx, row in enumerate(history) if float(row["val_accuracy"]) == best), default=-1)
        patience = len(history) - last_best - 1
        start_epoch = int(checkpoint["epoch"]) + 1
        output = resume_path.parent
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        sampler.set_epoch(epoch); train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, bool(cfg["amp"])); val_loss, val_accuracy = validate(model, val_loader, device, bool(cfg["amp"])); scheduler.step()
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}, "val_loss": val_loss, "val_accuracy": val_accuracy, "lr": optimizer.param_groups[0]["lr"]}; history.append(row)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "class_ids": class_ids, "class_protocol": class_protocol, "epoch": epoch, "variant": args.variant.lower(), "config": cfg, "seed": seed, "original_sampler": cfg["original_sampler"], "foreground_sampler": foreground_mode, "internal_test_accessed": False, "outer_folds_accessed": False, "history": history, "best_val_accuracy": best, "patience": patience, "rng_state": capture_rng()}
        if val_accuracy > best: best, patience = val_accuracy, 0; payload["best_val_accuracy"] = best; atomic_save(payload, output / "best.pt")
        else: patience += 1
        atomic_save(payload, output / "last.pt"); pd.DataFrame(history).to_csv(output / "training_curve.csv", index=False); print(json.dumps(row))
        if patience >= int(cfg["early_stopping_patience"]): break
    (output / "run_metadata.json").write_text(json.dumps({"variant": args.variant, "seed": seed, "original_sampler": cfg["original_sampler"], "foreground_sampler": foreground_mode, "internal_test_accessed": False, "outer_folds_accessed": False}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
