"""Train one frozen CXT-Fish outer-fold cell (F0 or F1).

This driver never loads outer_test.csv. It selects checkpoints only on the
fixed inner_dev split and records the complete provenance needed for the final
outer evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
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
from models.resnet_context import ResNet18Context
from tools.reproducibility import seed_everything


ROOT = Path(__file__).resolve().parents[1]


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as handle:
        temporary = Path(handle.name)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker_init(worker_id: int, seed: int) -> None:
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def capture_rng() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].detach().cpu().to(dtype=torch.uint8).contiguous())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.detach().cpu().to(dtype=torch.uint8).contiguous() for item in state["cuda"]])


def eval_transform(size: int):
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


@torch.no_grad()
def validate(model, loader, device, amp: bool) -> tuple[float, float]:
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


def train_epoch(model, loader, optimizer, scaler, device, amp: bool, method: str, cfg: dict) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    sums = {"original_ce": 0.0, "foreground_ce": 0.0, "loss": 0.0, "batches": 0}
    for batch in loader:
        if method == "F1":
            original, foreground, targets, _, _ = batch
            original = original.to(device, non_blocking=True)
            foreground = foreground.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits, _ = model(torch.cat([original, foreground], dim=0))
                n = len(targets)
                original_ce = criterion(logits[:n], targets)
                foreground_ce = criterion(logits[n:], targets)
                loss = original_ce + float(cfg["foreground_ce_weight"]) * foreground_ce
        else:
            images, targets, _, _ = batch
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits, _ = model(images)
                original_ce = criterion(logits, targets)
                foreground_ce = original_ce.detach() * 0.0
                loss = original_ce
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite outer training loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        sums["original_ce"] += float(original_ce.detach())
        sums["foreground_ce"] += float(foreground_ce.detach())
        sums["loss"] += float(loss.detach())
        sums["batches"] += 1
    batches = sums.pop("batches")
    return {key: value / batches for key, value in sums.items()} | {"batches": batches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_outer_validation_v1.yaml")
    parser.add_argument("--fold", choices=["1", "2", "3"], required=True)
    parser.add_argument("--method", choices=["F0", "F1"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume")
    parser.add_argument("--output-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.seed not in [3407, 2026, 17]:
        raise ValueError("Only the three registered seeds are allowed.")
    fold_root = ROOT / cfg["outer_manifest_output_root"] / f"fold_{args.fold}"
    train_path, dev_path = fold_root / "outer_train.csv", fold_root / "inner_dev.csv"
    if not train_path.is_file() or not dev_path.is_file():
        raise FileNotFoundError("Run build_cxt_fish_outer_manifests.py before training.")
    outer_test_path = fold_root / "outer_test.csv"
    if not outer_test_path.is_file():
        raise FileNotFoundError(outer_test_path)
    if args.dry_run:
        print(json.dumps({
            "fold": args.fold, "method": args.method, "seed": args.seed,
            "outer_train": str(train_path), "inner_dev": str(dev_path),
            "outer_test_loaded": False, "official_test_accessed": False,
        }, indent=2))
        return

    seed_everything(args.seed)
    train_records = pd.read_csv(train_path)
    dev_records = pd.read_csv(dev_path)
    if set(train_records.split.unique()) != {"train"} or set(dev_records.split.unique()) != {"val"}:
        raise ValueError("Outer manifests must label train and inner-dev records only.")
    if set(train_records.group_id) & set(dev_records.group_id):
        raise ValueError("Outer train and inner-dev groups overlap.")
    class_ids = sorted(set(train_records.species_id.astype(str)) | set(dev_records.species_id.astype(str)))
    if len(class_ids) != 16:
        raise ValueError("Frozen outer protocol requires all 16 classes.")
    image_size = int(cfg["image_size"])
    if args.method == "F1":
        train_set = Phase1CDataset(
            train_records, PairedTrainTransform(image_size), class_ids,
            blur_kernel=21, blur_sigma=5.0, feather_radius=3,
        )
    else:
        normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.1, .1, .1, .05), transforms.ToTensor(), normalize,
        ])
        train_set = F4KDataset(train_records, train_tf, class_ids=class_ids)
    dev_set = F4KDataset(dev_records, eval_transform(image_size), class_ids=class_ids)
    sampler = Phase1ASampler(train_set.records, "s1_track_uniform", args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = {"num_workers": 2, "pin_memory": device.type == "cuda", "worker_init_fn": partial(worker_init, seed=args.seed)}
    train_loader = DataLoader(train_set, batch_size=64, sampler=sampler, **common)
    dev_loader = DataLoader(dev_set, batch_size=64, shuffle=False, **common)
    model = ResNet18Context(len(class_ids)).to(device)
    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))
    scaler = torch.amp.GradScaler(device.type, enabled=bool(cfg["amp"]) and device.type == "cuda")
    output = ROOT / (args.output_root or "outputs/cxt_fish/final_outer") / f"fold_{args.fold}" / f"{args.method}_seed{args.seed}"
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        for key, expected in (("fold", args.fold), ("method", args.method), ("seed", args.seed)):
            if state.get(key) != expected:
                raise ValueError(f"resume {key} mismatch")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        history, best, start = state["history"], float(state["best_inner_dev_accuracy"]), int(state["epoch"]) + 1
        restore_rng(state["rng_state"])
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite {output}")
        history, best, start = [], -1.0, 1
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, int(cfg["epochs"]) + 1):
        sampler.set_epoch(epoch)
        train = train_epoch(model, train_loader, optimizer, scaler, device, bool(cfg["amp"]), args.method, {"foreground_ce_weight": 1.0})
        dev_loss, dev_accuracy = validate(model, dev_loader, device, bool(cfg["amp"]))
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train.items()}, "inner_dev_loss": dev_loss, "inner_dev_accuracy": dev_accuracy}
        history.append(row)
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "history": history, "epoch": epoch, "best_inner_dev_accuracy": best, "fold": args.fold, "method": args.method, "seed": args.seed,
            "config_sha256": sha256(ROOT / args.config), "outer_test_accessed": False, "internal_test_accessed": False, "official_test_accessed": False,
            "rng_state": capture_rng(),
        }
        if dev_accuracy > best:
            best = dev_accuracy
            payload["best_inner_dev_accuracy"] = best
            atomic_save({"model": model.state_dict(), "class_ids": class_ids, "epoch": epoch, "fold": args.fold, "method": args.method, "seed": args.seed, "best_inner_dev_accuracy": best, "config_sha256": payload["config_sha256"], "outer_test_accessed": False}, output / "best.pt")
        atomic_save(payload, output / "last.pt")
        pd.DataFrame(history).to_csv(output / "training_curve.csv", index=False)
        print(json.dumps(row))
    (output / "run_metadata.json").write_text(json.dumps({"fold": args.fold, "method": args.method, "seed": args.seed, "outer_test_accessed": False, "internal_test_accessed": False, "official_test_accessed": False, "config_sha256": sha256(ROOT / args.config)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
