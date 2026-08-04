"""Train one frozen route-C mask-guided outer-fold baseline cell."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.f4k_dataset import F4KDataset
from datasets.mask_contrastive_dataset import MaskContrastiveDataset, MaskContrastiveTransform
from datasets.phase1a_samplers import Phase1ASampler
from losses.mask_subject_nonprimary import subject_nonprimary_infonce
from models.resnet_contrastive import ResNet18Contrastive
from scripts.train_cxt_fish_clib_style_outer import (
    atomic_save, classifier_transform, eval_transform, pretrain_transform,
    sha256, state_dict_sha256, validate, worker_init,
)
from tools.reproducibility import seed_everything


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_mask_contrastive_outer_v1.yaml")
    parser.add_argument("--fold", choices=["1", "2", "3"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_path = ROOT / args.config; cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if args.seed not in cfg["seeds"]:
        raise ValueError("Only registered seeds are allowed")
    fold_root = ROOT / cfg["outer_manifest_root"] / f"fold_{args.fold}"
    train_path, dev_path, test_path = fold_root / "outer_train.csv", fold_root / "inner_dev.csv", fold_root / "outer_test.csv"
    if not all(path.is_file() for path in (train_path, dev_path, test_path)):
        raise FileNotFoundError("Frozen outer manifests are incomplete")
    if args.dry_run:
        print(json.dumps({"fold": args.fold, "seed": args.seed, "outer_test_loaded": False, "official_test_accessed": False, "views": cfg["views"]}, indent=2)); return
    seed_everything(args.seed)
    train_records, dev_records = pd.read_csv(train_path), pd.read_csv(dev_path)
    if set(train_records.group_id) & set(dev_records.group_id): raise ValueError("Outer train and inner-dev groups overlap")
    class_ids = sorted(set(train_records.species_id.astype(str)) | set(dev_records.species_id.astype(str)))
    if len(class_ids) != 16 or set(train_records.species_id.astype(str)) != set(class_ids) or set(dev_records.species_id.astype(str)) != set(class_ids):
        raise ValueError("Both outer-train and inner-dev must contain all 16 classes")
    assets = {"class_ids": cfg["class_ids_path"], "metadata": cfg["metadata_path"], "outer_folds": cfg["outer_folds_path"]}
    for name, relative in assets.items():
        if sha256(ROOT / relative) != cfg["frozen_asset_sha256"][name]: raise ValueError(f"Frozen asset hash mismatch: {name}")
    size = int(cfg["image_size"])
    train_set = MaskContrastiveDataset(train_records, MaskContrastiveTransform(size, tuple(cfg["pretrain_augmentation"]["crop_scale"])), class_ids, int(cfg["views"]["fill_value"]))
    classifier_set = F4KDataset(train_records, classifier_transform(size), class_ids=class_ids)
    dev_set = F4KDataset(dev_records, eval_transform(size), class_ids=class_ids)
    sampler = Phase1ASampler(train_set.records, "s1_track_uniform", args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = bool(cfg["training"]["amp"]) and device.type == "cuda"
    common = {"num_workers": int(cfg["training"]["num_workers"]), "pin_memory": device.type == "cuda", "worker_init_fn": partial(worker_init, seed=args.seed)}
    train_loader = DataLoader(train_set, batch_size=int(cfg["training"]["batch_size"]), sampler=sampler, **common)
    classifier_loader = DataLoader(classifier_set, batch_size=int(cfg["training"]["batch_size"]), sampler=sampler, **common)
    dev_loader = DataLoader(dev_set, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, **common)
    model = ResNet18Contrastive(len(class_ids)).to(device); initialization_sha = state_dict_sha256(model.state_dict())
    view_schema_sha = sha256(ROOT / "datasets" / "mask_contrastive_dataset.py")
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    pre_opt = AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    pre_sched = torch.optim.lr_scheduler.CosineAnnealingLR(pre_opt, T_max=int(cfg["training"]["pretrain_epochs"]))
    output = ROOT / cfg["output_root"] / f"fold_{args.fold}" / f"seed{args.seed}"
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True); history = []
    for epoch in range(1, int(cfg["training"]["pretrain_epochs"]) + 1):
        model.train(); sampler.set_epoch(epoch); total = batches = 0
        for original, foreground, non_primary, _, _, _ in train_loader:
            views = torch.stack((original, foreground, non_primary), dim=1).to(device, non_blocking=True)
            pre_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                _, embedding = model(views.flatten(0, 1)); loss = subject_nonprimary_infonce(embedding.view(views.shape[0], 3, -1), float(cfg["contrastive"]["temperature"]))
            if not torch.isfinite(loss): raise FloatingPointError("non-finite route-C contrastive loss")
            scaler.scale(loss).backward(); scaler.step(pre_opt); scaler.update(); total += float(loss.detach()); batches += 1
        pre_sched.step(); row = {"stage": "pretrain", "epoch": epoch, "contrastive_loss": total / max(1, batches), "batches": batches}; history.append(row); print(json.dumps(row), flush=True)
        atomic_save({"model": model.state_dict(), "optimizer": pre_opt.state_dict(), "scheduler": pre_sched.state_dict(), "scaler": scaler.state_dict(), "class_ids": class_ids, "fold": args.fold, "seed": args.seed, "epoch": epoch, "stage": "pretrain", "history": history, "config_sha256": sha256(cfg_path), "outer_train_sha256": sha256(train_path), "inner_dev_sha256": sha256(dev_path), "initialization_sha256": initialization_sha, "view_schema_sha256": view_schema_sha, "outer_test_accessed": False, "official_test_accessed": False}, output / "pretrain_last.pt")
    for parameter in model.encoder.parameters(): parameter.requires_grad_(False)
    model.encoder.eval(); cls_opt = AdamW(model.classifier.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    cls_sched = torch.optim.lr_scheduler.CosineAnnealingLR(cls_opt, T_max=int(cfg["training"]["classifier_epochs"])); criterion = nn.CrossEntropyLoss(); best = -1.0; patience = 0
    for epoch in range(1, int(cfg["training"]["classifier_epochs"]) + 1):
        model.classifier.train(); sampler.set_epoch(epoch + int(cfg["training"]["pretrain_epochs"])); total = batches = 0
        for images, targets, _, _ in classifier_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True); cls_opt.zero_grad(set_to_none=True)
            with torch.no_grad(): feature = model.encoder(images).flatten(1)
            loss = criterion(model.classifier(feature), targets)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite route-C classifier loss")
            loss.backward(); cls_opt.step(); total += float(loss.detach()); batches += 1
        dev_loss, dev_accuracy = validate(model, dev_loader, device, amp); cls_sched.step(); row = {"stage": "classifier", "epoch": epoch, "train_ce": total / max(1, batches), "inner_dev_loss": dev_loss, "inner_dev_accuracy": dev_accuracy, "batches": batches}; history.append(row); print(json.dumps(row), flush=True)
        state = {"model": model.state_dict(), "optimizer": cls_opt.state_dict(), "scheduler": cls_sched.state_dict(), "scaler": scaler.state_dict(), "class_ids": class_ids, "fold": args.fold, "seed": args.seed, "epoch": epoch, "stage": "classifier", "history": history, "best_inner_dev_accuracy": best, "config_sha256": sha256(cfg_path), "outer_train_sha256": sha256(train_path), "inner_dev_sha256": sha256(dev_path), "initialization_sha256": initialization_sha, "view_schema_sha256": view_schema_sha, "outer_test_accessed": False, "official_test_accessed": False}
        if dev_accuracy > best: best, patience = dev_accuracy, 0; state["best_inner_dev_accuracy"] = best; atomic_save(state, output / "best.pt")
        else: patience += 1
        atomic_save(state | {"patience": patience}, output / "last.pt"); pd.DataFrame(history).to_csv(output / "training_curve.csv", index=False)
        if patience >= 7: break
    (output / "run_metadata.json").write_text(json.dumps({"fold": args.fold, "seed": args.seed, "protocol_variant": "mask_guided_subject_nonprimary", "config_sha256": sha256(cfg_path), "outer_train_sha256": sha256(train_path), "inner_dev_sha256": sha256(dev_path), "initialization_sha256": initialization_sha, "view_schema_sha256": view_schema_sha, "outer_test_accessed": False, "official_test_accessed": False, "checkpoint_selection": "inner_dev_accuracy_with_patience_7"}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
