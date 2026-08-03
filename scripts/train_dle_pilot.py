"""Train one frozen-protocol DLE-Fish inner-fold variant."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.dle_dataset import DLEDataset, DLEValDataset
from datasets.phase1a_samplers import Phase1ASampler
from models.dle_resnet import DLEResNet18, positive_class_evidence_concentration
from tools.reproducibility import seed_everything


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as handle: temporary = Path(handle.name)
    torch.save(payload, temporary); os.replace(temporary, path)


def train_epoch(model, loader, optimizer, scaler, device, amp, variant):
    model.train(); ce = nn.CrossEntropyLoss(); totals = {"global_ce": 0.0, "foreground_ce": 0.0, "masked_ce": 0.0, "localization": 0.0, "valid_mask_fraction": 0.0}; batches = 0
    for original, foreground, mask, target, *_ in loader:
        original, foreground, mask, target = original.to(device), foreground.to(device), mask.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            original_out = model(original, mask if variant != "Q0" else None)
            foreground_out = model(foreground)
            global_ce = ce(original_out["global_logits"], target); foreground_ce = ce(foreground_out["global_logits"], target)
            masked_ce = original_out["global_logits"].sum() * 0.0; localization = original_out["global_logits"].sum() * 0.0; valid = torch.zeros(target.shape[0], dtype=torch.bool, device=device)
            if variant != "Q0": masked_ce = ce(original_out["masked_logits"], target)
            if variant in {"Q2a", "Q2b"}: localization, valid = positive_class_evidence_concentration(original_out, target)
            evidence_weight = 0.05 if variant == "Q2a" else 0.10 if variant == "Q2b" else 0.0
            loss = global_ce + foreground_ce + (masked_ce if variant != "Q0" else 0.0) + evidence_weight * localization
        if not torch.isfinite(loss): raise FloatingPointError("non-finite DLE loss")
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        totals["global_ce"] += float(global_ce.detach()); totals["foreground_ce"] += float(foreground_ce.detach()); totals["masked_ce"] += float(masked_ce.detach()); totals["localization"] += float(localization.detach()); totals["valid_mask_fraction"] += float(valid.float().mean()); batches += 1
    return {key: value / batches for key, value in totals.items()}


@torch.no_grad()
def validate(model, loader, device, amp):
    model.eval(); predictions = []; truth = []
    for image, _mask, target, *_ in loader:
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(image.to(device))["global_logits"]
        predictions.extend(logits.argmax(1).cpu().tolist()); truth.extend(target.tolist())
    return float(np.mean(np.asarray(predictions) == np.asarray(truth)))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/cxt_fish_phase2_dle.yaml"); parser.add_argument("--variant", choices=["Q0", "Q1", "Q2a", "Q2b"], required=True); parser.add_argument("--fold", type=int, choices=[0, 1], required=True); parser.add_argument("--seed", type=int); parser.add_argument("--output-root", default="outputs/cxt_fish/phase2_dle"); args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); seed = int(args.seed if args.seed is not None else cfg["pilot_seed"]); seed_everything(seed)
    protocol = json.loads(Path(cfg["class_ids_path"]).read_text(encoding="utf-8")); class_ids = [str(v) for v in protocol["class_ids"]]
    inner = json.loads(Path(cfg["inner_folds_path"]).read_text(encoding="utf-8")); heldout = set(inner["folds"][args.fold]["heldout_group_ids"])
    metadata = pd.read_csv(cfg["metadata_path"]); split = pd.read_csv(cfg["track_split_path"]); train_all = metadata.merge(split.loc[split.split == "train", ["image_path", "split"]], on="image_path", validate="one_to_one")
    train = train_all[~train_all.group_id.astype(str).isin(heldout)].reset_index(drop=True); val = train_all[train_all.group_id.astype(str).isin(heldout)].reset_index(drop=True)
    train_set = DLEDataset(train, __import__("datasets.dle_dataset", fromlist=["DLETrainTransform"]).DLETrainTransform(int(cfg["image_size"])), class_ids, blur_kernel=int(cfg["foreground_blur_kernel"]), blur_sigma=float(cfg["foreground_blur_sigma"]), feather_radius=int(cfg["mask_feather_radius"]))
    val_set = DLEValDataset(val, class_ids, int(cfg["image_size"]))
    sampler = Phase1ASampler(train, str(cfg["original_sampler"]), seed); loader = DataLoader(train_set, batch_size=int(cfg["batch_size"]), sampler=sampler, num_workers=2); val_loader = DataLoader(val_set, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = DLEResNet18(len(class_ids), pretrained=True).to(device); optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])); scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"])); scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda"); output = Path(args.output_root) / f"fold{args.fold}_{args.variant}_seed{seed}"
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True); history = []; best = -1.0; patience = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        sampler.set_epoch(epoch); train_metrics = train_epoch(model, loader, optimizer, scaler, device, bool(cfg.get("amp", True)), args.variant); val_accuracy = validate(model, val_loader, device, bool(cfg.get("amp", True))); scheduler.step(); row = {"epoch": epoch, **train_metrics, "val_accuracy": val_accuracy}; history.append(row)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "class_ids": class_ids, "variant": args.variant, "fold": args.fold, "seed": seed, "history": history, "config": cfg, "inner_folds": inner, "internal_test_accessed": False, "outer_folds_accessed": False}
        if val_accuracy > best: best, patience = val_accuracy, 0; payload["best_val_accuracy"] = best; atomic_save(payload, output / "best.pt")
        else: patience += 1
        atomic_save(payload, output / "last.pt"); pd.DataFrame(history).to_csv(output / "training_curve.csv", index=False); print(json.dumps(row))
        if patience >= int(cfg["early_stopping_patience"]): break


if __name__ == "__main__": main()
