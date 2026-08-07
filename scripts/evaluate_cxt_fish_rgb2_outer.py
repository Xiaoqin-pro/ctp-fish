"""Evaluate the completed F0-2RGB control on the frozen primary outer manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.resnet_context import ResNet18Context  # noqa: E402
from scripts.evaluate_cxt_fish_outer import OuterContextDataset, metric_block, sha256, tiers  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_outer_validation_v1.yaml")
    parser.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--seed", type=int, choices=[3407, 2026, 17], required=True)
    parser.add_argument("--output-root", default="outputs/cxt_fish/rgb2_control_outer_evaluation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    import yaml
    cfg_path = ROOT / args.config; cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    fold_root = ROOT / cfg["outer_manifest_output_root"] / f"fold_{args.fold}"
    test_path = fold_root / "outer_test.csv"; manifest_path = fold_root / "outer_context_swap.csv"
    checkpoint = ROOT / "outputs/cxt_fish/rgb2_control_outer" / f"fold_{args.fold}" / f"F0_2RGB_seed{args.seed}" / "best.pt"
    output = ROOT / args.output_root / f"fold_{args.fold}" / f"F0_2RGB_seed{args.seed}"
    gate = ROOT / "reports/cxt_fish_rgb2_training_complete.json"
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "outer_test_loaded": False, "official_test_accessed": False, "training_gate_required": str(gate)}, indent=2)); return
    if not gate.is_file():
        raise RuntimeError("Refusing RGB2 outer evaluation until the 9-cell training gate exists.")
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    if gate_payload.get("cells") != 9 or gate_payload.get("official_test_accessed") is not False:
        raise RuntimeError("Invalid RGB2 training completeness gate")
    for path in (test_path, manifest_path, checkpoint):
        if not path.is_file(): raise FileNotFoundError(path)
    if (output / "metrics.json").is_file():
        print(json.dumps({"status": "already_complete", "output": str(output)})); return
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Refusing to overwrite {output}")
    records, manifest = pd.read_csv(test_path), pd.read_csv(manifest_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_ids = [str(value) for value in state["class_ids"]]
    model = ResNet18Context(len(class_ids)); model.load_state_dict(state["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    train_records = pd.read_csv(fold_root / "outer_train.csv"); tier_map = tiers(train_records, class_ids)
    dataset = OuterContextDataset(records, manifest, cfg, class_ids)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    values = {name: [] for name in ("original", "foreground", "same_swap", "cross_swap")}; targets, donors, paths, groups = [], [], [], []
    with torch.no_grad():
        for original, foreground, same, cross, target, donor, path, group in loader:
            for name, images in zip(values, (original, foreground, same, cross)):
                logits, _ = model(images.to(device, non_blocking=True)); values[name].extend(logits.argmax(1).cpu().tolist())
            targets.extend(target.tolist()); donors.extend(donor.tolist()); paths.extend(path); groups.extend(group)
    frame = pd.DataFrame({"image_path": paths, "group_id": groups, "target": targets, "cross_donor_target": donors, **values})
    metrics = {name: metric_block(frame, name, class_ids, tier_map) for name in values}
    original_correct = frame.original.eq(frame.target); donor_attraction = frame.cross_swap.eq(frame.cross_donor_target)
    metrics["context"] = {"prediction_agreement": float(frame.original.eq(frame.cross_swap).mean()), "dar": float(donor_attraction.mean()), "dar_flip": None if not bool(original_correct.any()) else float(donor_attraction[original_correct].mean()), "delta_foreground_macro_f1": metrics["original"]["macro_f1"] - metrics["foreground"]["macro_f1"], "delta_same_swap_macro_f1": metrics["original"]["macro_f1"] - metrics["same_swap"]["macro_f1"], "delta_cross_swap_macro_f1": metrics["original"]["macro_f1"] - metrics["cross_swap"]["macro_f1"]}
    payload = {"status": "complete", "fold": args.fold, "method": "F0_2RGB", "seed": args.seed, "partition": "outer_test", "outer_test_accessed": True, "official_test_accessed": False, "control_type": "post_hoc_two_rgb_supervision_matched", "checkpoint_sha256": sha256(checkpoint), "config_sha256": sha256(cfg_path), "outer_test_sha256": sha256(test_path), "context_swap_sha256": sha256(manifest_path), "metrics": metrics, "tier_map": tier_map, "n_images": len(frame), "n_groups": int(frame.group_id.nunique())}
    output.mkdir(parents=True, exist_ok=True); frame.to_csv(output / "per_image.csv", index=False); (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "fold": args.fold, "seed": args.seed, "original_macro_f1": metrics["original"]["macro_f1"], "cross_swap_macro_f1": metrics["cross_swap"]["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
