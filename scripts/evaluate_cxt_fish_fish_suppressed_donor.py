"""Inference-only fish-suppressed donor construct-validity sensitivity."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.context_views import context_swap_view  # noqa: E402
from metrics.classification import classification_metrics  # noqa: E402
from metrics.track_metrics import track_balanced_accuracy  # noqa: E402
from models.resnet_context import ResNet18Context  # noqa: E402
from scripts.evaluate_cxt_fish_outer import sha256, tiers  # noqa: E402


def eval_transform(size: int):
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def suppress_donor(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    binary = (np.asarray(mask.convert("L"), dtype=np.uint8) > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    result = cv2.inpaint(rgb, dilated, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(result)


class Dataset(Dataset):
    def __init__(self, records: pd.DataFrame, manifest: pd.DataFrame, class_ids: list[str], size: int):
        self.records = records.sort_values("image_path").reset_index(drop=True)
        self.transform = eval_transform(size)
        self.class_to_index = {str(v): i for i, v in enumerate(class_ids)}
        self.donors = {str(row.recipient_image_path): row._asdict() for row in manifest.itertuples(index=False)}
        missing = [p for p in self.records.image_path.astype(str) if p not in self.donors]
        if missing:
            raise ValueError(f"missing frozen primary cross-class donor rows: {len(missing)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        donor = self.donors[str(row.image_path)]
        recipient = Image.open(row.image_path).convert("RGB")
        recipient_mask = Image.open(row.mask_path).convert("L")
        donor_rgb = Image.open(donor["donor_image_path"]).convert("RGB")
        donor_mask = Image.open(donor["donor_mask_path"]).convert("L")
        suppressed = suppress_donor(donor_rgb, donor_mask)
        composite = context_swap_view(recipient, recipient_mask, suppressed, feather_radius=3)
        return self.transform(composite), self.class_to_index[str(row.species_id)], str(row.image_path), str(row.group_id), int(donor["donor_species_id"])


def metric_block(frame: pd.DataFrame, class_ids: list[str], tier_map: dict[str, str]) -> dict:
    metric = classification_metrics(frame.target.to_numpy(), frame.prediction.to_numpy(), list(range(len(class_ids))))
    per_class = dict(zip(map(str, class_ids), metric["per_class_f1"]))
    for tier in ("head", "mid", "tail"):
        metric[f"{tier}_f1"] = float(np.mean([per_class[k] for k, value in tier_map.items() if value == tier]))
    grouped = frame[["group_id", "target", "prediction"]].copy(); grouped["correct"] = grouped.target.eq(grouped.prediction)
    metric["group_balanced_accuracy"], _ = track_balanced_accuracy(grouped)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--method", choices=["F0", "F1"], required=True)
    parser.add_argument("--seed", type=int, choices=[3407, 2026, 17], required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    fold_root = ROOT / f"outputs/cxt_fish/final_outer_manifests/fold_{args.fold}"
    test_path = fold_root / "outer_test.csv"
    manifest_path = fold_root / "outer_context_swap.csv"
    checkpoint = ROOT / f"outputs/cxt_fish/final_outer/fold_{args.fold}/{args.method}_seed{args.seed}/best.pt"
    frozen = ROOT / f"outputs/cxt_fish/final_outer_evaluation/fold_{args.fold}/{args.method}_seed{args.seed}/per_image.csv"
    for path in (test_path, manifest_path, checkpoint, frozen):
        if not path.is_file():
            raise FileNotFoundError(path)
    records = pd.read_csv(test_path)
    manifest = pd.read_csv(manifest_path)
    manifest = manifest.loc[(manifest.swap_type == "cross_class") & manifest.supported.astype(str).str.lower().eq("true")].copy()
    required = ["donor_image_path", "donor_mask_path", "recipient_image_path"]
    coverage = {name: int(manifest[name].map(lambda p: Path(str(p)).is_file()).sum()) for name in required}
    if any(coverage[name] != len(manifest) for name in required):
        (ROOT / "reports/cxt_fish_fish_suppressed_donor_coverage.md").write_text(json.dumps({"coverage": coverage, "rows": len(manifest), "official_test_accessed": False}, indent=2), encoding="utf-8")
        raise RuntimeError(f"donor image/mask coverage is incomplete: {coverage}")
    if args.check_only:
        print(json.dumps({"status": "coverage_ok", "fold": args.fold, "rows": len(manifest), "coverage": coverage, "official_test_accessed": False}, indent=2)); return
    out = ROOT / f"outputs/cxt_fish/fish_suppressed_donor_evaluation/fold_{args.fold}/{args.method}_seed{args.seed}"
    if (out / "metrics.json").is_file():
        print(json.dumps({"status": "already_complete", "output": str(out)})); return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_ids = [str(v) for v in state["class_ids"]]
    model = ResNet18Context(len(class_ids)); model.load_state_dict(state["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    train = pd.read_csv(fold_root / "outer_train.csv"); tier_map = tiers(train, class_ids)
    dataset = Dataset(records, manifest, class_ids, 224)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    paths, groups, targets, donors, predictions = [], [], [], [], []
    with torch.no_grad():
        for images, target, path, group, donor in loader:
            logits, _ = model(images.to(device, non_blocking=True)); predictions.extend(logits.argmax(1).cpu().tolist()); targets.extend(target.tolist()); paths.extend(path); groups.extend(group); donors.extend(donor.tolist())
    frame = pd.DataFrame({"image_path": paths, "group_id": groups, "target": targets, "cross_donor_target": donors, "prediction": predictions})
    original = pd.read_csv(frozen)[["image_path", "original"]]
    frame = frame.merge(original, on="image_path", how="left", validate="one_to_one")
    if frame.original.isna().any(): raise ValueError("frozen original predictions missing")
    metrics = metric_block(frame, class_ids, tier_map)
    donor_attraction = frame.prediction.eq(frame.cross_donor_target); original_correct = frame.original.eq(frame.target)
    metrics["context"] = {"prediction_agreement": float(frame.original.eq(frame.prediction).mean()), "dar": float(donor_attraction.mean()), "dar_flip": float(donor_attraction[original_correct].mean()) if original_correct.any() else None, "cross_swap_macro_f1": metrics["macro_f1"]}
    payload = {"status": "complete", "analysis_role": "post_hoc_donor_subject_suppressed_construct_validity_sensitivity", "fold": args.fold, "method": args.method, "seed": args.seed, "outer_test_accessed": True, "official_test_accessed": False, "checkpoint_sha256": sha256(checkpoint), "outer_test_sha256": sha256(test_path), "donor_manifest_sha256": sha256(manifest_path), "frozen_original_predictions_sha256": sha256(frozen), "inpainting": {"method": "opencv_telea", "radius": 3, "kernel": "elliptical_3x3", "dilation_iterations": 1}, "metrics": {"cross_suppressed": metrics, "context": metrics["context"]}, "n_images": len(frame), "n_groups": int(frame.group_id.nunique())}
    out.mkdir(parents=True, exist_ok=True); frame.to_csv(out / "per_image.csv", index=False); (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "fold": args.fold, "method": args.method, "seed": args.seed, "cross_suppressed_macro_f1": metrics["macro_f1"], "official_test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
