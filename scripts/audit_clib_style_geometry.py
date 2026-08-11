"""Train-only geometry audit for the deterministic CLIB-style views.

This audit reads only a fixed prefix of one frozen outer-train manifest. It
does not load predictions, inner-dev, outer-test or official TEST data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.clib_style_views import corner_boxes, subject_box


ROOT = Path(__file__).resolve().parents[1]


def mask_array(mask: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(mask.convert("L").resize(size, Image.Resampling.NEAREST), dtype=np.uint8) > 0


def inside(box: tuple[int, int, int, int], width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    result = np.zeros((height, width), dtype=bool)
    result[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_clib_style_outer_v1_1.yaml")
    parser.add_argument("--fold", choices=["1", "2", "3"], default="1")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--output", default="outputs/cxt_fish/clib_style_geometry_audit_v1_1")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    manifest = ROOT / cfg["outer_manifest_root"] / f"fold_{args.fold}" / "outer_train.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    all_records = pd.read_csv(manifest).sort_values(["species_id", "group_id", "image_path"])
    # Fixed species-stratified selection: two images per class for the default
    # 32-record audit. This is determined before looking at any audit value.
    per_class = max(1, int(args.limit) // max(1, all_records.species_id.astype(str).nunique()))
    records = (all_records.assign(_species=all_records.species_id.astype(str))
               .groupby("_species", sort=True, group_keys=False)
               .head(per_class).head(int(args.limit)).reset_index(drop=True))
    if len(records) != int(args.limit):
        raise ValueError("fixed geometry audit requires the requested number of train records")
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    contact = Image.new("RGB", (4 * 320, 8 * 240), "white")
    draw = ImageDraw.Draw(contact)
    for index, record in records.iterrows():
        image = Image.open(Path(record.image_path)).convert("RGB")
        mask = mask_array(Image.open(Path(record.mask_path)), image.size)
        width, height = image.size
        subject = inside(subject_box(image, float(cfg["views"]["ratio"])), width, height)
        corners = [inside(box, width, height) for box in corner_boxes(image, float(cfg["views"]["ratio"]))]
        corner_union = np.logical_or.reduce(corners)
        count = int(mask.sum()); subject_intersection = int(np.logical_and(mask, subject).sum())
        ys, xs = np.where(mask)
        cx, cy = (float(xs.mean()), float(ys.mean())) if count else (None, None)
        rows.append({
            "image_path": str(record.image_path), "group_id": str(record.group_id), "species_id": str(record.species_id),
            "mask_pixels": count, "subject_mask_recall": subject_intersection / count if count else None,
            "subject_mask_iou": subject_intersection / int(np.logical_or(mask, subject).sum()) if int(np.logical_or(mask, subject).sum()) else None,
            "mask_centroid_in_subject": bool(subject[int(round(cy)), int(round(cx))]) if count else False,
            "corner_mask_recall": int(np.logical_and(mask, corner_union).sum()) / count if count else None,
            "corner_contains_any_mask": bool(np.logical_and(mask, corner_union).any()),
        })
        thumb = image.copy(); thumb.thumbnail((300, 220))
        xoff, yoff = (index % 4) * 320 + 10, (index // 4) * 240 + 10
        contact.paste(thumb, (xoff, yoff)); scale_x, scale_y = thumb.width / width, thumb.height / height
        for box, color in [(subject_box(image, float(cfg["views"]["ratio"])), "red"), *[(b, "yellow") for b in corner_boxes(image, float(cfg["views"]["ratio"]))]]:
            x0, y0, x1, y1 = box
            draw.rectangle((xoff + x0 * scale_x, yoff + y0 * scale_y, xoff + x1 * scale_x, yoff + y1 * scale_y), outline=color, width=2)
        draw.text((xoff, yoff + thumb.height + 2), f"{index:02d} class={record.species_id}", fill="black")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "records.csv", index=False)
    contact.save(output / "contact_sheet_32.png")
    summary = {
        "status": "complete", "manifest": str(manifest), "manifest_sha256": __import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
        "n_records": len(frame), "classes": sorted(frame.species_id.unique().tolist()),
        "mean_subject_mask_recall": float(frame.subject_mask_recall.mean()),
        "zero_subject_recall": int((frame.subject_mask_recall == 0).sum()),
        "mean_subject_mask_iou": float(frame.subject_mask_iou.mean()),
        "centroid_in_subject_fraction": float(frame.mask_centroid_in_subject.mean()),
        "corner_contains_any_fraction": float(frame.corner_contains_any_mask.mean()),
        "official_test_accessed": False, "inner_dev_accessed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
