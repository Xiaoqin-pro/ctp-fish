"""Produce the fixed, stratified visual integrity gate for Phase 1D composites."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.donor_context import donor_aware_swap


def mask_stats(path: str) -> tuple[float, bool, tuple[int, int]]:
    mask = Image.open(path).convert("L"); value = np.asarray(mask) > 0
    edge = bool(value[0].any() or value[-1].any() or value[:, 0].any() or value[:, -1].any())
    return float(value.mean()), edge, mask.size


def select(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    work = frame.sort_values("recipient_image_path").copy()
    stats = [mask_stats(path) for path in work.recipient_mask_path]
    work["recipient_mask_fraction"] = [x[0] for x in stats]; work["edge_touch"] = [x[1] for x in stats]
    work["area_bin"] = pd.qcut(work.recipient_mask_fraction, 3, labels=["small", "medium", "large"], duplicates="drop").astype(str)
    chosen: list[int] = []
    # Guarantee all available species, then cycle deterministic area/edge strata.
    for _, group in work.groupby("recipient_species_id", sort=True): chosen.append(int(group.index[0]))
    strata = [(area, edge) for area in ("small", "medium", "large") for edge in (False, True)]
    cursor = 0
    while len(chosen) < min(count, len(work)):
        area, edge = strata[cursor % len(strata)]; cursor += 1
        candidates = work.loc[work.area_bin.eq(area) & work.edge_touch.eq(edge) & ~work.index.isin(chosen)]
        if candidates.empty: candidates = work.loc[~work.index.isin(chosen)]
        chosen.append(int(candidates.index[0]))
    return work.loc[chosen].reset_index(drop=True)


def panel(recipient: Image.Image, donor: Image.Image, composite: Image.Image, width: int = 180) -> Image.Image:
    height = 135; result = Image.new("RGB", (width * 3, height + 18), "white")
    for index, image in enumerate((recipient, donor, composite)):
        view = image.convert("RGB"); view.thumbnail((width, height)); canvas = Image.new("RGB", (width, height), "black"); canvas.paste(view, ((width - view.width) // 2, (height - view.height) // 2)); result.paste(canvas, (index * width, 18))
    draw = ImageDraw.Draw(result); draw.text((2, 2), "recipient                 donor                 composite", fill="black")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", default="splits/f4k16t_phase1d_train_donors.csv"); parser.add_argument("--output", default="outputs/cxt_fish/phase1d_composite_gate"); parser.add_argument("--count", type=int, default=50); parser.add_argument("--feather-radius", type=int, default=3); args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); selected = select(pd.read_csv(args.manifest), args.count)
    sheets: list[Image.Image] = []; checks = []
    for row in selected.itertuples(index=False):
        recipient, donor = Image.open(row.recipient_image_path).convert("RGB"), Image.open(row.donor_image_path).convert("RGB")
        rmask, dmask = Image.open(row.recipient_mask_path).convert("L"), Image.open(row.donor_mask_path).convert("L")
        composite = donor_aware_swap(recipient, rmask, donor, dmask, feather_radius=args.feather_radius)
        sheets.append(panel(recipient, donor, composite)); checks.append({"recipient_image_path": row.recipient_image_path, "donor_image_path": row.donor_image_path, "recipient_species_id": row.recipient_species_id, "donor_species_id": row.donor_species_id, "recipient_size": recipient.size, "recipient_mask_size": rmask.size, "donor_size": donor.size, "donor_mask_size": dmask.size, "composite_size": composite.size, "size_ok": recipient.size == rmask.size == composite.size and donor.size == dmask.size})
    for page_start in range(0, len(sheets), 10):
        page = Image.new("RGB", (540, 153 * min(10, len(sheets) - page_start)), "white")
        for index, item in enumerate(sheets[page_start:page_start + 10]): page.paste(item, (0, index * 153))
        page.save(output / f"page_{page_start // 10 + 1:02d}.png")
    audit = pd.DataFrame(checks); audit.to_csv(output / "composite_gate_audit.csv", index=False)
    summary = {"manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(), "selected": len(selected), "size_checks_passed": int(audit.size_ok.sum()), "same_species_pairs": int(selected.recipient_species_id.eq(selected.donor_species_id).sum()), "same_track_pairs": int(selected.recipient_group_id.eq(selected.donor_group_id).sum()), "auto_gate_pass": bool(audit.size_ok.all() and not selected.recipient_species_id.eq(selected.donor_species_id).any() and not selected.recipient_group_id.eq(selected.donor_group_id).any())}
    (output / "summary.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8"); print(summary)


if __name__ == "__main__": main()
