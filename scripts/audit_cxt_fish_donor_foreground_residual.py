"""Audit donor-fish pixels remaining in the frozen primary composites."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.context_views import feathered_mask  # noqa: E402


def describe(frame: pd.DataFrame, column: str) -> dict:
    values = frame[column].to_numpy(float)
    q = np.quantile(values, [0.25, 0.5, 0.75, 0.9, 0.95])
    return {"mean": float(values.mean()), "median": float(q[1]), "iqr": [float(q[0]), float(q[2])], "q25": float(q[0]), "q90": float(q[3]), "q95": float(q[4]), "max": float(values.max()), "gt0": float(np.mean(values > 0)), "gt1pct": float(np.mean(values > .01)), "gt5pct": float(np.mean(values > .05)), "gt10pct": float(np.mean(values > .10)), "gt20pct": float(np.mean(values > .20))}


def main() -> None:
    rows = []
    for fold in (1, 2, 3):
        manifest = pd.read_csv(ROOT / f"outputs/cxt_fish/final_outer_manifests/fold_{fold}/outer_context_swap.csv")
        cross = manifest.loc[(manifest.swap_type == "cross_class") & manifest.supported.astype(str).str.lower().eq("true")]
        for r in cross.itertuples(index=False):
            recipient = np.asarray(Image.open(r.recipient_image_path).convert("RGB"))
            donor = np.asarray(Image.open(r.donor_image_path).convert("RGB"))
            recipient_mask = np.asarray(Image.open(r.recipient_mask_path).convert("L").resize((recipient.shape[1], recipient.shape[0]), Image.Resampling.NEAREST)) > 0
            donor_mask = np.asarray(Image.open(r.donor_mask_path).convert("L").resize((recipient.shape[1], recipient.shape[0]), Image.Resampling.NEAREST)) > 0
            alpha = feathered_mask(Image.open(r.recipient_mask_path), (recipient.shape[1], recipient.shape[0]), 3)
            donor_area = float(np.asarray(Image.open(r.donor_mask_path).convert("L")).astype(bool).mean())
            outside = donor_mask & ~recipient_mask
            donor_region = ~recipient_mask
            rows.append({"fold": fold, "recipient_image_path": r.recipient_image_path, "recipient_species": str(r.recipient_species_id), "recipient_group": str(r.recipient_group_id), "donor_image_path": r.donor_image_path, "donor_species": str(r.donor_species_id), "donor_group": str(r.donor_group_id), "donor_mask_original_area_fraction": donor_area, "donor_foreground_pixels_after_resize": int(donor_mask.sum()), "donor_foreground_pixels_outside_recipient_hard_mask": int(outside.sum()), "donor_foreground_fraction_of_full_composite": float(outside.mean()), "donor_foreground_fraction_of_donor_region": float(outside.sum() / max(1, donor_region.sum())), "donor_foreground_fraction_alpha_weighted": float((donor_mask * (1.0 - alpha)).mean()), "donor_foreground_fraction_donor_region_alpha_weighted": float((donor_mask * (1.0 - alpha)).sum() / max(1.0, (1.0 - alpha).sum()))})
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "experiments/cxt_fish_donor_foreground_residual.csv", index=False)
    col = "donor_foreground_fraction_of_full_composite"
    lines = ["# Donor foreground residual audit", "", "This audit uses the frozen primary donor manifest and official recipient/donor masks. It is descriptive and does not alter any composite or model output.", "", f"- Pairs audited: {len(frame)}", "- Hard-mask residual definition: resized donor foreground outside the resized recipient hard mask.", "- Alpha-weighted residual is also reported using the frozen recipient feathering geometry.", "", "## Distribution"]
    for name, value in describe(frame, col).items(): lines.append(f"- {name}: {value}")
    lines += ["", "## By donor species", "", "| donor species | n | mean residual | median residual | >1% | >5% | >10% | >20% |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for species, part in frame.groupby("donor_species", sort=True):
        lines.append(f"| {species} | {len(part)} | {part[col].mean():.4f} | {part[col].median():.4f} | {(part[col] > .01).mean():.3f} | {(part[col] > .05).mean():.3f} | {(part[col] > .10).mean():.3f} | {(part[col] > .20).mean():.3f} |")
    (ROOT / "reports/cxt_fish_donor_foreground_residual_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plt.figure(figsize=(7, 4)); plt.hist(frame[col], bins=30); plt.xlabel("Donor-fish residual fraction of composite"); plt.ylabel("Pairs"); plt.tight_layout(); plt.savefig(ROOT / "reports/figures/cxt_fish_donor_residual_distribution.png", dpi=180); plt.close()
    print(json.dumps({"pairs": len(frame), "mean": float(frame[col].mean()), "median": float(frame[col].median()), "official_test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
