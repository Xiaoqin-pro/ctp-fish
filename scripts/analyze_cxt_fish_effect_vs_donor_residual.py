"""Describe frozen F0/F1 effect by pre-specified donor residual bins."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BINS = [(-1e-12, 0.0, "0"), (0.0, .01, "(0,1%]"), (.01, .05, "(1%,5%]"), (.05, .10, "(5%,10%]"), (.10, np.inf, ">10%")]


def main() -> None:
    residual = pd.read_csv(ROOT / "experiments/cxt_fish_donor_foreground_residual.csv")
    rows = []
    for fold in (1, 2, 3):
        for seed in (3407, 2026, 17):
            f0 = pd.read_csv(ROOT / f"outputs/cxt_fish/final_outer_evaluation/fold_{fold}/F0_seed{seed}/per_image.csv")
            f1 = pd.read_csv(ROOT / f"outputs/cxt_fish/final_outer_evaluation/fold_{fold}/F1_seed{seed}/per_image.csv")
            pair = f0[["image_path", "target", "original", "cross_swap"]].rename(columns={"original": "f0_original", "cross_swap": "f0_cross"}).merge(f1[["image_path", "original", "cross_swap"]].rename(columns={"original": "f1_original", "cross_swap": "f1_cross"}), on="image_path", validate="one_to_one").merge(residual[["recipient_image_path", "donor_foreground_fraction_of_full_composite"]], left_on="image_path", right_on="recipient_image_path", validate="one_to_one")
            pair["fold"] = fold; pair["seed"] = seed
            pair["f0_correct"] = pair.f0_cross.eq(pair.target); pair["cxt_correct"] = pair.f1_cross.eq(pair.target)
            pair["original_correct"] = pair.f0_original.eq(pair.target)
            pair["dar_flip_f0"] = pair.f0_cross.eq(pair.target) & ~pair.f0_original.eq(pair.target)
            pair["dar_flip_cxt"] = pair.f1_cross.eq(pair.target) & ~pair.f1_original.eq(pair.target)
            rows.append(pair)
    frame = pd.concat(rows, ignore_index=True)
    out = []
    for low, high, name in BINS:
        part = frame[(frame.donor_foreground_fraction_of_full_composite > low) & (frame.donor_foreground_fraction_of_full_composite <= high)]
        out.append({"bin": name, "n_cell_observations": len(part), "n_unique_recipients": part.image_path.nunique(), "f0_correctness": part.f0_correct.mean() if len(part) else np.nan, "cxt_correctness": part.cxt_correct.mean() if len(part) else np.nan, "paired_correctness_delta": (part.cxt_correct.astype(float) - part.f0_correct.astype(float)).mean() if len(part) else np.nan, "dar_flip_f0": part.dar_flip_f0.mean() if len(part) else np.nan, "dar_flip_cxt": part.dar_flip_cxt.mean() if len(part) else np.nan})
    result = pd.DataFrame(out)
    result.to_csv(ROOT / "experiments/cxt_fish_effect_vs_donor_residual.csv", index=False)
    lines = ["# Effect versus donor foreground residual", "", "Pre-specified descriptive residual bins are used. These associations are not causal claims.", "", "| bin | cell observations | unique recipients | F0 correctness | CXT correctness | paired delta | F0 DAR-flip | CXT DAR-flip |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result.itertuples(index=False): lines.append(f"| {r.bin} | {r.n_cell_observations} | {r.n_unique_recipients} | {r.f0_correctness:.4f} | {r.cxt_correctness:.4f} | {r.paired_correctness_delta:+.4f} | {r.dar_flip_f0:.4f} | {r.dar_flip_cxt:.4f} |" if np.isfinite(r.f0_correctness) else f"| {r.bin} | 0 | 0 | undefined | undefined | undefined | undefined | undefined |")
    (ROOT / "reports/cxt_fish_effect_vs_donor_residual.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plt.figure(figsize=(7, 4)); plt.bar(result.bin, result.paired_correctness_delta.fillna(0)); plt.axhline(0, color="black", linewidth=.8); plt.ylabel("CXT-Fish − F0 correctness"); plt.xlabel("Donor foreground residual bin"); plt.tight_layout(); plt.savefig(ROOT / "reports/figures/cxt_fish_effect_vs_donor_residual.png", dpi=180); plt.close()
    print(json.dumps({"bins": len(result), "rows": len(frame), "official_test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
