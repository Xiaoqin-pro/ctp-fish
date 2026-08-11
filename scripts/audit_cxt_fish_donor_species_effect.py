"""Describe primary donor-species weighting effects from frozen predictions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3407, 2026, 17)


def main() -> None:
    rows = []
    matrix_rows = []
    for fold in (1, 2, 3):
        manifest = pd.read_csv(ROOT / f"outputs/cxt_fish/final_outer_manifests/fold_{fold}/outer_context_swap.csv")
        cross = manifest.loc[manifest.swap_type.eq("cross_class")].copy()
        cross = cross.loc[cross.supported.astype(str).str.lower().eq("true")]
        for seed in SEEDS:
            f0 = pd.read_csv(ROOT / f"outputs/cxt_fish/final_outer_evaluation/fold_{fold}/F0_seed{seed}/per_image.csv")
            f1 = pd.read_csv(ROOT / f"outputs/cxt_fish/final_outer_evaluation/fold_{fold}/F1_seed{seed}/per_image.csv")
            cols = ["image_path", "group_id", "target", "cross_swap", "original"]
            pair = f0[cols].rename(columns={"cross_swap": "f0_cross", "original": "f0_original"}).merge(
                f1[cols].rename(columns={"cross_swap": "f1_cross", "original": "f1_original"}),
                on=["image_path", "group_id", "target"], validate="one_to_one"
            ).merge(cross[["recipient_image_path", "donor_species_id", "donor_image_path", "donor_group_id"]],
                    left_on="image_path", right_on="recipient_image_path", validate="one_to_one")
            pair["donor_species"] = pair.donor_species_id.astype(str)
            pair["f0_correct"] = pair.f0_cross.eq(pair.target)
            pair["f1_correct"] = pair.f1_cross.eq(pair.target)
            pair["delta_correct"] = pair.f1_correct.astype(float) - pair.f0_correct.astype(float)
            pair["fold"] = fold; pair["seed"] = seed
            rows.append(pair)
            for (donor_species, target_species), part in pair.groupby(["donor_species", "target"], sort=True):
                matrix_rows.append({"fold": fold, "seed": seed, "donor_species": donor_species,
                                    "recipient_species": str(target_species), "n": len(part),
                                    "f0_correct": part.f0_correct.mean(), "cxt_correct": part.f1_correct.mean(),
                                    "delta_correct": part.delta_correct.mean()})
    frame = pd.concat(rows, ignore_index=True)
    species = frame.groupby("donor_species", sort=True).agg(
        n_recipients=("image_path", "size"), unique_donor_images=("donor_image_path", "nunique"),
        unique_donor_groups=("donor_group_id", "nunique"), f0_correct=("f0_correct", "mean"),
        cxt_correct=("f1_correct", "mean"), effect=("delta_correct", "mean")
    ).reset_index()
    species["fraction_of_all_cross_composites"] = species.n_recipients / species.n_recipients.sum()
    species.to_csv(ROOT / "experiments/cxt_fish_donor_species_effects.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(ROOT / "experiments/cxt_fish_recipient_donor_species_matrix_effects.csv", index=False)
    natural = float(frame.delta_correct.mean())
    equal = float(species.effect.mean())
    report = ["# Donor-species effect audit", "", "Frozen primary seed-3407 donor pairing with all nine frozen F0/F1 model cells. The row-level correctness difference is descriptive and is not a new primary estimand.", "", f"- Frozen natural-manifest-weight correctness effect: {natural:+.6f}.", f"- Donor-species-equal-weight sensitivity: {equal:+.6f} (post-hoc descriptive reweighting).", "", "| donor species | recipients | fraction | donor images | donor groups | F0 correctness | CXT correctness | effect |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in species.itertuples(index=False):
        report.append(f"| {r.donor_species} | {r.n_recipients} | {r.fraction_of_all_cross_composites:.3f} | {r.unique_donor_images} | {r.unique_donor_groups} | {r.f0_correct:.4f} | {r.cxt_correct:.4f} | {r.effect:+.4f} |")
    report += ["", "The donor-species-equal-weight value is a sensitivity description, not a replacement for the frozen natural-manifest-weight result. No causal interpretation is made."]
    (ROOT / "reports/cxt_fish_donor_species_effect_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    plt.figure(figsize=(8, 4)); plt.bar(species.donor_species.astype(str), species.effect); plt.axhline(0, color="black", linewidth=.8); plt.ylabel("CXT-Fish − F0 correctness"); plt.xlabel("Donor species"); plt.tight_layout(); plt.savefig(ROOT / "reports/figures/cxt_fish_donor_species_effects.png", dpi=180); plt.close()
    print(json.dumps({"rows": len(frame), "donor_species": len(species), "natural_effect": natural, "equal_weight_effect": equal, "official_test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
