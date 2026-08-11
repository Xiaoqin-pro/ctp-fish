"""Summarize all five donor-realization sensitivity runs without selection."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DONOR_SEEDS = (4101, 4102, 4103, 4104, 4105)
MODEL_SEEDS = (3407, 2026, 17)


def read_metric(path: Path, primary: bool = False) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("official_test_accessed") is not False:
        raise RuntimeError(f"Official TEST flag violation: {path}")
    if primary:
        return float(payload["metrics"]["cross_swap"]["macro_f1"])
    return float(payload["metrics"]["cross_composite"]["macro_f1"])


def main() -> None:
    rows = []
    primary_root = ROOT / "outputs/cxt_fish/final_outer_evaluation"
    for fold in (1, 2, 3):
        for model_seed in MODEL_SEEDS:
            f0_path = primary_root / f"fold_{fold}/F0_seed{model_seed}/metrics.json"
            f1_path = primary_root / f"fold_{fold}/F1_seed{model_seed}/metrics.json"
            rows.extend([
                {"donor_realization": "primary_3407", "donor_seed": 3407, "fold": fold, "model_seed": model_seed, "method": "F0", "cross_composite_macro_f1": read_metric(f0_path, primary=True)},
                {"donor_realization": "primary_3407", "donor_seed": 3407, "fold": fold, "model_seed": model_seed, "method": "F1", "cross_composite_macro_f1": read_metric(f1_path, primary=True)},
            ])
    sensitivity_root = ROOT / "outputs/cxt_fish/donor_sensitivity/evaluation"
    for donor_seed in DONOR_SEEDS:
        for fold in (1, 2, 3):
            for model_seed in MODEL_SEEDS:
                for method in ("F0", "F1"):
                    path = sensitivity_root / f"seed_{donor_seed}/fold_{fold}/{method}_seed{model_seed}/metrics.json"
                    rows.append({"donor_realization": f"sensitivity_{donor_seed}", "donor_seed": donor_seed, "fold": fold, "model_seed": model_seed, "method": method, "cross_composite_macro_f1": read_metric(path)})
    per_cell = pd.DataFrame(rows)
    pivot = per_cell.pivot(index=["donor_realization", "donor_seed", "fold", "model_seed"], columns="method", values="cross_composite_macro_f1").reset_index()
    pivot["delta_f1_f0"] = pivot["F1"] - pivot["F0"]
    pivot["favorable"] = pivot["delta_f1_f0"] > 0
    summary_rows = []
    for realization, group in pivot.groupby("donor_realization", sort=False):
        summary_rows.append({
            "donor_realization": realization,
            "donor_seed": int(group.donor_seed.iloc[0]),
            "f0_mean_cross_f1": float(group["F0"].mean()),
            "f1_mean_cross_f1": float(group["F1"].mean()),
            "delta_f1_f0_equal_weight_cell_mean": float(group["delta_f1_f0"].mean()),
            "delta_sd_across_9_cells": float(group["delta_f1_f0"].std(ddof=1)),
            "favorable_cells": int(group["favorable"].sum()),
            "cells": int(len(group)),
        })
    summary = pd.DataFrame(summary_rows)
    alternatives = summary[summary.donor_seed.isin(DONOR_SEEDS)]
    alt_deltas = alternatives["delta_f1_f0_equal_weight_cell_mean"]
    report = [
        "# Donor-assignment sensitivity",
        "",
        "This is a post-hoc sensitivity analysis using frozen checkpoints. The",
        "primary seed-3407 realization is reported separately and remains the",
        "frozen manuscript result. Alternative realizations 4101--4105 are all",
        "reported without selection and do not redefine the primary endpoint.",
        "",
        "| Realization | F0 mean | F1 mean | F1-F0 delta | SD across cells | Favorable cells |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        report.append(f"| {row.donor_realization} | {row.f0_mean_cross_f1:.4f} | {row.f1_mean_cross_f1:.4f} | {row.delta_f1_f0_equal_weight_cell_mean:+.4f} | {row.delta_sd_across_9_cells:.4f} | {row.favorable_cells}/{row.cells} |")
    report += [
        "",
        "## Alternative-realization range",
        "",
        f"- Mean delta across five alternatives: {alt_deltas.mean():+.4f}",
        f"- SD of realization-level deltas: {alt_deltas.std(ddof=1):.4f}",
        f"- Range: [{alt_deltas.min():+.4f}, {alt_deltas.max():+.4f}]",
        f"- Positive realization-level deltas: {int((alt_deltas > 0).sum())}/5",
        "",
        "The interpretation is descriptive: if all or most realizations are",
        "positive, the effect is not confined to the original donor assignment;",
        "if directions differ, the report must retain that heterogeneity.",
    ]
    (ROOT / "experiments/cxt_fish_donor_sensitivity_per_cell.csv").parent.mkdir(parents=True, exist_ok=True)
    per_cell.to_csv(ROOT / "experiments/cxt_fish_donor_sensitivity_per_cell.csv", index=False)
    summary.to_csv(ROOT / "experiments/cxt_fish_donor_sensitivity_summary.csv", index=False)
    pivot.to_csv(ROOT / "experiments/cxt_fish_donor_sensitivity_distribution.csv", index=False)
    (ROOT / "reports/cxt_fish_donor_sensitivity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "realizations": len(summary), "cells": len(pivot)}, indent=2))


if __name__ == "__main__":
    main()
