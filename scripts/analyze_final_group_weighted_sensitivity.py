"""Generate post-hoc group-weighted robustness summaries from frozen predictions.

This script does not train, infer, or access the official Fish4Knowledge TEST
partition. It consumes the frozen outer per-image CSVs and outer-test manifests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTER_EVAL = ROOT / "outputs" / "cxt_fish" / "final_outer_evaluation"
MANIFEST_ROOT = ROOT / "outputs" / "cxt_fish" / "final_outer_manifests"
EXP = ROOT / "experiments"
REPORTS = ROOT / "reports"


def _cell_csv(fold: int, method: str, seed: int) -> Path:
    p = OUTER_EVAL / f"fold_{fold}" / f"{method}_seed{seed}" / "per_image.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _metric(df: pd.DataFrame, column: str) -> tuple[float, float]:
    correct = (df[column].to_numpy() == df["target"].to_numpy()).astype(float)
    work = df[["group_id", "target"]].copy()
    work["correct"] = correct
    group_acc = work.groupby("group_id", sort=True)["correct"].mean()
    species_group = work.groupby(["target", "group_id"], sort=True)["correct"].mean()
    species_group_balanced = species_group.groupby(level=0).mean().mean()
    return float(group_acc.mean()), float(species_group_balanced)


def main() -> None:
    records: list[dict[str, object]] = []
    expected_cells = {(fold, method, seed) for fold in (1, 2, 3)
                      for method in ("F0", "F1") for seed in (17, 2026, 3407)}
    seen: set[tuple[int, str, int]] = set()
    paired_identity: dict[tuple[int, int], pd.DataFrame] = {}
    for fold, method, seed in sorted(expected_cells):
        path = _cell_csv(fold, method, seed)
        df = pd.read_csv(path)
        key = (fold, seed)
        ids = df[["image_path", "group_id", "target"]].copy()
        if key in paired_identity:
            if not ids.equals(paired_identity[key]):
                raise ValueError(f"paired image/group/target identity mismatch: fold={fold}, seed={seed}")
        else:
            paired_identity[key] = ids
        for view in ("cross_swap",):
            gba, sgb = _metric(df, view)
            records.append({
                "fold": fold,
                "seed": seed,
                "method": method,
                "view": view,
                "group_balanced_accuracy": gba,
                "species_group_balanced_accuracy": sgb,
                "n_images": int(len(df)),
                "n_groups": int(df["group_id"].nunique()),
                "n_species": int(df["target"].nunique()),
                "official_test_accessed": False,
            })
        seen.add((fold, method, seed))
    if seen != expected_cells:
        raise ValueError(f"expected {len(expected_cells)} cells, found {len(seen)}")

    per_cell = pd.DataFrame(records).sort_values(["fold", "seed", "method"])
    per_cell.to_csv(EXP / "cxt_fish_group_weighted_robustness_per_cell.csv", index=False)
    summaries: list[dict[str, object]] = []
    for metric in ("group_balanced_accuracy", "species_group_balanced_accuracy"):
        pivot = per_cell.pivot_table(index=["fold", "seed"], columns="method", values=metric)
        delta = pivot["F1"] - pivot["F0"]
        summaries.extend([
            {
                "metric": metric,
                "F0_mean": float(pivot["F0"].mean()),
                "F0_sd": float(pivot["F0"].std(ddof=1)),
                "F1_mean": float(pivot["F1"].mean()),
                "F1_sd": float(pivot["F1"].std(ddof=1)),
                "delta_pp": float(delta.mean() * 100.0),
                "delta_sd_pp": float(delta.std(ddof=1) * 100.0),
                "favourable_cells": int((delta > 0).sum()),
                "n_cells": int(len(delta)),
                "estimand": "equal-weight nine cell mean of fold-seed group-weighted accuracy differences",
                "official_test_accessed": False,
            }
        ])
    summary = pd.DataFrame(summaries)
    summary.to_csv(EXP / "cxt_fish_group_weighted_robustness_summary.csv", index=False)

    count_rows: list[dict[str, object]] = []
    for fold in (1, 2, 3):
        p = MANIFEST_ROOT / f"fold_{fold}" / "outer_test.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p, dtype={"species_id": str, "group_id": str})
        for species, g in df.groupby("species_id", sort=True):
            count_rows.append({
                "fold": fold,
                "species_id": species,
                "n_images": int(len(g)),
                "n_recorded_groups": int(g["group_id"].nunique()),
                "official_test_accessed": False,
            })
    counts = pd.DataFrame(count_rows).sort_values(["fold", "species_id"])
    if counts["species_id"].nunique() != 16 or len(counts) != 48:
        raise ValueError("expected exactly 16 species x 3 folds")
    counts.to_csv(EXP / "cxt_fish_outer_recorded_group_counts_16x3.csv", index=False)
    counts.pivot(index="species_id", columns="fold", values="n_recorded_groups").to_csv(
        EXP / "cxt_fish_outer_recorded_group_counts_16x3_wide.csv"
    )

    report = {
        "analysis": "post-hoc group-weighted robustness sensitivity",
        "source": "frozen outer per-image predictions and frozen outer-test manifests",
        "cells": 18,
        "view": "cross_swap",
        "metrics": summary.to_dict(orient="records"),
        "group_count_table": {
            "rows": int(len(counts)),
            "species": int(counts["species_id"].nunique()),
            "folds": 3,
            "minimum_groups_per_fold_species": int(counts["n_recorded_groups"].min()),
            "maximum_groups_per_fold_species": int(counts["n_recorded_groups"].max()),
        },
        "official_test_accessed": False,
    }
    (EXP / "cxt_fish_group_weighted_robustness_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    table_lines = [
        "| Metric | F0 mean | F0 SD | CXT-Fish mean | CXT-Fish SD | Delta (pp) | Delta SD (pp) | Favourable cells |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        table_lines.append(
            f"| {row['metric']} | {row['F0_mean']:.4f} | {row['F0_sd']:.4f} | {row['F1_mean']:.4f} | {row['F1_sd']:.4f} | {row['delta_pp']:.2f} | {row['delta_sd_pp']:.2f} | {row['favourable_cells']}/{row['n_cells']} |"
        )
    md = [
        "# CXT-Fish group-weighted robustness sensitivity",
        "",
        "This post-hoc analysis uses only frozen outer per-image predictions and manifests; it does not train or re-infer models and does not access the official Fish4Knowledge TEST partition.",
        "",
        "The primary manuscript cross-class composite macro-F1 remains class-balanced but image-weighted within species. Here, each recorded group receives equal weight within species, and species then receive equal weight. The nine fold–seed cells are averaged with equal weight.",
        "",
        *table_lines,
        "",
        f"The 16 × 3 fold–species table contains {len(counts)} strata. The number of recorded groups per stratum ranges from {counts['n_recorded_groups'].min()} to {counts['n_recorded_groups'].max()}; this finite-cluster structure is reported explicitly rather than regularized.",
        "",
        "All rows record official_test_accessed=false.",
    ]
    (REPORTS / "cxt_fish_group_weighted_robustness_sensitivity.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
