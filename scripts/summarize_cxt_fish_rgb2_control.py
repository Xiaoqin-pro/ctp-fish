"""Summarize F0-2RGB against frozen CXT-Fish F1 after all nine cells finish."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3407, 2026, 17)


def load_metric(path: Path, view: str):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("official_test_accessed") is not False:
        raise RuntimeError(f"Official TEST flag violation: {path}")
    metric = payload["metrics"][view]
    return {"macro_f1": metric["macro_f1"], "foreground_macro_f1": payload["metrics"]["foreground"]["macro_f1"], "same_swap_macro_f1": payload["metrics"]["same_swap"]["macro_f1"], "cross_swap_macro_f1": payload["metrics"]["cross_swap"]["macro_f1"], "dar_flip": payload["metrics"]["context"]["dar_flip"], "agreement": payload["metrics"]["context"]["prediction_agreement"]}


def bootstrap_cross_delta(cells: list[tuple[pd.DataFrame, pd.DataFrame]], replicates: int = 5000, seed: int = 3407):
    rng = np.random.default_rng(seed); labels = np.arange(16); values = np.empty(replicates, dtype=float)
    for iteration in range(replicates):
        deltas = []
        for rgb2, f1 in cells:
            groups = list(rgb2.groupby("group_id").indices.values())
            selected = rng.integers(0, len(groups), size=len(groups)); indices = np.concatenate([groups[index] for index in selected])
            target = rgb2.target.to_numpy()[indices]
            rgb = f1_score(target, rgb2.cross_swap.to_numpy()[indices], labels=labels, average="macro", zero_division=0)
            cxt = f1_score(target, f1.cross_swap.to_numpy()[indices], labels=labels, average="macro", zero_division=0)
            deltas.append(float(cxt - rgb))
        values[iteration] = float(np.mean(deltas))
    return {"point_estimate": float(values.mean()), "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))], "bootstrap_replicates": replicates, "estimand": "equal_weight_cell_mean_F1_minus_F0_2RGB_cross_class_donor_context_composite_macro_f1", "cluster_unit": "group_id", "stratification": "within_fold_group_resampling", "official_test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-bootstrap", action="store_true")
    args = parser.parse_args()
    rgb_root = ROOT / "outputs/cxt_fish/rgb2_control_outer_evaluation"
    f1_root = ROOT / "outputs/cxt_fish/final_outer_evaluation"
    rows = []; cells = []
    for fold in (1, 2, 3):
        for seed in SEEDS:
            rgb_path = rgb_root / f"fold_{fold}/F0_2RGB_seed{seed}/metrics.json"; f1_path = f1_root / f"fold_{fold}/F1_seed{seed}/metrics.json"
            if not rgb_path.is_file() or not f1_path.is_file(): raise FileNotFoundError(f"Missing paired metrics fold={fold} seed={seed}")
            rgb_payload = json.loads(rgb_path.read_text(encoding="utf-8")); f1_payload = json.loads(f1_path.read_text(encoding="utf-8"))
            if rgb_payload.get("official_test_accessed") is not False or f1_payload.get("official_test_accessed") is not False: raise RuntimeError("Official TEST flag violation")
            for method, payload in (("F0_2RGB", rgb_payload), ("CXT-Fish", f1_payload)):
                metrics = payload["metrics"]; rows.append({"fold": fold, "seed": seed, "method": method, "clean_macro_f1": metrics["original"]["macro_f1"], "foreground_macro_f1": metrics["foreground"]["macro_f1"], "same_composite_macro_f1": metrics["same_swap"]["macro_f1"], "cross_composite_macro_f1": metrics["cross_swap"]["macro_f1"], "dar_flip": metrics["context"]["dar_flip"], "agreement": metrics["context"]["prediction_agreement"]})
            rgb_frame = pd.read_csv(rgb_path.parent / "per_image.csv")
            f1_frame = pd.read_csv(f1_path.parent / "per_image.csv")
            keys = ["image_path", "group_id", "target"]
            if not rgb_frame[keys].equals(f1_frame[keys]):
                raise RuntimeError(f"Paired RGB2/F1 rows differ at fold={fold}, seed={seed}")
            cells.append((rgb_frame, f1_frame))
    frame = pd.DataFrame(rows)
    metric_columns = ["clean_macro_f1", "foreground_macro_f1", "same_composite_macro_f1", "cross_composite_macro_f1", "dar_flip", "agreement"]
    summary = frame.groupby("method")[metric_columns].mean().reset_index()
    summary.to_csv(ROOT / "experiments/cxt_fish_rgb2_control_summary.csv", index=False)
    frame.to_csv(ROOT / "experiments/cxt_fish_rgb2_control_per_cell.csv", index=False)
    bootstrap_path = ROOT / "experiments/cxt_fish_rgb2_vs_f1_bootstrap_5000.json"
    if args.reuse_bootstrap and bootstrap_path.is_file():
        bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    else:
        bootstrap = bootstrap_cross_delta(cells)
        bootstrap_path.write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    report = ["# F0-2RGB reviewer control", "", "This is a post-hoc supervision- and compute-matched mechanism control. It is not a new confirmatory endpoint and does not alter the frozen CXT-Fish result.", "", "| Method | Clean | Foreground | Same composite | Cross composite | DAR-flip | Agreement |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False): report.append(f"| {row.method} | {row.clean_macro_f1:.4f} | {row.foreground_macro_f1:.4f} | {row.same_composite_macro_f1:.4f} | {row.cross_composite_macro_f1:.4f} | {row.dar_flip:.4f} | {row.agreement:.4f} |")
    report += ["", f"Exploratory post-hoc CXT-Fish minus F0-2RGB cross-composite cell-mean delta: {bootstrap['point_estimate']:+.4f}, 95% interval [{bootstrap['ci95'][0]:+.4f}, {bootstrap['ci95'][1]:+.4f}].", "Official TEST accessed: false."]
    (ROOT / "reports/cxt_fish_rgb2_control_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "cells": len(cells), "bootstrap": bootstrap}, indent=2))


if __name__ == "__main__":
    main()
