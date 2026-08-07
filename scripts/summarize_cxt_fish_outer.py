"""Summarize the completed frozen CXT-Fish outer-test evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]


def paired_group_bootstrap(root: Path, seed: int = 3407, samples: int = 1000) -> dict[str, tuple[float, float]]:
    """Bootstrap mean F1-F0 deltas by resampling trajectory groups within pairs."""
    pairs = []
    for seed_value in (3407, 2026, 17):
        for fold in (1, 2, 3):
            f0 = pd.read_csv(root / f"fold_{fold}" / f"F0_seed{seed_value}" / "per_image.csv")
            f1 = pd.read_csv(root / f"fold_{fold}" / f"F1_seed{seed_value}" / "per_image.csv")
            key = ["image_path", "group_id", "target", "cross_donor_target"]
            if not f0[key].equals(f1[key]):
                raise RuntimeError(f"F0/F1 paired image order mismatch for fold {fold}, seed {seed_value}")
            pairs.append((f0, f1, list(f0.groupby("group_id").indices.values())))
    rng = np.random.default_rng(seed)
    values = {name: np.empty(samples, dtype=float) for name in ("original_macro_f1", "cross_swap_macro_f1", "track_balanced_accuracy")}
    labels = np.arange(16)
    for iteration in range(samples):
        replicate = {name: [] for name in values}
        for f0, f1, groups in pairs:
            selected = rng.integers(0, len(groups), size=len(groups))
            indices = np.concatenate([groups[index] for index in selected])
            target = f0.target.to_numpy()[indices]
            replicate["original_macro_f1"].append(float(f1_score(target, f1.original.to_numpy()[indices], labels=labels, average="macro", zero_division=0) - f1_score(target, f0.original.to_numpy()[indices], labels=labels, average="macro", zero_division=0)))
            replicate["cross_swap_macro_f1"].append(float(f1_score(target, f1.cross_swap.to_numpy()[indices], labels=labels, average="macro", zero_division=0) - f1_score(target, f0.cross_swap.to_numpy()[indices], labels=labels, average="macro", zero_division=0)))
            f0_group = f0.assign(correct=f0.original.eq(f0.target)).groupby("group_id").correct.mean()
            f1_group = f1.assign(correct=f1.original.eq(f1.target)).groupby("group_id").correct.mean()
            selected_groups = f0_group.index.to_numpy()[selected]
            replicate["track_balanced_accuracy"].append(float(f1_group.loc[selected_groups].mean() - f0_group.loc[selected_groups].mean()))
        for name in values:
            values[name][iteration] = float(np.mean(replicate[name]))
    return {name: (float(np.quantile(value, .025)), float(np.quantile(value, .975))) for name, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", default="outputs/cxt_fish/final_outer_evaluation")
    parser.add_argument("--output-csv", default="experiments/cxt_fish_outer_results.csv")
    parser.add_argument("--output-report", default="reports/cxt_fish_outer_confirmation_report.md")
    args = parser.parse_args()
    root = ROOT / args.evaluation_root
    rows = []
    for seed in (3407, 2026, 17):
        for fold in (1, 2, 3):
            for method in ("F0", "F1"):
                path = root / f"fold_{fold}" / f"{method}_seed{seed}" / "metrics.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("official_test_accessed") is not False:
                    raise RuntimeError(f"Official TEST flag is not false: {path}")
                if payload.get("partition") != "outer_test":
                    raise RuntimeError(f"Unexpected partition: {path}")
                metrics = payload["metrics"]
                row = {"seed": seed, "fold": fold, "method": method,
                       "n_images": payload["n_images"], "n_groups": payload["n_groups"]}
                for view in ("original", "foreground", "same_swap", "cross_swap"):
                    for name in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "head_f1", "mid_f1", "tail_f1", "track_balanced_accuracy"):
                        row[f"{view}_{name}"] = metrics[view][name]
                for name in ("dar", "dar_flip", "prediction_agreement", "delta_foreground_macro_f1", "delta_same_swap_macro_f1", "delta_cross_swap_macro_f1"):
                    row[name] = metrics["context"][name]
                rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["seed", "fold", "method"]).reset_index(drop=True)
    if len(frame) != 18 or frame.duplicated(["seed", "fold", "method"]).any():
        raise RuntimeError("Expected exactly 18 unique outer cells")
    output_csv = ROOT / args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)

    means = frame.groupby("method")[["original_macro_f1", "original_tail_f1", "original_track_balanced_accuracy", "cross_swap_macro_f1", "dar_flip"]].mean()
    sds = frame.groupby("method")[["original_macro_f1", "original_tail_f1", "original_track_balanced_accuracy", "cross_swap_macro_f1", "dar_flip"]].std(ddof=1)
    pivot = frame.pivot(index=["seed", "fold"], columns="method")
    deltas = pd.DataFrame({
        "original_macro_f1": pivot["original_macro_f1"]["F1"] - pivot["original_macro_f1"]["F0"],
        "original_tail_f1": pivot["original_tail_f1"]["F1"] - pivot["original_tail_f1"]["F0"],
        "track_balanced_accuracy": pivot["original_track_balanced_accuracy"]["F1"] - pivot["original_track_balanced_accuracy"]["F0"],
        "cross_swap_macro_f1": pivot["cross_swap_macro_f1"]["F1"] - pivot["cross_swap_macro_f1"]["F0"],
        "dar_flip": pivot["dar_flip"]["F1"] - pivot["dar_flip"]["F0"],
    })
    report_lines = [
        "# CXT-Fish frozen outer confirmation",
        "",
        "## Protocol",
        "",
        "This report summarizes the pre-registered three-fold, three-seed outer confirmation after all 18 F0/F1 cells were trained and frozen. Checkpoints were selected only on the fold-specific inner development split. The outer-test partition was read only in this final evaluation; official TEST was not accessed.",
        "",
        "- Cells: 3 folds x 3 registered seeds (3407, 2026, 17) x F0/F1",
        "- Primary views: original RGB, foreground-only, same-class cross-track swap, cross-class swap",
        "- Unit of generalization: group/trajectory-disjoint outer folds",
        "- Outer-test access before this evaluation: false",
        "- Official TEST accessed: false",
        "",
        "## Mean +/- SD across 9 fold-seed pairs",
        "",
        "| Method | Original macro-F1 | Original tail-F1 | Track-balanced accuracy | Cross-swap macro-F1 | DAR-flip |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("F0", "F1"):
        values = means.loc[method]; sd = sds.loc[method]
        report_lines.append(f"| {method} | {values.original_macro_f1:.4f} +/- {sd.original_macro_f1:.4f} | {values.original_tail_f1:.4f} +/- {sd.original_tail_f1:.4f} | {values.original_track_balanced_accuracy:.4f} +/- {sd.original_track_balanced_accuracy:.4f} | {values.cross_swap_macro_f1:.4f} +/- {sd.cross_swap_macro_f1:.4f} | {values.dar_flip:.4f} +/- {sd.dar_flip:.4f} |")
    bootstrap = paired_group_bootstrap(root)
    report_lines += [
        "",
        "## F1 - F0 paired deltas",
        "",
        "| Metric | Mean delta | SD | Paired group bootstrap 95% CI | Favorable pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in (("original_macro_f1", "Original macro-F1"), ("original_tail_f1", "Original tail-F1"), ("track_balanced_accuracy", "Track-balanced accuracy"), ("cross_swap_macro_f1", "Cross-swap macro-F1"), ("dar_flip", "DAR-flip (lower is better)")):
        value = deltas[name]
        favorable = (value < 0).sum() if name == "dar_flip" else (value > 0).sum()
        ci = bootstrap.get(name)
        ci_text = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci is not None else "not computed"
        report_lines.append(f"| {label} | {value.mean():+.4f} | {value.std(ddof=1):.4f} | {ci_text} | {int(favorable)}/9 |")
    report_lines += [
        "",
        "## Interpretation",
        "",
        "F1 is not an overall clean-accuracy improvement method in this outer confirmation: its mean original macro-F1 is slightly below F0. Its consistent signal is context robustness: cross-class context-swap macro-F1 is higher on average and DAR-flip is lower on average, while tail-F1 and track-balanced accuracy are approximately preserved or slightly improved. The appropriate claim is therefore foreground-sufficiency training as a simple robustness intervention, not a universally superior classifier.",
        "",
        "The complete per-cell table is in `experiments/cxt_fish_outer_results.csv`. Raw per-image outputs remain local under `outputs/cxt_fish/final_outer_evaluation/` and are not part of the tracked result table.",
    ]
    output_report = ROOT / args.output_report
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "cells": len(frame), "output_csv": str(output_csv), "output_report": str(output_report)}, indent=2))


if __name__ == "__main__":
    main()
