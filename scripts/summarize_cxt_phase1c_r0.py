"""Summarize the frozen three-seed Phase 1C R0 validation replication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


SEEDS = (3407, 2026, 17)
VARIANTS = ("F0", "F1", "F2", "F3")


def metric_path(seed: int, variant: str) -> Path:
    if variant == "F0":
        return Path(f"outputs/cxt_fish/phase1a_replication/A1_s1_ce_seed{seed}/evaluation_phase1c_f0_seed{seed}/metrics_val.json")
    root = "phase1c_pilot" if seed == 3407 else "phase1c_replication"
    return Path(f"outputs/cxt_fish/{root}/{variant}_seed{seed}/evaluation_phase1c_{variant.lower()}_seed{seed}/metrics_val.json")


def tier_map(config: dict) -> dict[str, str]:
    split = pd.read_csv(config["track_split_path"])
    counts = split.loc[split.split == "train", "species_id"].astype(str).value_counts()
    ordered = sorted(counts.index, key=lambda value: (-int(counts[value]), value))
    return {label: tier for tier, labels in zip(("head", "mid", "tail"), np.array_split(np.asarray(ordered, dtype=object), 3)) for label in labels}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_phase1c.yaml")
    parser.add_argument("--summary", default="experiments/cxt_fish_phase1c_r0_summary.csv")
    parser.add_argument("--deltas", default="experiments/cxt_fish_phase1c_r0_deltas_vs_f0.csv")
    parser.add_argument("--report", default="reports/cxt_fish_phase1c_r0_replication.md")
    args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); tiers = tier_map(cfg)
    rows: list[dict] = []
    for seed in SEEDS:
        for variant in VARIANTS:
            path = metric_path(seed, variant); value = json.loads(path.read_text(encoding="utf-8"))
            if value["partition"] != "val" or value["internal_test_accessed"] or value["outer_folds_accessed"]:
                raise ValueError(f"invalid provenance: {path}")
            original_f1 = dict(zip(map(str, value["original"]["class_ids"]), value["original"]["per_class_f1"]))
            rows.append({
                "seed": seed, "variant": variant,
                "original_macro_f1": value["original"]["macro_f1"],
                "original_tail_f1": float(np.mean([original_f1[k] for k, v in tiers.items() if v == "tail"])),
                "track_balanced_accuracy": value["original"]["track_balanced_accuracy"],
                "foreground_macro_f1": value["foreground"]["macro_f1"],
                "same_swap_macro_f1": value["same_swap"]["macro_f1"],
                "cross_swap_macro_f1": value["cross_swap"]["macro_f1"],
                "dar": value["context"]["dar"], "dar_flip": value["context"]["dar_flip"],
                "prediction_agreement": value["context"]["prediction_agreement"],
                "cross_swap_drop": value["context"]["delta_cross_swap_macro_f1"],
            })
    summary = pd.DataFrame(rows)
    metric_columns = [column for column in summary.columns if column not in {"seed", "variant"}]
    delta_rows: list[dict] = []
    for variant in ("F1", "F2", "F3"):
        for seed in SEEDS:
            baseline = summary.query("seed == @seed and variant == 'F0'").iloc[0]
            candidate = summary.query("seed == @seed and variant == @variant").iloc[0]
            for metric in metric_columns:
                delta_rows.append({"seed": seed, "variant": variant, "metric": metric, "delta_vs_f0": candidate[metric] - baseline[metric]})
    deltas = pd.DataFrame(delta_rows)
    stats = deltas.groupby(["variant", "metric"], as_index=False).agg(mean_delta=("delta_vs_f0", "mean"), sd_delta=("delta_vs_f0", "std"), positive_seeds=("delta_vs_f0", lambda x: int((x > 0).sum())), negative_seeds=("delta_vs_f0", lambda x: int((x < 0).sum())) )
    for path in (Path(args.summary), Path(args.deltas), Path(args.report)): path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False); stats.to_csv(args.deltas, index=False)
    means = summary.groupby("variant")[metric_columns].mean(); stds = summary.groupby("variant")[metric_columns].std(ddof=1)
    def stat(variant: str, metric: str) -> str: return f"{means.loc[variant, metric]:.4f} ± {stds.loc[variant, metric]:.4f}"
    def delta(variant: str, metric: str) -> pd.Series: return stats.query("variant == @variant and metric == @metric").iloc[0]
    f3_cross, f3_flip, f3_clean = delta("F3", "cross_swap_macro_f1"), delta("F3", "dar_flip"), delta("F3", "original_macro_f1")
    report = f"""# CXT-Fish Phase 1C R0 three-seed replication

## Scope

F0–F3 are compared on the frozen validation partition for seeds 3407, 2026, and 17. F0 reuses the corresponding frozen A1 checkpoint. Internal test and outer folds were not accessed.

## Three-seed absolute metrics

| Variant | Original macro F1 | Tail F1 | Track-balanced accuracy | Cross-class swap macro F1 | DAR_flip |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(f"| {v} | {stat(v, 'original_macro_f1')} | {stat(v, 'original_tail_f1')} | {stat(v, 'track_balanced_accuracy')} | {stat(v, 'cross_swap_macro_f1')} | {stat(v, 'dar_flip')} |" for v in VARIANTS) + f"""

## F3 relative to matched F0

- Cross-class swap macro F1: {f3_cross.mean_delta:+.4f} ± {f3_cross.sd_delta:.4f}; positive in {f3_cross.positive_seeds}/3 seeds.
- Conditional donor attraction (DAR_flip): {f3_flip.mean_delta:+.4f} ± {f3_flip.sd_delta:.4f}; lower is favorable, so it is lower in {f3_flip.negative_seeds}/3 seeds.
- Original macro F1: {f3_clean.mean_delta:+.4f} ± {f3_clean.sd_delta:.4f}; negative in {f3_clean.negative_seeds}/3 seeds.

## Interpretation boundary

R0 determines whether the seed-3407 context-robustness signal replicates. It does not select a new method or alter alpha, mu, blur, masks, sampler, architecture, or evaluation. R1 is limited to the predeclared original-domain BatchNorm-only diagnostic for F3.
"""
    Path(args.report).write_text(report, encoding="utf-8")
    print(stats.to_json(orient="records", indent=2))


if __name__ == "__main__": main()
