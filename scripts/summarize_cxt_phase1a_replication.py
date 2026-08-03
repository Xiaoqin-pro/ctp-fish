"""Aggregate fixed three-seed Phase 1A validation replications only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


CONFIGS = {
    "A1": ("A1_s1_ce", "S1", "CE", "none"),
    "A4-I": ("A4I_s0_balanced_image", "S0", "Balanced Softmax", "image-count"),
    "A2": ("A2_s2_ce", "S2", "CE", "none"),
}


def build_tier_map(split: pd.DataFrame) -> dict[str, str]:
    counts = split.loc[split["split"] == "train", "species_id"].astype(str).value_counts()
    ordered = sorted(counts.index.tolist(), key=lambda item: (-int(counts[item]), item))
    groups = np.array_split(np.array(ordered, dtype=object), 3)
    return {species: tier for tier, group in zip(("head", "mid", "tail"), groups) for species in group.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet18_gate0.yaml")
    parser.add_argument("--root", default="outputs/cxt_fish/phase1a_replication")
    parser.add_argument("--output", default="experiments/cxt_fish_phase1a_replication_summary.csv")
    parser.add_argument("--report", default="reports/cxt_fish_phase1a_replication_report.md")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seeds = [int(value) for value in cfg["seeds"]]
    tiers = build_tier_map(pd.read_csv(cfg["track_split_path"]))
    rows = []
    for identifier, (prefix, sampler, objective, prior) in CONFIGS.items():
        for seed in seeds:
            path = Path(args.root) / f"{prefix}_seed{seed}" / "evaluation_phase1a_replication_val" / "metrics_val.json"
            if not path.exists(): raise FileNotFoundError(path)
            metric = json.loads(path.read_text(encoding="utf-8"))
            f1 = dict(zip(map(str, metric["class_ids"]), metric["per_class_f1"]))
            rows.append({
                "id": identifier, "seed": seed, "sampler": sampler, "objective": objective, "prior": prior,
                "macro_f1": metric["macro_f1"], "balanced_accuracy": metric["balanced_accuracy"],
                "track_balanced_accuracy": metric["track_balanced_accuracy"],
                "tail_f1": float(np.mean([f1[species] for species, tier in tiers.items() if tier == "tail"])),
            })
    detailed = pd.DataFrame(rows)
    aggregate = detailed.groupby(["id", "sampler", "objective", "prior"], as_index=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        tail_f1_mean=("tail_f1", "mean"), tail_f1_sd=("tail_f1", "std"),
        track_balanced_accuracy_mean=("track_balanced_accuracy", "mean"),
        track_balanced_accuracy_sd=("track_balanced_accuracy", "std"),
    ).sort_values(["tail_f1_mean", "track_balanced_accuracy_mean", "macro_f1_mean"], ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(Path(args.output).with_name("cxt_fish_phase1a_replication_per_seed.csv"), index=False)
    aggregate.to_csv(args.output, index=False)
    lines = ["# CXT-Fish Phase 1A Three-Seed Replication", "", "## Scope", "", "The three validation-selected configurations were independently retrained with frozen seeds 3407, 2026, and 17. Results below use only the fixed track-level validation partition; no internal test or outer fold was accessed.", "", "## Aggregate validation results", "", "| ID | Sampler | Objective | Prior | Tail F1 | Track-balanced accuracy | Macro F1 |", "|---|---|---|---|---:|---:|---:|"]
    for row in aggregate.itertuples(index=False):
        lines.append(f"| {row.id} | {row.sampler} | {row.objective} | {row.prior} | {row.tail_f1_mean:.4f} ± {row.tail_f1_sd:.4f} | {row.track_balanced_accuracy_mean:.4f} ± {row.track_balanced_accuracy_sd:.4f} | {row.macro_f1_mean:.4f} ± {row.macro_f1_sd:.4f} |")
    lines += ["", "These are development results. The internal test remains locked pending method-component freezing."]
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(aggregate.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
