"""Freeze the validation-only CXT-Fish Phase 1C seed-3407 pilot summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import yaml


LOCATIONS = {
    "F0": "outputs/cxt_fish/phase1a_replication/A1_s1_ce_seed3407/evaluation_phase1c_f0_seed3407/metrics_val.json",
    "F1": "outputs/cxt_fish/phase1c_pilot/F1_seed3407/evaluation_phase1c_f1_seed3407/metrics_val.json",
    "F2": "outputs/cxt_fish/phase1c_pilot/F2_seed3407/evaluation_phase1c_f2_seed3407/metrics_val.json",
    "F3": "outputs/cxt_fish/phase1c_pilot/F3_seed3407/evaluation_phase1c_f3_seed3407/metrics_val.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_phase1c.yaml")
    parser.add_argument("--summary", default="experiments/cxt_fish_phase1c_single_seed_summary.csv")
    parser.add_argument("--report", default="reports/cxt_fish_phase1c_single_seed.md")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    split = pd.read_csv(cfg["track_split_path"])
    counts = split.loc[split.split == "train", "species_id"].astype(str).value_counts()
    ordered = sorted(counts.index, key=lambda value: (-int(counts[value]), value))
    tier_map = {species: tier for tier, group in zip(("head", "mid", "tail"), np.array_split(np.asarray(ordered, dtype=object), 3)) for species in group}
    rows: list[dict] = []
    for variant, location in LOCATIONS.items():
        value = json.loads(Path(location).read_text(encoding="utf-8"))
        if value["partition"] != "val" or value["internal_test_accessed"] or value["outer_folds_accessed"]:
            raise ValueError(f"invalid Phase 1C provenance: {location}")
        class_f1 = dict(zip(map(str, value["original"]["class_ids"]), value["original"]["per_class_f1"]))
        cross_f1 = dict(zip(map(str, value["cross_swap"]["class_ids"]), value["cross_swap"]["per_class_f1"]))
        rows.append({
            "variant": variant,
            "original_macro_f1": value["original"]["macro_f1"],
            "foreground_macro_f1": value["foreground"]["macro_f1"],
            "same_swap_macro_f1": value["same_swap"]["macro_f1"],
            "cross_swap_macro_f1": value["cross_swap"]["macro_f1"],
            "original_tail_f1": float(np.mean([class_f1[label] for label, tier in tier_map.items() if tier == "tail"])),
            "cross_swap_tail_f1": float(np.mean([cross_f1[label] for label, tier in tier_map.items() if tier == "tail"])),
            "track_balanced_accuracy": value["original"]["track_balanced_accuracy"],
            "prediction_agreement": value["context"]["prediction_agreement"],
            "dar": value["context"]["dar"],
            "dar_flip": value["context"]["dar_flip"],
            "delta_foreground": value["context"]["delta_foreground_macro_f1"],
            "delta_same_swap": value["context"]["delta_same_swap_macro_f1"],
            "delta_cross_swap": value["context"]["delta_cross_swap_macro_f1"],
        })
    frame = pd.DataFrame(rows).set_index("variant")
    for candidate in ("F1", "F2", "F3"):
        for column in ("original_macro_f1", "foreground_macro_f1", "same_swap_macro_f1", "cross_swap_macro_f1", "dar", "dar_flip", "prediction_agreement"):
            frame.loc[candidate, f"delta_vs_f0_{column}"] = frame.loc[candidate, column] - frame.loc["F0", column]
    for path in (Path(args.summary), Path(args.report)): path.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_csv(args.summary, index=False)
    f0, f1, f2, f3 = (frame.loc[name] for name in ("F0", "F1", "F2", "F3"))
    report = f"""# CXT-Fish Phase 1C single-seed pilot

## Scope

This report compares F0–F3 at registered seed 3407 on the frozen track-level validation partition only. F0 reuses the frozen A1 checkpoint; F1/F2/F3 are sequential new runs. Internal test and outer folds were not accessed.

## Context metrics

| Variant | Original macro F1 | Foreground macro F1 | Same-class swap F1 | Cross-class swap F1 | DAR_flip |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(f"| {name} | {row.original_macro_f1:.4f} | {row.foreground_macro_f1:.4f} | {row.same_swap_macro_f1:.4f} | {row.cross_swap_macro_f1:.4f} | {row.dar_flip:.4f} |" for name, row in frame.iterrows()) + f"""

## Fixed comparisons

- F1−F0: cross-class swap macro F1 {f1.cross_swap_macro_f1 - f0.cross_swap_macro_f1:+.4f}; DAR_flip {f1.dar_flip - f0.dar_flip:+.4f}; original macro F1 {f1.original_macro_f1 - f0.original_macro_f1:+.4f}.
- F2−F0: cross-class swap macro F1 {f2.cross_swap_macro_f1 - f0.cross_swap_macro_f1:+.4f}; DAR_flip {f2.dar_flip - f0.dar_flip:+.4f}; original macro F1 {f2.original_macro_f1 - f0.original_macro_f1:+.4f}.
- F3−F1: cross-class swap macro F1 {f3.cross_swap_macro_f1 - f1.cross_swap_macro_f1:+.4f}; DAR_flip {f3.dar_flip - f1.dar_flip:+.4f}; original macro F1 {f3.original_macro_f1 - f1.original_macro_f1:+.4f}.

## Interpretation boundary

F3 provides a positive single-seed robustness signal: lower donor attraction and higher cross-class-swap F1 than F0/F1. It also has lower original-image macro F1 than F0/F1. This pilot therefore supports only a decision about whether to run the two registered replication seeds; it is not a final method claim, and it must not be used to change foreground construction, loss weights, sampling, or evaluation rules.
"""
    Path(args.report).write_text(report, encoding="utf-8")
    print(frame.reset_index().to_json(orient="records", indent=2))


if __name__ == "__main__": main()
