"""Summarize the frozen C0/C1/C2 seed-3407 validation pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


RUNS = {"C0": "C0_seed3407", "C1": "C1_seed3407", "C2": "C2_seed3407"}


def _tiers(split: pd.DataFrame) -> dict[str, str]:
    counts = split.loc[split.split == "train", "species_id"].astype(str).value_counts()
    ordered = sorted(counts.index, key=lambda value: (-int(counts[value]), value))
    return {species: tier for tier, group in zip(("head", "mid", "tail"), np.array_split(np.asarray(ordered, dtype=object), 3)) for species in group}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_phase1b.yaml")
    parser.add_argument("--root", default="outputs/cxt_fish/phase1b_pilot")
    parser.add_argument("--summary", default="experiments/cxt_fish_phase1b_pilot_summary.csv")
    parser.add_argument("--per-class", default="experiments/cxt_fish_phase1b_pilot_per_class_deltas.csv")
    parser.add_argument("--per-track", default="experiments/cxt_fish_phase1b_pilot_per_track_deltas.csv")
    parser.add_argument("--report", default="reports/cxt_fish_phase1b_pilot_report.md")
    args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text())
    tiers = _tiers(pd.read_csv(cfg["track_split_path"]))
    metrics: dict[str, dict] = {}; per_track: dict[str, pd.DataFrame] = {}
    rows = []; per_class_rows = []
    for identifier, run in RUNS.items():
        directory = Path(args.root) / run / "evaluation_phase1b_val"
        metric = json.loads((directory / "metrics_val.json").read_text())
        metrics[identifier] = metric; per_track[identifier] = pd.read_csv(directory / "per_track_val.csv").rename(columns={"track_accuracy": identifier})
        f1 = dict(zip(map(str, metric["class_ids"]), metric["per_class_f1"]))
        tier_values = {tier: float(np.mean([f1[species] for species, label in tiers.items() if label == tier])) for tier in ("head", "mid", "tail")}
        rows.append({"id": identifier, "macro_f1": metric["macro_f1"], "balanced_accuracy": metric["balanced_accuracy"], "track_balanced_accuracy": metric["track_balanced_accuracy"], **{f"{tier}_f1": value for tier, value in tier_values.items()}})
        for species, value, support in zip(metric["class_ids"], metric["per_class_f1"], metric["per_class_support"]):
            per_class_rows.append({"id": identifier, "species_id": str(species), "tier": tiers[str(species)], "f1": value, "val_support": support})
    summary = pd.DataFrame(rows)
    for compared in ("C1", "C2"):
        for column in ("macro_f1", "balanced_accuracy", "track_balanced_accuracy", "head_f1", "mid_f1", "tail_f1"):
            summary.loc[summary.id == compared, f"delta_vs_{'C0' if compared == 'C1' else 'C1'}_{column}"] = float(summary.loc[summary.id == compared, column].iloc[0] - summary.loc[summary.id == ('C0' if compared == 'C1' else 'C1'), column].iloc[0])
    classes = pd.DataFrame(per_class_rows).pivot(index=["species_id", "tier", "val_support"], columns="id", values="f1").reset_index()
    classes["c1_minus_c0"] = classes.C1 - classes.C0; classes["c2_minus_c1"] = classes.C2 - classes.C1
    tracks = per_track["C0"].merge(per_track["C1"], on="group_id", validate="one_to_one").merge(per_track["C2"], on="group_id", validate="one_to_one")
    tracks["c1_minus_c0"] = tracks.C1 - tracks.C0; tracks["c2_minus_c1"] = tracks.C2 - tracks.C1
    for path in (args.summary, args.per_class, args.per_track, args.report): Path(path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False); classes.to_csv(args.per_class, index=False); tracks.to_csv(args.per_track, index=False)
    c2 = summary.set_index("id").loc["C2"]; c1 = summary.set_index("id").loc["C1"]
    c2_positive_classes = int((classes.c2_minus_c1 > 0).sum()); c2_positive_tracks = int((tracks.c2_minus_c1 > 0).sum())
    report = f"""# CXT-Fish Phase 1B single-seed pilot

## Scope

This report compares C0/C1/C2 with seed 3407 on the fixed track-level validation partition only. Internal test and outer folds were not accessed.

## Metrics

| ID | Macro F1 | Tail F1 | Track-balanced accuracy |
|---|---:|---:|---:|
""" + "\n".join(f"| {row.id} | {row.macro_f1:.4f} | {row.tail_f1:.4f} | {row.track_balanced_accuracy:.4f} |" for row in summary.itertuples(index=False)) + f"""

## Mechanism comparisons

- C1 - C0: tail F1 {(c1.tail_f1 - summary.set_index('id').loc['C0'].tail_f1):+.4f}; track-balanced accuracy {(c1.track_balanced_accuracy - summary.set_index('id').loc['C0'].track_balanced_accuracy):+.4f}; macro F1 {(c1.macro_f1 - summary.set_index('id').loc['C0'].macro_f1):+.4f}.
- C2 - C1: tail F1 {(c2.tail_f1 - c1.tail_f1):+.4f}; track-balanced accuracy {(c2.track_balanced_accuracy - c1.track_balanced_accuracy):+.4f}; macro F1 {(c2.macro_f1 - c1.macro_f1):+.4f}.
- C2 exceeds C1 on {c2_positive_classes}/{len(classes)} classes and {c2_positive_tracks}/{len(tracks)} validation tracks.

This is a single-seed development result. It determines whether reserved seeds should be run; it is not an internal-test or outer-fold claim.
"""
    Path(args.report).write_text(report, encoding="utf-8")
    print(summary.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
