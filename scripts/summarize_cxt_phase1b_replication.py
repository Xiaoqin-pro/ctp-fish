"""Freeze the three-seed CXT-Fish Phase 1B validation replication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


SEEDS = (3407, 2026, 17)
VARIANTS = ("C0", "C1", "C2")


def tiers(split: pd.DataFrame) -> dict[str, str]:
    counts = split.loc[split.split == "train", "species_id"].astype(str).value_counts()
    ordered = sorted(counts.index, key=lambda value: (-int(counts[value]), value))
    return {
        species: tier
        for tier, group in zip(("head", "mid", "tail"), np.array_split(np.asarray(ordered, dtype=object), 3))
        for species in group
    }


def run_root(project_root: Path, seed: int) -> Path:
    name = "phase1b_pilot" if seed == 3407 else "phase1b_replication"
    return project_root / "outputs" / "cxt_fish" / name


def load_metric(project_root: Path, seed: int, variant: str) -> tuple[dict, pd.DataFrame]:
    directory = run_root(project_root, seed) / f"{variant}_seed{seed}" / "evaluation_phase1b_val"
    metric = json.loads((directory / "metrics_val.json").read_text(encoding="utf-8"))
    tracks = pd.read_csv(directory / "per_track_val.csv").rename(columns={"track_accuracy": variant})
    if metric["partition"] != "val" or metric["internal_test_accessed"] or metric["outer_folds_accessed"]:
        raise ValueError(f"invalid evaluation provenance: {directory}")
    return metric, tracks


def aggregate_rows(project_root: Path, tier_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    class_rows: list[dict] = []
    track_rows: list[pd.DataFrame] = []
    for seed in SEEDS:
        tracks: dict[str, pd.DataFrame] = {}
        for variant in VARIANTS:
            metric, tracks[variant] = load_metric(project_root, seed, variant)
            f1 = dict(zip(map(str, metric["class_ids"]), metric["per_class_f1"]))
            row = {
                "seed": seed,
                "variant": variant,
                "macro_f1": metric["macro_f1"],
                "balanced_accuracy": metric["balanced_accuracy"],
                "track_balanced_accuracy": metric["track_balanced_accuracy"],
            }
            for tier in ("head", "mid", "tail"):
                row[f"{tier}_f1"] = float(np.mean([f1[s] for s, label in tier_map.items() if label == tier]))
            summary_rows.append(row)
            for species, value, support in zip(metric["class_ids"], metric["per_class_f1"], metric["per_class_support"]):
                class_rows.append({"seed": seed, "variant": variant, "species_id": str(species), "tier": tier_map[str(species)], "f1": value, "val_support": support})
        combined = tracks["C0"].merge(tracks["C1"], on="group_id", validate="one_to_one").merge(tracks["C2"], on="group_id", validate="one_to_one")
        combined["seed"] = seed
        combined["c1_minus_c0"] = combined.C1 - combined.C0
        combined["c2_minus_c1"] = combined.C2 - combined.C1
        track_rows.append(combined)
    summary = pd.DataFrame(summary_rows)
    classes = pd.DataFrame(class_rows).pivot(index=["seed", "species_id", "tier", "val_support"], columns="variant", values="f1").reset_index()
    classes["c1_minus_c0"] = classes.C1 - classes.C0
    classes["c2_minus_c1"] = classes.C2 - classes.C1
    return summary, classes, pd.concat(track_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_phase1b.yaml")
    parser.add_argument("--summary", default="experiments/cxt_fish_phase1b_replication_summary.csv")
    parser.add_argument("--per-class", default="experiments/cxt_fish_phase1b_replication_per_class_deltas.csv")
    parser.add_argument("--per-track", default="experiments/cxt_fish_phase1b_replication_per_track_deltas.csv")
    parser.add_argument("--report", default="reports/cxt_fish_phase1b_replication_report.md")
    args = parser.parse_args()
    project_root = Path.cwd()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary, classes, tracks = aggregate_rows(project_root, tiers(pd.read_csv(cfg["track_split_path"])))
    wide = summary.pivot(index="seed", columns="variant", values=["macro_f1", "track_balanced_accuracy", "tail_f1"]).sort_index()
    comparisons: list[dict] = []
    for left, right, label in (("C1", "C0", "c1_minus_c0"), ("C2", "C1", "c2_minus_c1")):
        for metric in ("macro_f1", "track_balanced_accuracy", "tail_f1"):
            delta = wide[metric][left] - wide[metric][right]
            comparisons.append({"comparison": label, "metric": metric, "mean_delta": delta.mean(), "std_delta": delta.std(ddof=1), "positive_seeds": int((delta > 0).sum()), "seeds": len(delta)})
    comparison_frame = pd.DataFrame(comparisons)
    c2_tail = comparison_frame.query("comparison == 'c2_minus_c1' and metric == 'tail_f1'").iloc[0]
    c2_track = comparison_frame.query("comparison == 'c2_minus_c1' and metric == 'track_balanced_accuracy'").iloc[0]
    c2_macro = comparison_frame.query("comparison == 'c2_minus_c1' and metric == 'macro_f1'").iloc[0]
    stable = bool((c2_tail.positive_seeds >= 2 or c2_track.positive_seeds >= 2) and c2_macro.mean_delta >= -0.005)
    decision = "retain for Phase 1C base" if stable else "do not retain cross-track contrast as an independent claim"
    for path in (args.summary, args.per_class, args.per_track, args.report):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    classes.to_csv(args.per_class, index=False)
    tracks.to_csv(args.per_track, index=False)
    report = ["# CXT-Fish Phase 1B three-seed replication", "", "## Scope", "", "C0/C1/C2 were evaluated only on the frozen track-level validation partition for seeds 3407, 2026, and 17. Internal test and outer folds were not accessed.", "", "## Per-seed validation metrics", "", "| Seed | Variant | Macro F1 | Tail F1 | Track-balanced accuracy |", "|---:|---|---:|---:|---:|"]
    report += [f"| {r.seed} | {r.variant} | {r.macro_f1:.4f} | {r.tail_f1:.4f} | {r.track_balanced_accuracy:.4f} |" for r in summary.sort_values(["seed", "variant"]).itertuples(index=False)]
    report += ["", "## Paired mechanism comparisons", "", "| Comparison | Metric | Mean delta | SD | Positive seeds |", "|---|---|---:|---:|---:|"]
    report += [f"| {r.comparison} | {r.metric} | {r.mean_delta:+.4f} | {r.std_delta:.4f} | {r.positive_seeds}/{r.seeds} |" for r in comparison_frame.itertuples(index=False)]
    report += ["", "## Frozen decision", "", f"- Decision: **{decision}**.", "- The core C2-vs-C1 comparison is interpreted from direction consistency across the three registered seeds; it is not an internal-test or outer-fold result.", "- C0 remains the structured-batch control. A1 is an earlier natural training baseline and is not treated as a one-variable comparison with C0.", "- No context-control module, prototype loss, memory bank, new backbone, internal test, or outer-fold evaluation was used in this phase."]
    Path(args.report).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(comparison_frame.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
