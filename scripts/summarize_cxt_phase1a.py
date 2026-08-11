"""Summarize the fixed Phase 1A validation-only screening matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


RUNS = [
    ("A0", "A0_s0_ce_seed407", "S0", "CE", "none"),
    ("A1", "A1_s1_ce_seed407", "S1", "CE", "none"),
    ("A2", "A2_s2_ce_seed407", "S2", "CE", "none"),
    ("A3", "A3_s3_ce_seed407", "S3", "CE", "none"),
    ("A4-I", "A4I_s0_balanced_image_seed407", "S0", "Balanced Softmax", "image-count"),
    ("A4-T", "A4T_s0_balanced_track_seed407", "S0", "Balanced Softmax", "track-count"),
    ("A5-I", "A5I_s1_balanced_image_seed407", "S1", "Balanced Softmax", "image-count"),
    ("A5-T", "A5T_s1_balanced_track_seed407", "S1", "Balanced Softmax", "track-count"),
]


def tier_map(track_split: pd.DataFrame) -> dict[str, str]:
    counts = track_split.loc[track_split["split"] == "train", "species_id"].astype(str).value_counts()
    ordered = sorted(counts.index.tolist(), key=lambda species: (-int(counts[species]), species))
    groups = np.array_split(np.array(ordered, dtype=object), 3)
    return {species: tier for tier, group in zip(("head", "mid", "tail"), groups) for species in group.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet18_gate0.yaml")
    parser.add_argument("--root", default="outputs/cxt_fish/phase1a_screen")
    parser.add_argument("--summary", default="experiments/cxt_fish_phase1a_screening_summary.csv")
    parser.add_argument("--per-class", default="experiments/cxt_fish_phase1a_screening_per_class.csv")
    parser.add_argument("--selection", default="experiments/cxt_fish_phase1a_selection.json")
    parser.add_argument("--report", default="reports/cxt_fish_phase1a_screening_report.md")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tiers = tier_map(pd.read_csv(cfg["track_split_path"]))
    root = Path(args.root)
    rows: list[dict] = []
    per_class_rows: list[dict] = []
    for identifier, directory, sampler, objective, prior in RUNS:
        path = root / directory / "evaluation_phase1a_val" / "metrics_val.json"
        if not path.exists():
            raise FileNotFoundError(path)
        metric = json.loads(path.read_text(encoding="utf-8"))
        f1_by_species = dict(zip(map(str, metric["class_ids"]), metric["per_class_f1"]))
        tier_f1 = {tier: float(np.mean([f1_by_species[s] for s, value in tiers.items() if value == tier])) for tier in ("head", "mid", "tail")}
        rows.append({
            "id": identifier, "run": directory, "sampler": sampler, "objective": objective, "prior": prior,
            "macro_f1": metric["macro_f1"], "balanced_accuracy": metric["balanced_accuracy"],
            "track_balanced_accuracy": metric["track_balanced_accuracy"], "tail_f1": tier_f1["tail"],
            "mid_f1": tier_f1["mid"], "head_f1": tier_f1["head"],
        })
        for species, f1, support in zip(metric["class_ids"], metric["per_class_f1"], metric["per_class_support"]):
            per_class_rows.append({"id": identifier, "run": directory, "species_id": str(species), "tier": tiers[str(species)], "f1": f1, "val_support": support})

    summary = pd.DataFrame(rows)
    # Fixed lexicographic selection rule; exact ties prefer simpler sampler/objective.
    sampler_complexity = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}
    objective_complexity = {"CE": 0, "Balanced Softmax": 1}
    summary["_sampler_complexity"] = summary.sampler.map(sampler_complexity)
    summary["_objective_complexity"] = summary.objective.map(objective_complexity)
    ranked = summary.sort_values(
        ["tail_f1", "track_balanced_accuracy", "macro_f1", "_sampler_complexity", "_objective_complexity", "id"],
        ascending=[False, False, False, True, True, True], kind="stable"
    ).reset_index(drop=True)
    ranked.insert(0, "validation_rank", np.arange(1, len(ranked) + 1))
    ranked.drop(columns=["_sampler_complexity", "_objective_complexity"], inplace=True)

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.summary, index=False)
    pd.DataFrame(per_class_rows).to_csv(args.per_class, index=False)
    selected = ranked.head(3)[["validation_rank", "id", "run", "sampler", "objective", "prior", "tail_f1", "track_balanced_accuracy", "macro_f1"]].to_dict(orient="records")
    replication_seeds = [int(seed) for seed in cfg["seeds"]]
    Path(args.selection).write_text(json.dumps({
        "protocol": "CXT-Fish Phase 1A validation-only screening",
        "selection_order": ["tail_f1", "track_balanced_accuracy", "macro_f1", "simpler_configuration_on_exact_tie"],
        "selected_for_three_seed_replication": selected,
        "replication_seeds": replication_seeds,
        "internal_test_accessed": False,
        "outer_folds_accessed": False,
    }, indent=2), encoding="utf-8")
    table = "\n".join(
        f"| {row.validation_rank} | {row.id} | {row.sampler} | {row.objective} | {row.prior} | "
        f"{row.tail_f1:.4f} | {row.track_balanced_accuracy:.4f} | {row.macro_f1:.4f} |"
        for row in ranked.itertuples(index=False)
    )
    chosen = ", ".join(item["id"] for item in selected)
    Path(args.report).write_text(
        "# CXT-Fish Phase 1A Screening\n\n"
        "## Scope\n\n"
        "All eight frozen sampling/loss configurations were trained with seed 407 and evaluated on the fixed track-level validation partition only. "
        "The internal test and outer folds were not accessed.\n\n"
        "## Fixed validation ranking\n\n"
        "Ranking is lexicographic: tail F1, then track-balanced accuracy, then macro F1; an exact tie prefers the simpler configuration.\n\n"
        "| Rank | ID | Sampler | Objective | Prior | Tail F1 | Track-balanced accuracy | Macro F1 |\n"
        "|---:|---|---|---|---|---:|---:|---:|\n"
        f"{table}\n\n"
        "## Three-seed replication selection\n\n"
        f"The top three configurations are **{chosen}**. Each will be retrained with the three frozen replication seeds {replication_seeds} before any internal-test access.\n",
        encoding="utf-8",
    )
    print(json.dumps({"completed_runs": len(ranked), "selected": selected, "internal_test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
