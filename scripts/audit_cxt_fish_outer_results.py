"""Audit frozen CXT-Fish outer results without training or checkpoint selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]


def tier_map(train: pd.DataFrame, class_ids: list[str]) -> dict[str, str]:
    counts = train.species_id.astype(str).value_counts()
    ordered = sorted(class_ids, key=lambda value: (-int(counts.get(value, 0)), value))
    return {value: tier for tier, group in zip(("head", "mid", "tail"), np.array_split(np.asarray(ordered, dtype=object), 3)) for value in group.tolist()}


def paired_bootstrap(root: Path, class_ids: list[str], tiers: dict[str, str], samples: int = 1000, seed: int = 3407) -> dict[str, tuple[float, float]]:
    def confusion(target: np.ndarray, prediction: np.ndarray, indices: np.ndarray, classes: int) -> np.ndarray:
        matrix = np.zeros((classes, classes), dtype=np.int64)
        np.add.at(matrix, (target[indices], prediction[indices]), 1)
        return matrix

    def macro_vector(matrices: np.ndarray, selected_labels: np.ndarray) -> np.ndarray:
        true_positive = np.diagonal(matrices, axis1=1, axis2=2).astype(float)
        false_positive = matrices.sum(axis=1) - true_positive
        false_negative = matrices.sum(axis=2) - true_positive
        denominator = 2.0 * true_positive + false_positive + false_negative
        values = np.divide(2.0 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0)
        return values[:, selected_labels].mean(axis=1)

    def macro_from_matrix(matrix: np.ndarray, selected_labels: np.ndarray) -> float:
        true_positive = np.diag(matrix).astype(float)
        false_positive = matrix.sum(axis=0) - true_positive
        false_negative = matrix.sum(axis=1) - true_positive
        denominator = 2.0 * true_positive + false_positive + false_negative
        values = np.divide(2.0 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0)
        return float(values[selected_labels].mean())

    labels = np.arange(len(class_ids)); tail_labels = np.asarray([i for i, value in enumerate(class_ids) if tiers[value] == "tail"])
    pairs = []
    for seed_value in (3407, 2026, 17):
        for fold in (1, 2, 3):
            f0 = pd.read_csv(root / f"fold_{fold}" / f"F0_seed{seed_value}" / "per_image.csv")
            f1 = pd.read_csv(root / f"fold_{fold}" / f"F1_seed{seed_value}" / "per_image.csv")
            if not f0[["image_path", "group_id", "target", "cross_donor_target"]].equals(f1[["image_path", "group_id", "target", "cross_donor_target"]]):
                raise RuntimeError("F0/F1 paired rows are not identical")
            groups = list(f0.groupby("group_id").indices.values())
            target0 = f0.target.to_numpy(dtype=int); target1 = f1.target.to_numpy(dtype=int)
            stats = {"groups": groups}
            for view in ("original", "cross_swap"):
                pred0 = f0[view].to_numpy(dtype=int); pred1 = f1[view].to_numpy(dtype=int)
                stats[f"f0_{view}"] = np.stack([confusion(target0, pred0, indices, len(class_ids)) for indices in groups])
                stats[f"f1_{view}"] = np.stack([confusion(target1, pred1, indices, len(class_ids)) for indices in groups])
            f0_correct = f0.original.eq(f0.target).to_numpy(); f1_correct = f1.original.eq(f1.target).to_numpy()
            f0_flip = f0.cross_swap.eq(f0.cross_donor_target).to_numpy(); f1_flip = f1.cross_swap.eq(f1.cross_donor_target).to_numpy()
            stats["f0_dar_num"] = np.asarray([np.logical_and(f0_correct[index], f0_flip[index]).sum() for index in groups], dtype=float)
            stats["f1_dar_num"] = np.asarray([np.logical_and(f1_correct[index], f1_flip[index]).sum() for index in groups], dtype=float)
            stats["f0_dar_den"] = np.asarray([f0_correct[index].sum() for index in groups], dtype=float)
            stats["f1_dar_den"] = np.asarray([f1_correct[index].sum() for index in groups], dtype=float)
            stats["f0_tba"] = np.asarray([f0_correct[index].mean() for index in groups], dtype=float)
            stats["f1_tba"] = np.asarray([f1_correct[index].mean() for index in groups], dtype=float)
            stats["f0_disagreement"] = np.asarray([f0.original.to_numpy()[index].__ne__(f0.cross_swap.to_numpy()[index]).mean() for index in groups], dtype=float)
            stats["f1_disagreement"] = np.asarray([f1.original.to_numpy()[index].__ne__(f1.cross_swap.to_numpy()[index]).mean() for index in groups], dtype=float)
            for view in ("original", "cross_swap"):
                stats[f"f0_{view}_macro"] = macro_vector(stats[f"f0_{view}"], labels)
                stats[f"f1_{view}_macro"] = macro_vector(stats[f"f1_{view}"], labels)
            same_f0 = np.stack([confusion(target0, f0.same_swap.to_numpy(dtype=int), indices, len(class_ids)) for indices in groups])
            same_f1 = np.stack([confusion(target1, f1.same_swap.to_numpy(dtype=int), indices, len(class_ids)) for indices in groups])
            stats["f0_same_swap_macro"] = macro_vector(same_f0, labels)
            stats["f1_same_swap_macro"] = macro_vector(same_f1, labels)
            stats["f0_original_tail"] = macro_vector(stats["f0_original"], tail_labels)
            stats["f1_original_tail"] = macro_vector(stats["f1_original"], tail_labels)
            pairs.append(stats)
    rng = np.random.default_rng(seed)
    names = ("original_macro_f1", "tail_f1", "track_balanced_accuracy", "same_swap_macro_f1", "cross_swap_macro_f1", "dar_flip", "agreement")
    result = {name: np.empty(samples, dtype=float) for name in names}
    for iteration in range(samples):
        delta = {name: [] for name in names}
        for stats in pairs:
            selected = rng.integers(0, len(stats["groups"]), size=len(stats["groups"]))
            weights = np.bincount(selected, minlength=len(stats["groups"])).astype(np.float64)
            for name, view in (("original_macro_f1", "original"), ("cross_swap_macro_f1", "cross_swap")):
                f0_matrix = weights.dot(stats[f"f0_{view}"].reshape(len(stats["groups"]), -1)).reshape(len(class_ids), len(class_ids))
                f1_matrix = weights.dot(stats[f"f1_{view}"].reshape(len(stats["groups"]), -1)).reshape(len(class_ids), len(class_ids))
                delta[name].append(macro_from_matrix(f1_matrix, labels) - macro_from_matrix(f0_matrix, labels))
            delta["same_swap_macro_f1"].append(float((stats["f1_same_swap_macro"][selected] - stats["f0_same_swap_macro"][selected]).mean()))
            f0_matrix = weights.dot(stats["f0_original"].reshape(len(stats["groups"]), -1)).reshape(len(class_ids), len(class_ids))
            f1_matrix = weights.dot(stats["f1_original"].reshape(len(stats["groups"]), -1)).reshape(len(class_ids), len(class_ids))
            delta["tail_f1"].append(macro_from_matrix(f1_matrix, tail_labels) - macro_from_matrix(f0_matrix, tail_labels))
            f0_dar_den = stats["f0_dar_den"][selected].sum(); f1_dar_den = stats["f1_dar_den"][selected].sum()
            delta["dar_flip"].append((stats["f1_dar_num"][selected].sum() / f1_dar_den if f1_dar_den else 0.0) - (stats["f0_dar_num"][selected].sum() / f0_dar_den if f0_dar_den else 0.0))
            delta["track_balanced_accuracy"].append(stats["f1_tba"][selected].mean() - stats["f0_tba"][selected].mean())
            delta["agreement"].append(stats["f1_disagreement"][selected].mean() - stats["f0_disagreement"][selected].mean())
        for name in names:
            result[name][iteration] = float(np.mean(delta[name]))
    return {name: (float(np.quantile(values, .025)), float(np.quantile(values, .975))) for name, values in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", default="outputs/cxt_fish/final_outer_evaluation")
    args = parser.parse_args()
    eval_root = ROOT / args.evaluation_root
    class_ids = [str(value) for value in json.loads((ROOT / "splits/f4k16t_class_ids.json").read_text(encoding="utf-8"))["class_ids"]]
    split = pd.read_csv(ROOT / "splits/f4k16t_track_level_dev.csv")
    train = split[split.split == "train"].copy()
    tiers = tier_map(train, class_ids)
    tail = [value for value in class_ids if tiers[value] == "tail"]

    metadata = pd.read_csv(ROOT / "outputs/gate0/audit/f4k_metadata.csv")
    all_counts = metadata.assign(species_id=metadata.species_id.astype(str)).groupby("species_id").agg(images=("image_path", "size"), tracks=("group_id", "nunique")).reset_index()
    frozen_counts = split.assign(species_id=split.species_id.astype(str)).groupby(["split", "species_id"]).agg(images=("image_path", "size"), tracks=("group_id", "nunique")).reset_index()
    all_counts["dataset_scope"] = "gate0_metadata"
    all_counts["split"] = "all"
    all_counts["frozen_16"] = all_counts.species_id.isin(class_ids)
    frozen_counts["dataset_scope"] = "frozen_16_split"
    frozen_counts["frozen_16"] = True
    class_audit = pd.concat([all_counts, frozen_counts], ignore_index=True, sort=False)
    class_audit.to_csv(ROOT / "experiments/cxt_fish_outer_class_track_audit.csv", index=False)

    rows = []
    per_class_rows = []
    for seed_value in (3407, 2026, 17):
        for fold in (1, 2, 3):
            for method in ("F0", "F1"):
                path = eval_root / f"fold_{fold}" / f"{method}_seed{seed_value}" / "per_image.csv"
                frame = pd.read_csv(path)
                target = frame.target.to_numpy()
                for view in ("original", "foreground", "same_swap", "cross_swap"):
                    f1 = f1_score(target, frame[view].to_numpy(), labels=np.arange(len(class_ids)), average=None, zero_division=0)
                    for index, value in enumerate(f1):
                        per_class_rows.append({"seed": seed_value, "fold": fold, "method": method, "view": view, "species_id": class_ids[index], "f1": float(value), "tier": tiers[class_ids[index]]})
                original = f1_score(target, frame.original.to_numpy(), labels=np.arange(len(class_ids)), average="macro", zero_division=0)
                rows.append({"seed": seed_value, "fold": fold, "method": method, "original_macro_f1": float(original)})
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(ROOT / "experiments/cxt_fish_outer_per_class.csv", index=False)
    fold_audit = pd.DataFrame(rows).groupby(["fold", "method"], as_index=False).agg(original_macro_f1_mean=("original_macro_f1", "mean"), original_macro_f1_sd=("original_macro_f1", "std"))
    fold_audit.to_csv(ROOT / "experiments/cxt_fish_outer_fold_audit.csv", index=False)

    paired = paired_bootstrap(eval_root, class_ids, tiers)
    report = [
        "# CXT-Fish outer result audit",
        "",
        "This is a statistical and class-coverage audit of the frozen outer confirmation. It does not retrain models or alter any checkpoint/result.",
        "",
        f"- Frozen class count: {len(class_ids)}",
        f"- Gate-0 metadata class count: {metadata.species_id.astype(str).nunique()}",
        f"- Tail classes (fixed by descending frozen-train image count): {', '.join(tail)}",
        "- Bootstrap: 1,000 paired resamples of trajectory groups, averaged across 9 fold-seed pairs",
        "- Official TEST accessed: false",
        "",
        "## Paired F1 - F0 bootstrap intervals",
        "",
        "| Metric | 95% CI |",
        "|---|---:|",
    ]
    labels = {"original_macro_f1": "Original macro-F1", "tail_f1": "Tail-F1", "track_balanced_accuracy": "Track-balanced accuracy", "same_swap_macro_f1": "Same-class swap macro-F1", "cross_swap_macro_f1": "Cross-class swap macro-F1", "dar_flip": "DAR-flip", "agreement": "Swap disagreement"}
    for key, label in labels.items():
        if key in paired:
            report.append(f"| {label} | [{paired[key][0]:+.4f}, {paired[key][1]:+.4f}] |")
        else:
            report.append(f"| {label} | not computed |")
    report += [
        "",
        "## Fold heterogeneity",
        "",
        "The fold-level table is stored in `experiments/cxt_fish_outer_fold_audit.csv`. Fold 3 is retained as a pre-specified heterogeneity audit, not removed or reweighted.",
        "",
        "## Artifacts",
        "",
        "- `experiments/cxt_fish_outer_class_track_audit.csv`: 23-class Gate-0 metadata versus frozen 16-class split image/track counts.",
        "- `experiments/cxt_fish_outer_per_class.csv`: per-class F1 for every method, seed, fold and view.",
        "- `experiments/cxt_fish_outer_fold_audit.csv`: fold-level clean macro-F1 summary.",
    ]
    (ROOT / "reports/cxt_fish_outer_result_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    figures = ROOT / "reports/figures"
    figures.mkdir(parents=True, exist_ok=True)
    wide = per_class[per_class.view.isin(["original", "cross_swap"])].pivot_table(index=["seed", "fold", "species_id", "tier"], columns=["method", "view"], values="f1").reset_index()
    if ("F0", "cross_swap") in wide.columns and ("F1", "cross_swap") in wide.columns:
        wide["cross_delta"] = wide[("F1", "cross_swap")] - wide[("F0", "cross_swap")]
        plot = wide.groupby("species_id", as_index=False).cross_delta.mean().sort_values("cross_delta")
        plt.figure(figsize=(8, 4)); plt.bar(plot.species_id, plot.cross_delta); plt.axhline(0, color="black", linewidth=.8); plt.ylabel("F1 - F0 cross-swap F1"); plt.xlabel("species_id"); plt.tight_layout(); plt.savefig(figures / "cxt_fish_outer_per_class_cross_swap_delta.png", dpi=160); plt.close()
    cell = pd.DataFrame(rows).pivot(index=["seed", "fold"], columns="method", values="original_macro_f1").reset_index(); cell["delta"] = cell["F1"] - cell["F0"]
    plt.figure(figsize=(7, 4)); plt.bar(np.arange(len(cell)), cell.delta); plt.axhline(0, color="black", linewidth=.8); plt.ylabel("F1 - F0 original macro-F1"); plt.xlabel("fold-seed pair"); plt.tight_layout(); plt.savefig(figures / "cxt_fish_outer_paired_clean_delta.png", dpi=160); plt.close()
    print(json.dumps({"status": "complete", "tail_classes": tail, "class_rows": len(class_audit), "per_class_rows": len(per_class), "bootstrap": paired}, indent=2))


if __name__ == "__main__":
    main()
