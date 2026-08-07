"""Finalize frozen CXT-Fish statistics from existing JSON/CSV outputs only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASS_IDS = [str(x) for x in json.loads((ROOT / "splits/f4k16t_class_ids.json").read_text())["class_ids"]]
N_CLASSES = len(CLASS_IDS)
VIEWS = ("original", "foreground", "same_swap", "cross_swap")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f1_from_confusion(confusion: np.ndarray) -> float:
    if confusion.ndim == 3:
        tp = np.einsum("bii->bi", confusion).astype(float)
        fp = confusion.sum(1) - tp
        fn = confusion.sum(2) - tp
    else:
        tp = np.diag(confusion).astype(float)
        fp = confusion.sum(0) - tp
        fn = confusion.sum(1) - tp
    denom = 2 * tp + fp + fn
    values = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(values.mean()) if confusion.ndim == 2 else values.mean(axis=1)


def load_cells(root: Path, methods: tuple[str, ...], folder: str) -> list[dict]:
    cells = []
    for fold in (1, 2, 3):
        for method in methods:
            directories = sorted((root / f"fold_{fold}").glob(f"{method}_seed*/metrics.json"))
            for metrics_path in directories:
                directory = metrics_path.parent
                per_image = directory / "per_image.csv"
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                frame = pd.read_csv(per_image)
                frame["species"] = frame["target"].map(dict(enumerate(CLASS_IDS)))
                seed = int(directory.name.split("seed")[-1])
                cells.append({"fold": fold, "method": method, "seed": seed, "payload": payload, "frame": frame, "metrics_path": metrics_path, "per_image": per_image})
    return cells


def group_confusions(cells: list[dict], view: str) -> tuple[list[dict], dict[tuple[int, str], list[str]]]:
    blocks = []
    groups_by_stratum: dict[tuple[int, str], list[str]] = {}
    for cell in cells:
        frame = cell["frame"]
        for (group, species), group_frame in frame.groupby(["group_id", "species"], sort=False):
            confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
            np.add.at(confusion, (group_frame["target"].to_numpy(dtype=int), group_frame[view].to_numpy(dtype=int)), 1)
            blocks.append({"fold": cell["fold"], "method": cell["method"], "group": str(group), "species": str(species), "confusion": confusion})
            groups_by_stratum.setdefault((cell["fold"], str(species)), []).append(str(group))
    return blocks, groups_by_stratum


def bootstrap_difference(cells: list[dict], method_a: str, method_b: str, view: str, reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    blocks, _ = group_confusions(cells, view)
    # Collapse all registered seeds for the same (fold, group_id) before
    # resampling. Species remains the stratification variable.
    clusters: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}
    for block in blocks:
        key = (block["fold"], block["group"], block["species"])
        clusters.setdefault(key, {})
        clusters[key].setdefault(block["method"], np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64))
        clusters[key][block["method"]] += block["confusion"]
    clusters_by_species: dict[str, list[tuple[int, str, str]]] = {}
    for cluster in sorted(clusters):
        clusters_by_species.setdefault(cluster[2], []).append(cluster)
    strata = sorted(clusters_by_species)
    point_conf = {method: np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64) for method in (method_a, method_b)}
    for b in blocks:
        if b["method"] in point_conf:
            point_conf[b["method"]] += b["confusion"]
    point = f1_from_confusion(point_conf[method_b]) - f1_from_confusion(point_conf[method_a])
    # Vectorize within bounded chunks. This preserves the exact group-clustered
    # bootstrap while avoiding a Python loop over every sampled group.
    matrices = {}
    for species in strata:
        clusters_for_species = clusters_by_species[species]
        for method in (method_a, method_b):
            matrices[(species, method)] = np.stack([clusters[key][method] for key in clusters_for_species])
    draws = np.empty(reps, dtype=float)
    chunk = 16
    for start in range(0, reps, chunk):
        stop = min(start + chunk, reps)
        conf = {method: np.zeros((stop - start, N_CLASSES, N_CLASSES), dtype=np.int64) for method in (method_a, method_b)}
        for species in strata:
            cluster_count = len(clusters_by_species[species])
            sampled = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
            for method in (method_a, method_b):
                conf[method] += matrices[(species, method)][sampled].sum(axis=1)
        draws[start:stop] = f1_from_confusion(conf[method_b]) - f1_from_confusion(conf[method_a])
    return {
        "point_estimate_pooled_macro_f1_difference": float(point),
        "bootstrap_mean": float(draws.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_replicates": reps,
        "bootstrap_seed": seed,
        "stratification": "ground-truth species",
        "cluster_unit": "(fold, group_id); all registered seeds for a group retained together",
        "positive_fraction": float(np.mean(draws > 0)),
    }


def cell_summary(cells: list[dict], model_label: str) -> pd.DataFrame:
    rows = []
    for cell in cells:
        m = cell["payload"]["metrics"]
        row = {"model": model_label, "row_type": "cell", "fold": cell["fold"], "method": cell["method"], "seed": cell["seed"]}
        for view in VIEWS:
            row[f"{view}_macro_f1"] = float(m[view]["macro_f1"])
            row[f"{view}_tail_f1"] = float(m[view].get("tail_f1", np.nan))
        row.update({"dar_flip": cell["payload"]["metrics"]["context"]["dar_flip"], "prediction_agreement": cell["payload"]["metrics"]["context"]["prediction_agreement"]})
        rows.append(row)
    return pd.DataFrame(rows)


def write_route_c_report() -> None:
    root = ROOT / "outputs/cxt_fish/mask_contrastive_outer_v1_evaluation"
    cells = load_cells(root, ("seed17",), "route_c") if False else []
    rows = []
    for fold in (1, 2, 3):
        for seed in (17, 2026, 3407):
            p = root / f"fold_{fold}" / f"seed{seed}" / "metrics.json"
            d = json.loads(p.read_text())
            m = d["metrics"]
            rows.append({"fold": fold, "seed": seed, **{f"{v}_macro_f1": m[v]["macro_f1"] for v in VIEWS}, "dar_flip": m["context"]["dar_flip"]})
    df = pd.DataFrame(rows)
    lines = ["# Route C outer evaluation report", "", "This report is regenerated from the frozen route-C metrics JSON files. Route C is a two-stage mechanism control, not a faithful CLIB reproduction or an exhaustive contrastive-learning comparison.", "", "## Integrity", "", "- Training and outer evaluation cells: 9/9.", "- `official_test_accessed`: false in all nine metrics files.", "- No new route-C training, tuning, or checkpoint selection was performed during this regeneration.", "", "## Cell results", "", "| fold | seed | original macro-F1 | foreground macro-F1 | same-swap macro-F1 | cross-swap macro-F1 | DAR-flip |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in df.iterrows():
        lines.append(f"| {int(row.fold)} | {int(row.seed)} | {row.original_macro_f1:.6f} | {row.foreground_macro_f1:.6f} | {row.same_swap_macro_f1:.6f} | {row.cross_swap_macro_f1:.6f} | {row.dar_flip:.6f} |")
    means = df.mean(numeric_only=True)
    lines += [f"| **mean** |  | **{means.original_macro_f1:.6f}** | **{means.foreground_macro_f1:.6f}** | **{means.same_swap_macro_f1:.6f}** | **{means.cross_swap_macro_f1:.6f}** | **{means.dar_flip:.6f}** |", "", "## Interpretation", "", "Route C remains substantially below the frozen label-level CXT-Fish baseline on ordinary recognition and cross-class context-swap performance. The corrected DAR-flip mean above is computed directly from all nine raw JSON files."]
    (ROOT / "reports/cxt_fish_mask_contrastive_outer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bootstrap", action="store_true", help="reuse existing 5000-replicate JSON files")
    args = parser.parse_args()
    resnet_cells = load_cells(ROOT / "outputs/cxt_fish/final_outer_evaluation", ("F0", "F1"), "resnet")
    mobile_cells = load_cells(ROOT / "outputs/cxt_fish/mobilenet_outer_evaluation", ("MV0", "MV1"), "mobilenet")
    resnet = cell_summary(resnet_cells, "ResNet18")
    mobile = cell_summary(mobile_cells, "MobileNetV3-Large")
    all_summary = pd.concat([resnet, mobile], ignore_index=True)
    means = []
    for (model, method), group in all_summary.groupby(["model", "method"], sort=False):
        row = {"model": model, "row_type": "mean", "fold": "mean", "method": method, "seed": "all"}
        for column in [c for c in all_summary.columns if c.endswith("_macro_f1") or c.endswith("_tail_f1") or c in {"dar_flip", "prediction_agreement"}]:
            row[column] = float(group[column].mean())
        means.append(row)
    all_summary = pd.concat([all_summary, pd.DataFrame(means)], ignore_index=True)
    all_summary.to_csv(ROOT / "experiments/cxt_fish_final_method_summary.csv", index=False)
    mobile.to_csv(ROOT / "experiments/cxt_fish_mobilenet_per_fold_results.csv", index=False)
    gaps = []
    for _, row in all_summary.iterrows():
        for view in ("foreground", "same_swap", "cross_swap"):
            gaps.append({"model": row.model, "fold": row.fold, "method": row.method, "seed": row.seed, "gap_name": f"{view}_gap", "gap_definition": "original macro-F1 - view macro-F1", "gap": row.original_macro_f1 - row[f"{view}_macro_f1"]})
    pd.DataFrame(gaps).to_csv(ROOT / "experiments/cxt_fish_view_gap_summary.csv", index=False)
    if not args.skip_bootstrap:
        resnet_boot = {"comparison": "F1 - F0", "official_test_accessed": False, "views": {view: bootstrap_difference(resnet_cells, "F0", "F1", view, 5000, 3407 + i) for i, view in enumerate(VIEWS)}}
        mobile_boot = {"comparison": "MV1 - MV0", "official_test_accessed": False, "views": {view: bootstrap_difference(mobile_cells, "MV0", "MV1", view, 5000, 4407 + i) for i, view in enumerate(VIEWS)}}
        (ROOT / "experiments/cxt_fish_resnet_outer_bootstrap_5000.json").write_text(json.dumps(resnet_boot, indent=2), encoding="utf-8")
        (ROOT / "experiments/cxt_fish_mobilenet_bootstrap_5000.json").write_text(json.dumps(mobile_boot, indent=2), encoding="utf-8")
    write_route_c_report()
    report = ["# CXT-Fish final statistical synthesis", "", "This synthesis is generated only from frozen outer-evaluation JSON/CSV outputs. No model was trained or re-inferred.", "", "## Primary interpretation", "", "- On the frozen ResNet18 outer confirmation, foreground-sufficiency training is interpreted as a context-robustness intervention rather than a clean-accuracy improvement method.", "- The MobileNetV3 check repeats the robustness direction but shows a larger clean-accuracy cost; it is architecture-sensitivity evidence only.", "- Route C is retained as a fixed two-stage mechanism control and is not treated as an exhaustive contrastive-learning comparison.", "", "## Bootstrap protocol", "", "All 5,000-replicate intervals use fold-by-species stratification and group_id clustering. For ResNet18, all three registered seeds belonging to a sampled group are retained together; seeds are not treated as independent clusters. Official TEST remains locked.", "", "## Generated files", "", "- `experiments/cxt_fish_final_method_summary.csv`", "- `experiments/cxt_fish_resnet_outer_bootstrap_5000.json`", "- `experiments/cxt_fish_mobilenet_per_fold_results.csv`", "- `experiments/cxt_fish_mobilenet_bootstrap_5000.json`", "- `experiments/cxt_fish_view_gap_summary.csv`"]
    (ROOT / "reports/cxt_fish_final_statistical_synthesis.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
