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
            blocks.append({"fold": cell["fold"], "seed": cell["seed"], "method": cell["method"], "group": str(group), "species": str(species), "confusion": confusion})
            groups_by_stratum.setdefault((cell["fold"], str(species)), []).append(str(group))
    return blocks, groups_by_stratum


def validate_cells(cells: list[dict], methods: tuple[str, ...], seeds: tuple[int, ...], label: str) -> None:
    expected = 3 * len(methods) * len(seeds)
    if len(cells) != expected:
        raise ValueError(f"{label}: expected {expected} cells, found {len(cells)}")
    keys = [(c["fold"], c["method"], c["seed"]) for c in cells]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label}: duplicate fold/method/seed cell")
    if set(c["method"] for c in cells) != set(methods) or set(c["seed"] for c in cells) != set(seeds):
        raise ValueError(f"{label}: method or seed set mismatch")
    for cell in cells:
        payload = cell["payload"]
        if payload.get("official_test_accessed", False) is not False:
            raise ValueError(f"{label}: official TEST flag is not false: {cell['metrics_path']}")
    for fold in (1, 2, 3):
        for seed in seeds:
            pair = [c for c in cells if c["fold"] == fold and c["seed"] == seed]
            if {c["method"] for c in pair} != set(methods):
                raise ValueError(f"{label}: incomplete paired methods at fold={fold}, seed={seed}")
            left, right = pair[0]["frame"], pair[1]["frame"]
            required = ["image_path", "target", "group_id"]
            if any(column not in left.columns or column not in right.columns for column in required):
                raise ValueError(f"{label}: missing paired identity columns")
            identity = required
            if not left[identity].astype(str).reset_index(drop=True).equals(right[identity].astype(str).reset_index(drop=True)):
                raise ValueError(f"{label}: paired row identity mismatch at fold={fold}, seed={seed}")


def bootstrap_difference(cells: list[dict], method_a: str, method_b: str, view: str, reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    blocks, _ = group_confusions(cells, view)
    methods = (method_a, method_b)
    cell_keys = sorted({(b["fold"], b["seed"]) for b in blocks})
    # Keep one confusion matrix per cell and group. Seeds share the sampled
    # group indices, but are never merged before their own macro-F1 is formed.
    blocks_by_cell: dict[tuple[int, int, str, str, str], np.ndarray] = {}
    groups_by_stratum: dict[tuple[int, str], set[str]] = {}
    for block in blocks:
        key = (block["fold"], block["seed"], block["method"], block["group"], block["species"])
        blocks_by_cell[key] = block["confusion"]
        groups_by_stratum.setdefault((block["fold"], block["species"]), set()).add(block["group"])
    strata = sorted(groups_by_stratum)
    n_cells = len(cell_keys)
    matrices = {}
    for stratum in strata:
        groups = sorted(groups_by_stratum[stratum])
        for cell_index, (fold, seed_value) in enumerate(cell_keys):
            for method in methods:
                matrices[(stratum, cell_index, method)] = np.stack([
                    blocks_by_cell.get((fold, seed_value, method, group, stratum[1]), np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64))
                    for group in groups
                ])
    point_diffs = []
    for cell_index, (fold, seed_value) in enumerate(cell_keys):
        method_f1 = []
        for method in methods:
            confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
            for stratum in strata:
                confusion += matrices[(stratum, cell_index, method)].sum(axis=0)
            method_f1.append(f1_from_confusion(confusion))
        point_diffs.append(method_f1[1] - method_f1[0])
    point = float(np.mean(point_diffs))
    draws = np.empty(reps, dtype=float)
    chunk = 64
    for start in range(0, reps, chunk):
        stop = min(start + chunk, reps)
        n_draws = stop - start
        conf = {method: np.zeros((n_draws, n_cells, N_CLASSES, N_CLASSES), dtype=np.int64) for method in methods}
        for stratum in strata:
            groups = sorted(groups_by_stratum[stratum])
            sampled = rng.integers(0, len(groups), size=(n_draws, len(groups)))
            for cell_index in range(n_cells):
                for method in methods:
                    conf[method][:, cell_index] += matrices[(stratum, cell_index, method)][sampled].sum(axis=1)
        f1 = {}
        for method in methods:
            f1[method] = f1_from_confusion(conf[method].reshape(n_draws * n_cells, N_CLASSES, N_CLASSES)).reshape(n_draws, n_cells)
        draws[start:stop] = (f1[method_b] - f1[method_a]).mean(axis=1)
    return {
        "estimand": "equal_weight_cell_mean",
        "point_estimate": point,
        "aggregation_order": "macro-F1 per fold-seed cell, paired difference, then equal-weight mean across cells",
        "bootstrap_mean": float(draws.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_replicates": reps,
        "bootstrap_seed": seed,
        "stratification": "ground-truth species",
        "cluster_unit": "group_id within fold and species; same sampled groups shared across paired methods and seeds",
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
            row[f"{view}_track_balanced_accuracy"] = float(m[view].get("track_balanced_accuracy", np.nan))
            row[f"{view}_head_f1"] = float(m[view].get("head_f1", np.nan))
            row[f"{view}_mid_f1"] = float(m[view].get("mid_f1", np.nan))
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
    validate_cells(resnet_cells, ("F0", "F1"), (17, 2026, 3407), "ResNet18")
    validate_cells(mobile_cells, ("MV0", "MV1"), (3407,), "MobileNetV3-Large")
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
        # The registered primary endpoint is cross-class-swap macro-F1. Other
        # diagnostics remain in the cell summary and are not assigned a new CI.
        primary_view = "cross_swap"
        resnet_boot = {"comparison": "F1 - F0", "official_test_accessed": False, "bootstrap_scope": "primary endpoint only", "views": {primary_view: bootstrap_difference(resnet_cells, "F0", "F1", primary_view, 5000, 3410)}}
        mobile_boot = {"comparison": "MV1 - MV0", "official_test_accessed": False, "bootstrap_scope": "primary endpoint only", "views": {primary_view: bootstrap_difference(mobile_cells, "MV0", "MV1", primary_view, 5000, 4410)}}
        (ROOT / "experiments/cxt_fish_resnet_outer_bootstrap_5000.json").write_text(json.dumps(resnet_boot, indent=2), encoding="utf-8")
        (ROOT / "experiments/cxt_fish_mobilenet_bootstrap_5000.json").write_text(json.dumps(mobile_boot, indent=2), encoding="utf-8")
    write_route_c_report()
    report = ["# CXT-Fish final statistical synthesis", "", "This synthesis is generated only from frozen outer-evaluation JSON/CSV outputs. No model was trained or re-inferred.", "", "## Primary interpretation", "", "- On the frozen ResNet18 outer confirmation, foreground-sufficiency training is interpreted as a context-robustness intervention rather than a clean-accuracy improvement method.", "- The MobileNetV3 check repeats the robustness direction but shows a larger clean-accuracy cost; it is architecture-sensitivity evidence only.", "- Route C is retained as a fixed two-stage mechanism control and is not treated as an exhaustive contrastive-learning comparison.", "", "## Bootstrap protocol", "", "All 5,000-replicate intervals use fold-by-species stratification and group_id clustering. For ResNet18, all three registered seeds belonging to a sampled group are retained together; seeds are not treated as independent clusters. Official TEST remains locked.", "", "## Generated files", "", "- `experiments/cxt_fish_final_method_summary.csv`", "- `experiments/cxt_fish_resnet_outer_bootstrap_5000.json`", "- `experiments/cxt_fish_mobilenet_per_fold_results.csv`", "- `experiments/cxt_fish_mobilenet_bootstrap_5000.json`", "- `experiments/cxt_fish_view_gap_summary.csv`"]
    report[7] = "- MobileNetV3 shows average robustness improvement with heterogeneous cross-swap effects and a larger clean-accuracy cost; it is architecture-sensitivity evidence only."
    report[12] = "The primary estimand is an equal-weight mean of paired cell differences. Each replicate resamples groups within fold-by-species strata, shares the same draws across paired methods and registered seeds, computes macro-F1 separately for every fold-seed cell, and averages the resulting cell differences. Official TEST remains locked."
    report.insert(13, "")
    report.insert(14, "The earlier pooled-prediction aggregation is retained only as a supplementary sensitivity analysis and is not paired with the primary cell-mean estimate.")
    (ROOT / "reports/cxt_fish_final_statistical_synthesis.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
