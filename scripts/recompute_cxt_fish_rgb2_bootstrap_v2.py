"""Recompute the frozen F0-2RGB exploratory bootstrap from predictions only.

The observed effect is the equal-weight mean of nine paired cell differences.
Each bootstrap draw is species-stratified within fold, clusters by group_id,
and shares the fold-level draw across both methods and all three seeds.
No model, checkpoint, manifest, or image is modified or re-inferred.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDS = (1, 2, 3)
SEEDS = (3407, 2026, 17)
N_CLASSES = 16


def _f1(confusion: np.ndarray) -> float:
    tp = np.diag(confusion).astype(float)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    values = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(values.mean())


def _load_cells(rgb_root: Path, cxt_root: Path) -> tuple[list[dict], pd.DataFrame]:
    cells: list[dict] = []
    summary_rows: list[dict] = []
    for fold in FOLDS:
        for seed in SEEDS:
            rgb_dir = rgb_root / f"fold_{fold}" / f"F0_2RGB_seed{seed}"
            cxt_dir = cxt_root / f"fold_{fold}" / f"F1_seed{seed}"
            rgb_metrics = rgb_dir / "metrics.json"
            cxt_metrics = cxt_dir / "metrics.json"
            if not rgb_metrics.is_file() or not cxt_metrics.is_file():
                raise FileNotFoundError(f"missing paired metrics fold={fold} seed={seed}")
            rgb_payload = json.loads(rgb_metrics.read_text(encoding="utf-8"))
            cxt_payload = json.loads(cxt_metrics.read_text(encoding="utf-8"))
            if rgb_payload.get("official_test_accessed") is not False or cxt_payload.get("official_test_accessed") is not False:
                raise ValueError("official TEST flag is not false")
            rgb = pd.read_csv(rgb_dir / "per_image.csv")
            cxt = pd.read_csv(cxt_dir / "per_image.csv")
            keys = ["image_path", "group_id", "target"]
            if not rgb[keys].astype(str).reset_index(drop=True).equals(cxt[keys].astype(str).reset_index(drop=True)):
                raise ValueError(f"paired row identity mismatch fold={fold} seed={seed}")
            for method, payload in (("F0_2RGB", rgb_payload), ("CXT-Fish", cxt_payload)):
                summary_rows.append({
                    "fold": fold, "seed": seed, "method": method,
                    "clean_macro_f1": payload["metrics"]["original"]["macro_f1"],
                    "foreground_macro_f1": payload["metrics"]["foreground"]["macro_f1"],
                    "same_composite_macro_f1": payload["metrics"]["same_swap"]["macro_f1"],
                    "cross_composite_macro_f1": payload["metrics"]["cross_swap"]["macro_f1"],
                    "dar_flip": payload["metrics"]["context"]["dar_flip"],
                    "agreement": payload["metrics"]["context"]["prediction_agreement"],
                })
            cells.append({"fold": fold, "seed": seed, "rgb": rgb, "cxt": cxt})
    if len(cells) != 9:
        raise ValueError(f"expected 9 paired cells, found {len(cells)}")
    return cells, pd.DataFrame(summary_rows)


def _observed(summary: pd.DataFrame) -> float:
    cxt = summary.loc[summary.method == "CXT-Fish", "cross_composite_macro_f1"].to_numpy(float)
    rgb = summary.loc[summary.method == "F0_2RGB", "cross_composite_macro_f1"].to_numpy(float)
    if len(cxt) != 9 or len(rgb) != 9:
        raise ValueError("observed point estimate requires 9 cells per method")
    paired = cxt - rgb
    observed = float(paired.mean())
    method_means = float(cxt.mean() - rgb.mean())
    if not np.isclose(observed, method_means, atol=1e-12, rtol=0.0):
        raise AssertionError("paired cell mean does not equal difference of equal-weight method means")
    return observed


def _confusion(frame: pd.DataFrame, prediction: str) -> dict[tuple[str, str], np.ndarray]:
    result: dict[tuple[str, str], np.ndarray] = {}
    for (species, group), part in frame.groupby(["target", "group_id"], sort=True):
        matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        np.add.at(matrix, (part.target.to_numpy(int), part[prediction].to_numpy(int)), 1)
        result[(str(species), str(group))] = matrix
    return result


def _bootstrap(cells: list[dict], reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks: dict[tuple[int, int, str], dict[tuple[str, str], np.ndarray]] = {}
    strata: dict[int, dict[str, list[str]]] = {}
    for cell in cells:
        key = (cell["fold"], cell["seed"])
        blocks[key] = {
            "rgb": _confusion(cell["rgb"], "cross_swap"),
            "cxt": _confusion(cell["cxt"], "cross_swap"),
        }
        strata.setdefault(cell["fold"], {})
        for species, group in blocks[key]["rgb"]:
            strata[cell["fold"]].setdefault(species, []).append(group)
    for fold in strata:
        for species in strata[fold]:
            strata[fold][species] = sorted(set(strata[fold][species]))

    keys = sorted(blocks)
    arrays: dict[tuple[int, str, int, str], np.ndarray] = {}
    for cell_index, (fold, seed_value) in enumerate(keys):
        for species, groups in sorted(strata[fold].items()):
            ordered = sorted(groups)
            for method in ("rgb", "cxt"):
                arrays[(fold, species, cell_index, method)] = np.stack([
                    blocks[(fold, seed_value)][method].get((species, group), np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64))
                    for group in ordered
                ])

    draws = np.empty(reps, dtype=float)
    chunk = 256
    for start in range(0, reps, chunk):
        stop = min(start + chunk, reps)
        n_draws = stop - start
        conf = {method: np.zeros((n_draws, len(keys), N_CLASSES, N_CLASSES), dtype=np.int64) for method in ("rgb", "cxt")}
        for fold in FOLDS:
            for species, groups in sorted(strata[fold].items()):
                sampled = rng.integers(0, len(groups), size=(n_draws, len(groups)))
                for cell_index, (cell_fold, _seed_value) in enumerate(keys):
                    if cell_fold != fold:
                        continue
                    for method in ("rgb", "cxt"):
                        conf[method][:, cell_index] += arrays[(fold, species, cell_index, method)][sampled].sum(axis=1)
        values = {}
        for method in ("rgb", "cxt"):
            matrix = conf[method]
            tp = np.diagonal(matrix, axis1=2, axis2=3).astype(float)
            fp = matrix.sum(axis=2) - tp
            fn = matrix.sum(axis=3) - tp
            denom = 2.0 * tp + fp + fn
            per_class = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
            values[method] = per_class.mean(axis=2)
        draws[start:stop] = (values["cxt"] - values["rgb"]).mean(axis=1)
    return draws


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3410)
    args = parser.parse_args()
    rgb_root = ROOT / "outputs/cxt_fish/rgb2_control_outer_evaluation"
    cxt_root = ROOT / "outputs/cxt_fish/final_outer_evaluation"
    cells, summary = _load_cells(rgb_root, cxt_root)
    observed = _observed(summary)
    draws = _bootstrap(cells, args.replicates, args.seed)
    payload = {
        "schema_version": "cxt_fish_rgb2_bootstrap_v2",
        "observed_point_estimate": observed,
        "bootstrap_mean": float(draws.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_replicates": args.replicates,
        "bootstrap_seed": args.seed,
        "interval_method": "percentile",
        "estimand": "equal_weight_cell_mean_CXT_minus_F0_2RGB_cross_class_donor_context_composite_macro_f1",
        "cluster_unit": "group_id",
        "stratification": "ground_truth_species_within_fold",
        "shared_resampling_structure": "paired_methods_and_three_seeds_within_fold",
        "analysis_role": "exploratory_post_hoc_mechanism_control",
        "official_test_accessed": False,
        "supersedes": "experiments/cxt_fish_rgb2_vs_f1_bootstrap_5000.json",
    }
    out = ROOT / "experiments/cxt_fish_rgb2_vs_f1_bootstrap_5000_v2.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary.to_csv(ROOT / "experiments/cxt_fish_rgb2_control_per_cell.csv", index=False)
    means = summary.groupby("method", sort=False).mean(numeric_only=True).reset_index()
    means.to_csv(ROOT / "experiments/cxt_fish_rgb2_control_summary.csv", index=False)
    pp = observed * 100
    lo, hi = payload["ci95"][0] * 100, payload["ci95"][1] * 100
    report = [
        "# F0-2RGB statistical repair",
        "",
        "This is an exploratory post-hoc mechanism-control reanalysis from frozen per-image predictions. No model was trained or re-inferred.",
        "",
        "The superseded JSON used the bootstrap distribution mean as `point_estimate`. The observed point estimate is instead the equal-weight mean of the nine paired cell differences.",
        "",
        f"- Observed CXT-Fish minus F0-2RGB cross-composite effect: **{pp:+.2f} pp**.",
        f"- Bootstrap mean: {payload['bootstrap_mean'] * 100:+.2f} pp.",
        f"- 5,000-replicate percentile interval: [{lo:+.2f}, {hi:+.2f}] pp.",
        "- Resampling: ground-truth-species-stratified group clusters within fold; paired methods and all three seeds share each fold draw.",
        "- `official_test_accessed`: false.",
        "",
        "The former `experiments/cxt_fish_rgb2_vs_f1_bootstrap_5000.json` is retained as a superseded statistical artifact and is not used for manuscript numbers.",
    ]
    (ROOT / "reports/cxt_fish_rgb2_statistical_repair.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (ROOT / "reports/cxt_fish_rgb2_control_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (ROOT / "reports/manuscript_rgb2_corrected_numbers.md").write_text(
        "# Corrected F0-2RGB numbers\n\n"
        "All values below are generated from frozen outer per-image predictions.\n\n"
        f"- CXT-Fish clean macro-F1: {means.loc[means.method == 'CXT-Fish', 'clean_macro_f1'].iloc[0]:.10f}\n"
        f"- F0-2RGB clean macro-F1: {means.loc[means.method == 'F0_2RGB', 'clean_macro_f1'].iloc[0]:.10f}\n"
        f"- CXT-Fish cross-composite macro-F1: {means.loc[means.method == 'CXT-Fish', 'cross_composite_macro_f1'].iloc[0]:.10f}\n"
        f"- F0-2RGB cross-composite macro-F1: {means.loc[means.method == 'F0_2RGB', 'cross_composite_macro_f1'].iloc[0]:.10f}\n"
        f"- Observed paired difference: {pp:+.2f} pp\n"
        f"- Exploratory 5,000-replicate percentile interval: [{lo:+.2f}, {hi:+.2f}] pp\n\n"
        "The earlier +6.75 pp value was the bootstrap distribution mean, not the observed point estimate, and is superseded.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
