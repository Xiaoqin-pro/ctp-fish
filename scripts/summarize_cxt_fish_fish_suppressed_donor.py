"""Summarize the fixed-pair fish-suppressed donor sensitivity."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDS = (1, 2, 3)
SEEDS = (3407, 2026, 17)
N_CLASSES = 16


def f1(conf: np.ndarray) -> float:
    tp = np.diag(conf).astype(float); fp = conf.sum(0) - tp; fn = conf.sum(1) - tp; den = 2 * tp + fp + fn
    return float(np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0).mean())


def load() -> tuple[list[dict], pd.DataFrame]:
    cells = []; rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            root = ROOT / f"outputs/cxt_fish/fish_suppressed_donor_evaluation/fold_{fold}"
            f0 = root / f"F0_seed{seed}"; f1 = root / f"F1_seed{seed}"
            for method, directory in (("F0", f0), ("F1", f1)):
                payload = json.loads((directory / "metrics.json").read_text(encoding="utf-8"));
                if payload.get("official_test_accessed") is not False: raise ValueError("official TEST flag violation")
                m = payload["metrics"]["cross_suppressed"]
                rows.append({"fold": fold, "seed": seed, "method": method, "cross_suppressed_macro_f1": m["macro_f1"], "head_f1": m.get("head_f1"), "mid_f1": m.get("mid_f1"), "tail_f1": m.get("tail_f1"), "group_balanced_accuracy": m.get("group_balanced_accuracy"), "dar": payload["metrics"]["context"]["dar"], "dar_flip": payload["metrics"]["context"]["dar_flip"], "agreement": payload["metrics"]["context"]["prediction_agreement"]})
            f0_frame = pd.read_csv(f0 / "per_image.csv"); f1_frame = pd.read_csv(f1 / "per_image.csv")
            keys = ["image_path", "group_id", "target"]
            if not f0_frame[keys].astype(str).reset_index(drop=True).equals(f1_frame[keys].astype(str).reset_index(drop=True)): raise ValueError(f"paired identity mismatch fold={fold} seed={seed}")
            cells.append({"fold": fold, "seed": seed, "f0": f0_frame, "f1": f1_frame})
    return cells, pd.DataFrame(rows)


def blocks(frame: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    result = {}
    for (species, group), part in frame.groupby(["target", "group_id"], sort=True):
        conf = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64); np.add.at(conf, (part.target.to_numpy(int), part.prediction.to_numpy(int)), 1); result[(str(species), str(group))] = conf
    return result


def bootstrap(cells: list[dict], reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); keys = sorted((c["fold"], c["seed"]) for c in cells); all_blocks = {}
    strata = {}
    for c in cells:
        key = (c["fold"], c["seed"]); all_blocks[key] = {"f0": blocks(c["f0"]), "f1": blocks(c["f1"])}; strata.setdefault(c["fold"], {})
        for species, group in all_blocks[key]["f0"]: strata[c["fold"].__int__()].setdefault(species, []).append(group)
    for fold in strata:
        for species in strata[fold]: strata[fold][species] = sorted(set(strata[fold][species]))
    arrays = {}
    for idx, (fold, seed_value) in enumerate(keys):
        for species, groups in sorted(strata[fold].items()):
            for method in ("f0", "f1"):
                arrays[(fold, species, idx, method)] = np.stack([all_blocks[(fold, seed_value)][method].get((species, group), np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)) for group in sorted(groups)])
    draws = np.empty(reps); chunk = 128
    for start in range(0, reps, chunk):
        stop = min(start + chunk, reps); n = stop - start; conf = {m: np.zeros((n, len(keys), N_CLASSES, N_CLASSES), dtype=np.int64) for m in ("f0", "f1")}
        for fold in FOLDS:
            for species, groups in sorted(strata[fold].items()):
                sample = rng.integers(0, len(groups), size=(n, len(groups)))
                for idx, (cell_fold, _seed_value) in enumerate(keys):
                    if cell_fold != fold: continue
                    for method in ("f0", "f1"): conf[method][:, idx] += arrays[(fold, species, idx, method)][sample].sum(axis=1)
        values = {}
        for method in ("f0", "f1"):
            m = conf[method]; tp = np.diagonal(m, axis1=2, axis2=3).astype(float); fp = m.sum(2) - tp; fn = m.sum(3) - tp; den = 2 * tp + fp + fn; values[method] = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0).mean(2)
        draws[start:stop] = (values["f1"] - values["f0"]).mean(1)
    return draws


def main() -> None:
    cells, summary = load(); f1_values = summary.loc[summary.method == "F1", "cross_suppressed_macro_f1"].to_numpy(); f0_values = summary.loc[summary.method == "F0", "cross_suppressed_macro_f1"].to_numpy(); observed = float((f1_values - f0_values).mean()); draws = bootstrap(cells, 5000, 3410)
    payload = {"schema_version": "cxt_fish_fish_suppressed_donor_bootstrap_v1", "observed_point_estimate": observed, "bootstrap_mean": float(draws.mean()), "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))], "bootstrap_replicates": 5000, "bootstrap_seed": 3410, "interval_method": "percentile", "estimand": "equal_weight_cell_mean_CXT_minus_F0_fish_suppressed_cross_class_donor_composite_macro_f1", "cluster_unit": "group_id", "stratification": "ground_truth_species_within_fold", "shared_resampling_structure": "paired_methods_and_three_seeds_within_fold", "analysis_role": "post_hoc_construct_validity_sensitivity", "official_test_accessed": False}
    (ROOT / "experiments/cxt_fish_fish_suppressed_donor_bootstrap_5000.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); summary.to_csv(ROOT / "experiments/cxt_fish_fish_suppressed_donor_summary.csv", index=False)
    original = pd.read_csv(ROOT / "experiments/cxt_fish_final_method_summary.csv")
    original = original.loc[(original.model == "ResNet18") & (original.row_type == "cell")]
    original_f0 = original.loc[original.method == "F0", "cross_swap_macro_f1"].mean(); original_cxt = original.loc[original.method == "F1", "cross_swap_macro_f1"].mean()
    lines = ["# Fish-suppressed donor construct-validity sensitivity", "", "This is a post-hoc donor-subject-suppressed construct-validity sensitivity. It reuses the frozen primary recipient/donor pairings and checkpoints; only the donor subject is suppressed with the frozen Telea rule.", "", f"- Original frozen donor composite: CXT-Fish − F0 = {(original_cxt - original_f0) * 100:+.2f} pp.", f"- Fish-suppressed donor composite observed effect: {observed * 100:+.2f} pp.", f"- Exploratory 5,000-replicate percentile interval: [{payload['ci95'][0] * 100:+.2f}, {payload['ci95'][1] * 100:+.2f}] pp.", "", "All outcomes are reported without a positive-result gate. This analysis does not prove natural-background robustness and does not change the frozen main result.", "", "| fold | seed | F0 fish-suppressed | CXT-Fish fish-suppressed | delta |", "|---:|---:|---:|---:|---:|"]
    for fold in FOLDS:
        for seed in SEEDS:
            a = summary[(summary.fold == fold) & (summary.seed == seed) & (summary.method == "F0")].iloc[0]; b = summary[(summary.fold == fold) & (summary.seed == seed) & (summary.method == "F1")].iloc[0]; lines.append(f"| {fold} | {seed} | {a.cross_suppressed_macro_f1:.6f} | {b.cross_suppressed_macro_f1:.6f} | {(b.cross_suppressed_macro_f1 - a.cross_suppressed_macro_f1) * 100:+.2f} pp |")
    (ROOT / "reports/cxt_fish_fish_suppressed_donor_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Simple paper-facing comparison, with no donor-family reweighting.
    plt.figure(figsize=(7, 4)); x = np.arange(2); plt.bar(x - .18, [original_f0, float(summary.loc[summary.method == "F0", "cross_suppressed_macro_f1"].mean())], .36, label="F0"); plt.bar(x + .18, [original_cxt, float(summary.loc[summary.method == "F1", "cross_suppressed_macro_f1"].mean())], .36, label="CXT-Fish"); plt.xticks(x, ["Original donor composite", "Fish-suppressed donor"]); plt.ylabel("Cross-composite macro-F1"); plt.legend(); plt.tight_layout(); plt.savefig(ROOT / "reports/figures/cxt_fish_fish_suppressed_donor_comparison.png", dpi=180); plt.close()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
