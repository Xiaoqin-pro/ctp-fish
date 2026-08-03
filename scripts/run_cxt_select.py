"""Run the frozen CXT-Select zero-training evaluation on current val only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics.classification import classification_metrics
from metrics.track_metrics import cluster_bootstrap_mean, track_balanced_accuracy
from scripts.evaluate_cxt_phase1c import ContextEvaluationDataset, load_checkpoint, predict, evaluation_transform
from tools.io_utils import atomic_csv_dump, atomic_json_dump


SCORES = ("f1_msp", "f1_entropy", "label_disagreement", "js_divergence", "random")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_arrays(f1_logits: np.ndarray, f3_logits: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    def softmax(x):
        x = x - x.max(axis=1, keepdims=True); e = np.exp(x); return e / e.sum(axis=1, keepdims=True)
    p = softmax(f1_logits); q = softmax(f3_logits); m = 0.5 * (p + q)
    js = 0.5 * (p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(m, 1e-12))).sum(1)) + 0.5 * (q * (np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(m, 1e-12))).sum(1))
    return {
        "f1_msp": 1.0 - p.max(axis=1),
        "f1_entropy": -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1),
        "label_disagreement": (f1_logits.argmax(1) != f3_logits.argmax(1)).astype(float),
        "js_divergence": js,
        "random": np.random.default_rng(seed).random(len(f1_logits)),
    }


def auroc_ap(score: np.ndarray, target: np.ndarray) -> dict:
    target = np.asarray(target, dtype=bool)
    if target.all() or (~target).all():
        return {"auroc": None, "auprc": None, "defined": False, "undefined_reason": "single_class_event_target", "prevalence": float(target.mean())}
    return {"auroc": float(roc_auc_score(target, score)), "auprc": float(average_precision_score(target, score)), "defined": True, "prevalence": float(target.mean())}


def auc_risk(score: np.ndarray, error: np.ndarray) -> float:
    order = np.lexsort((np.arange(len(score)), score)); errors = np.asarray(error, dtype=float)[order]
    coverage = np.arange(1, len(score) + 1, dtype=float) / len(score); risk = np.cumsum(errors) / np.arange(1, len(score) + 1)
    return float(np.trapezoid(risk, coverage))


def subset_metrics(frame: pd.DataFrame, accepted: np.ndarray, class_ids: list[str], tier_map: dict[str, str]) -> dict:
    sub = frame.iloc[np.flatnonzero(accepted)]
    metric = classification_metrics(sub.target.to_numpy(), sub.f1_prediction.to_numpy(), list(range(len(class_ids))))
    by_class = dict(zip(map(str, class_ids), metric["per_class_f1"])); metric["tail_f1"] = float(np.mean([by_class[x] for x, tier in tier_map.items() if tier == "tail"]))
    track = sub[["group_id", "target", "f1_prediction"]].rename(columns={"f1_prediction": "prediction"}); track["correct"] = track.target.eq(track.prediction); value, _ = track_balanced_accuracy(track); metric["track_balanced_accuracy"] = value
    metric["coverage"] = float(len(sub) / len(frame)); metric["per_class_coverage"] = {str(c): float(accepted[frame.species_id.astype(str).to_numpy() == str(c)].mean()) for c in class_ids}
    metric["per_track_coverage"] = {str(group): float(accepted[frame.group_id.astype(str).to_numpy() == str(group)].mean()) for group in frame.group_id.astype(str).unique()}
    orig_correct = frame.f1_prediction.eq(frame.target); cross_correct = frame.cross_prediction.eq(frame.target)
    metric["cross_swap_macro_f1"] = float(classification_metrics(sub.target.to_numpy(), sub.cross_prediction.to_numpy(), list(range(len(class_ids))))["macro_f1"])
    cross_track = sub[["group_id", "target", "cross_prediction"]].rename(columns={"cross_prediction": "prediction"}); cross_track["correct"] = cross_track.target.eq(cross_track.prediction); metric["cross_swap_track_balanced_accuracy"] = track_balanced_accuracy(cross_track)[0]
    eligible = accepted & orig_correct.to_numpy(); metric["dar_flip"] = float((frame.dar_flip.to_numpy()[eligible]).mean()) if eligible.any() else None; metric["prediction_agreement"] = float(frame.original_agreement.to_numpy()[accepted].mean())
    return metric


def evaluate_score(frame: pd.DataFrame, score_name: str, score: np.ndarray, class_ids: list[str], tier_map: dict[str, str], coverage_levels: list[float], review_fraction: float) -> tuple[dict, list[dict], dict]:
    clean_error = frame.f1_prediction.ne(frame.target).to_numpy(); cross_flip = (frame.f1_prediction.eq(frame.target) & frame.cross_prediction.ne(frame.target)).to_numpy(); dar_flip = (frame.f1_prediction.eq(frame.target) & frame.cross_prediction.eq(frame.cross_donor_target)).to_numpy()
    detection = {"clean_error": auroc_ap(score, clean_error), "cross_swap_flip": auroc_ap(score[frame.f1_prediction.eq(frame.target).to_numpy()], cross_flip[frame.f1_prediction.eq(frame.target).to_numpy()]), "dar_flip": auroc_ap(score[frame.f1_prediction.eq(frame.target).to_numpy()], dar_flip[frame.f1_prediction.eq(frame.target).to_numpy()])}
    rows = []; n = len(frame); order = np.lexsort((np.arange(n), score));
    for coverage in coverage_levels:
        count = n if coverage >= 1.0 else max(1, int(np.floor(coverage * n))); accepted = np.zeros(n, dtype=bool); accepted[order[:count]] = True; item = subset_metrics(frame, accepted, class_ids, tier_map); item.update({"score": score_name, "coverage_target": coverage}); rows.append(item)
    top = max(1, int(np.ceil(review_fraction * n))); reviewed = np.zeros(n, dtype=bool); reviewed[order[-top:]] = True
    capture = {"clean_error": float(clean_error[reviewed].sum() / max(1, clean_error.sum())), "cross_swap_flip": float(cross_flip[reviewed].sum() / max(1, cross_flip.sum())), "dar_flip": float(dar_flip[reviewed].sum() / max(1, dar_flip.sum()))}
    return {"detection": detection, "aurc_clean_error": auc_risk(score, clean_error), "top_review_capture": capture}, rows, {"clean_error": clean_error, "cross_swap_flip": cross_flip, "dar_flip": dar_flip}


def bootstrap_pair(frame: pd.DataFrame, js: np.ndarray, msp: np.ndarray, reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed); groups = frame.groupby(["species_id", "group_id"], sort=True).indices; by_species: dict[str, list[np.ndarray]] = {}
    for (species, _group), indices in groups.items(): by_species.setdefault(str(species), []).append(np.asarray(indices, dtype=int))
    values = {"aurc_clean_error": [], "selective_risk_90": [], "cross_swap_auprc": [], "dar_flip_auprc": []}
    for _ in range(reps):
        sampled = [rng.choice(group_list, size=len(group_list), replace=True) for group_list in by_species.values()]; indices = np.concatenate([np.concatenate(x) for x in sampled]); sub = frame.iloc[indices].reset_index(drop=True)
        for name, score in (("js", js[indices]), ("msp", msp[indices])):
            clean = sub.f1_prediction.ne(sub.target).to_numpy(); cross_eligible = sub.f1_prediction.eq(sub.target).to_numpy(); cross = sub.cross_prediction.ne(sub.target).to_numpy()[cross_eligible]; dar = sub.cross_prediction.eq(sub.cross_donor_target).to_numpy()[cross_eligible]
            key = {"js": "js", "msp": "msp"}[name]; values.setdefault(key, [])
            values["aurc_clean_error"].append((key, auc_risk(score, clean))); count = max(1, int(.9 * len(score))); order = np.lexsort((np.arange(len(score)), score)); accepted = order[:count]; values["selective_risk_90"].append((key, float(clean[accepted].mean()))); values["cross_swap_auprc"].append((key, float(average_precision_score(cross, score[cross_eligible])) if cross.any() and (~cross).any() else np.nan)); values["dar_flip_auprc"].append((key, float(average_precision_score(dar, score[cross_eligible])) if dar.any() and (~dar).any() else np.nan))
    output = {}
    for metric, entries in values.items():
        if not entries: continue
        a = np.asarray([value for key, value in entries if key == "js"], dtype=float); b = np.asarray([value for key, value in entries if key == "msp"], dtype=float); d = a - b; d = d[np.isfinite(d)]; output[metric] = {"js_minus_msp_mean": float(np.mean(d)), "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))], "defined_replicates": int(len(d))}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/cxt_fish_phase3_selective.yaml"); parser.add_argument("--output-root", default="outputs/cxt_fish/phase3_selective")
    args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); split = pd.read_csv(cfg["track_split_path"]); metadata = pd.read_csv(cfg["metadata_path"]); records = metadata.merge(split.loc[split.split == "val", ["image_path", "split"]], on="image_path", validate="one_to_one");
    if set(records.split) != {"val"} or "test" in set(records.split): raise ValueError("CXT-Select requires current val only")
    class_ids = [str(v) for v in torch.load(Path(cfg["checkpoint_pairs"][str(cfg["seeds"][0])]["f1"]), map_location="cpu", weights_only=False)["class_ids"]]; output_root = Path(args.output_root); output_root.mkdir(parents=True, exist_ok=True); detection_rows = []; curve_rows = []; bootstrap = {}
    train_records = metadata.merge(split.loc[split.split == "train", ["image_path", "split"]], on="image_path", validate="one_to_one"); train_counts = train_records.species_id.astype(str).value_counts(); ordered = sorted(class_ids, key=lambda x: (-int(train_counts.get(x, 0)), x)); tier_map = {x: tier for tier, group in zip(("head", "mid", "tail"), np.array_split(np.asarray(ordered, dtype=object), 3)) for x in group.tolist()}
    manifest = pd.read_csv(cfg["swap_manifest_path"]); dataset = ContextEvaluationDataset(records, manifest, evaluation_transform(int(cfg["image_size"])), cfg, class_ids); loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=2)
    for seed in cfg["seeds"]:
        pair = cfg["checkpoint_pairs"][str(seed)]; models = [load_checkpoint(Path(pair[name]), torch.device("cuda" if torch.cuda.is_available() else "cpu"))[0] for name in ("f1", "f3")]; device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); logits = {name: {view: [] for view in ("original", "foreground", "same_swap", "cross_swap")} for name in ("f1", "f3")}; targets = []; donors = []; paths = []; groups = []
        with torch.no_grad():
            for original, foreground, same, cross, target, donor, path, group in loader:
                for model_name, model in zip(("f1", "f3"), models):
                    for view_name, image in (("original", original), ("foreground", foreground), ("same_swap", same), ("cross_swap", cross)):
                        logits[model_name][view_name].append(predict(model, image.to(device)).cpu().numpy())
                targets.extend(target.tolist()); donors.extend(donor.tolist()); paths.extend(path); groups.extend(group)
        f1_logits = {view: np.concatenate(logits["f1"][view]) for view in logits["f1"]}; f3_logits = {view: np.concatenate(logits["f3"][view]) for view in logits["f3"]}; np.savez_compressed(output_root / f"seed_{seed}_logits.npz", **{f"f1_{view}": value.astype(np.float32) for view, value in f1_logits.items()}, **{f"f3_{view}": value.astype(np.float32) for view, value in f3_logits.items()}, target=np.asarray(targets, dtype=np.int64), cross_donor_target=np.asarray(donors, dtype=np.int64))
        frame = pd.DataFrame({"seed": seed, "image_path": paths, "group_id": groups, "target": targets, "cross_donor_target": donors}); frame["species_id"] = frame.image_path.map(records.set_index("image_path").species_id.astype(str)); f1o = f1_logits["original"]; f3o = f3_logits["original"]; frame["f1_prediction"] = f1o.argmax(1); frame["f3_prediction"] = f3o.argmax(1); frame["cross_prediction"] = f1_logits["cross_swap"].argmax(1); frame["f3_cross_prediction"] = f3_logits["cross_swap"].argmax(1); frame["original_agreement"] = frame.f1_prediction.eq(frame.f3_prediction); frame["dar_flip"] = frame.cross_prediction.eq(frame.cross_donor_target); scores = score_arrays(f1o, f3o, int(seed)); risk_columns = {}
        for name, score in scores.items():
            result, curves, events = evaluate_score(frame, name, score, class_ids, tier_map, list(map(float, cfg["coverage_levels"])), float(cfg["review_fraction"])); detection_rows.append({"seed": seed, "score": name, **result}); curve_rows.extend([{**row, "seed": seed} for row in curves]); risk_columns[f"risk_{name}"] = score
        frame.assign(**risk_columns).to_csv(output_root / f"seed_{seed}_per_image.csv", index=False)
        bootstrap[str(seed)] = bootstrap_pair(frame, scores["js_divergence"], scores["f1_msp"], int(cfg["bootstrap_replicates"]), int(cfg["bootstrap_seed"]))
        del models
    pd.DataFrame(detection_rows).to_json(output_root / "detection_metrics.json", orient="records", indent=2); pd.DataFrame(curve_rows).to_csv(output_root / "risk_coverage.csv", index=False); atomic_json_dump({"schema_version": "cxt_select_bootstrap_v1", "results": bootstrap, "official_test_accessed": False, "internal_test_accessed": False, "outer_folds_accessed": False}, output_root / "bootstrap.json", overwrite=True); atomic_json_dump({"seeds": cfg["seeds"], "records": len(records), "checkpoint_config_sha256": sha256(Path(args.config)), "current_val_exploratory_only": True, "official_test_accessed": False}, output_root / "metadata.json", overwrite=True); print(json.dumps({"seeds": cfg["seeds"], "records": len(records), "output": str(output_root), "official_test_accessed": False}, indent=2))


if __name__ == "__main__": main()
