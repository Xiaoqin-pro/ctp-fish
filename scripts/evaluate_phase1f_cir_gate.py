"""Zero-training CIR-Fish gate on the frozen Phase 1C F1 checkpoint.

The gate tests whether a donor's detached context-response residual can predict
the effect of a real cross-class context intervention.  It never trains, reads
the internal test, calibration, or outer folds, and does not write checkpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.context_views import context_swap_view, foreground_blur_view
from models.resnet_context import ResNet18Context
from models.resnet_classifier import build_resnet18
from tools.io_utils import atomic_csv_dump, atomic_json_dump


MEAN = [.485, .456, .406]
STD = [.229, .224, .225]


def evaluation_transform(size: int):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_model(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    class_ids = [str(value) for value in state["class_ids"]]
    keys = state["model"].keys()
    model = ResNet18Context(len(class_ids)) if any(k.startswith("encoder.") for k in keys) else build_resnet18(len(class_ids))
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state, class_ids


class ViewDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform, cfg: dict, *, include_swap: bool = False) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.transform = transform
        self.cfg = cfg
        self.include_swap = include_swap

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        image = Image.open(row.image_path).convert("RGB")
        mask = Image.open(row.mask_path).convert("L")
        fg = foreground_blur_view(
            image,
            mask,
            blur_kernel=21,
            blur_sigma=5.0,
            feather_radius=int(self.cfg["mask_feather_radius"]),
        )
        views = [self.transform(image), self.transform(fg)]
        if self.include_swap:
            donor = Image.open(row.donor_image_path).convert("RGB")
            swapped = context_swap_view(image, mask, donor, feather_radius=int(self.cfg["mask_feather_radius"]))
            views.append(self.transform(swapped))
        return tuple(views) + (str(row.image_path), str(row.group_id))


def predict_logits(model, loader, device: torch.device, view_count: int) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {name: [] for name in ("original", "foreground", "swap")[:view_count]}
    paths: list[str] = []
    groups: list[str] = []
    with torch.no_grad():
        for batch in loader:
            *images, batch_paths, batch_groups = batch
            for name, image in zip(values, images):
                output = model(image.to(device, non_blocking=True))
                logits = output[0] if isinstance(output, tuple) else output
                values[name].append(logits.float().cpu().numpy())
            paths.extend(batch_paths)
            groups.extend(batch_groups)
    values["image_path"] = np.asarray(paths, dtype=object)
    values["group_id"] = np.asarray(groups, dtype=object)
    return {
        key: np.concatenate(value) if isinstance(value, list) and value and key not in {"image_path", "group_id"} else value
        for key, value in values.items()
    }


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64)
    if np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def cluster_bootstrap_difference(frame: pd.DataFrame, value_col: str, reps: int, seed: int) -> dict:
    groups = np.asarray(sorted(frame.group_id.unique()), dtype=object)
    rng = np.random.default_rng(seed)
    true = frame["true_donor_attraction"].to_numpy(dtype=float)
    shuffled = frame["shuffle_donor_attraction"].to_numpy(dtype=float)
    by_group = {group: np.flatnonzero(frame.group_id.to_numpy() == group) for group in groups}
    differences = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([by_group[group] for group in sampled])
        differences[index] = true[rows].mean() - shuffled[rows].mean()
    return {
        "replicates": int(reps),
        "seed": int(seed),
        "mean": float(differences.mean()),
        "ci95": [float(np.quantile(differences, .025)), float(np.quantile(differences, .975))],
        "cluster_count": int(len(groups)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_phase1f.yaml")
    parser.add_argument("--output", default="outputs/cxt_fish/phase1f_cir_gate")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if cfg["locked_partition"] != "val" or set(cfg["prohibited_partitions"]) != {"train", "test", "outer_folds"}:
        raise ValueError("CIR gate partition lock is invalid")
    metadata = pd.read_csv(cfg["metadata_path"])
    split = pd.read_csv(cfg["track_split_path"])
    manifest = pd.read_csv(cfg["swap_manifest_path"])
    # The frozen split CSV may declare other partitions.  The gate must select
    # validation rows before any image path is opened; merely parsing the
    # declaration is not evaluation access.
    if "val" not in set(split.split.unique()):
        raise ValueError("split file has no validation partition")
    records = metadata.merge(split.loc[split.split == "val", ["image_path", "split"]], on="image_path", validate="one_to_one")
    cross = manifest.loc[manifest.swap_type == "cross_class"].copy()
    if cross.empty or cross.recipient_image_id.nunique() != len(records) or len(cross) != len(records):
        raise ValueError("CIR gate requires exactly one cross-class donor per validation recipient")
    if (cross.recipient_species_id.astype(str).to_numpy() == cross.donor_species_id.astype(str).to_numpy()).any():
        raise ValueError("cross-class manifest contains same-class pairs")
    records["recipient_image_id"] = records.image_path.astype(str)
    pairs = records.merge(cross, left_on="recipient_image_id", right_on="recipient_image_id", validate="one_to_one", suffixes=("", "_donor"))
    if len(pairs) != len(records) or set(pairs.split) != {"val"}:
        raise ValueError("CIR gate pair set is not exactly validation-only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state, class_ids = load_model(Path(cfg["checkpoint_path"]), device)
    class_to_index = {label: index for index, label in enumerate(class_ids)}
    pairs["target_idx"] = pairs.species_id.astype(str).map(class_to_index)
    pairs["donor_idx"] = pairs.donor_species_id.astype(str).map(class_to_index)
    if pairs[["target_idx", "donor_idx"]].isna().any().any():
        raise ValueError("checkpoint class IDs do not cover the validation manifest")
    transform = evaluation_transform(int(cfg["image_size"]))
    unique = pd.concat([
        pairs[["image_path", "mask_path", "group_id"]],
        pairs[["donor_image_path", "donor_mask_path", "donor_group_id"]].rename(columns={"donor_image_path": "image_path", "donor_mask_path": "mask_path", "donor_group_id": "group_id"}),
    ], ignore_index=True).drop_duplicates("image_path")
    workers = int(cfg["num_workers"] if args.num_workers is None else args.num_workers)
    loader = DataLoader(ViewDataset(unique, transform, cfg), batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    cache = predict_logits(model, loader, device, 2)
    cache_frame = pd.DataFrame({"image_path": cache["image_path"], "group_id": cache["group_id"]})
    cache_frame["row"] = np.arange(len(cache_frame))
    lookup = cache_frame.set_index("image_path")["row"].to_dict()
    rec_rows = np.asarray([lookup[str(path)] for path in pairs.image_path], dtype=np.int64)
    donor_rows = np.asarray([lookup[str(path)] for path in pairs.donor_image_path], dtype=np.int64)
    rec_orig, rec_fg = cache["original"][rec_rows], cache["foreground"][rec_rows]
    donor_orig, donor_fg = cache["original"][donor_rows], cache["foreground"][donor_rows]
    donor_residual = donor_orig - donor_fg
    injected = rec_fg + donor_residual
    permutation = np.random.default_rng(int(cfg["gate_seed"])).permutation(len(pairs))
    shuffled_injected = rec_fg + donor_residual[permutation]
    swap_loader = DataLoader(ViewDataset(pairs[["image_path", "mask_path", "group_id", "donor_image_path"]], transform, cfg, include_swap=True), batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    swap_cache = predict_logits(model, swap_loader, device, 3)
    real_swap = swap_cache["swap"]
    target = pairs.target_idx.to_numpy(dtype=np.int64)
    donor = pairs.donor_idx.to_numpy(dtype=np.int64)
    def select(logits, indices): return logits[np.arange(len(logits)), indices]
    fg_pred = rec_fg.argmax(1)
    orig_pred = rec_orig.argmax(1)
    swap_pred = real_swap.argmax(1)
    inj_pred = injected.argmax(1)
    shuffle_pred = shuffled_injected.argmax(1)
    residual_scores = donor_residual - donor_residual.mean(1, keepdims=True)
    one_vs_rest = []
    for cls in range(len(class_ids)):
        one_vs_rest.append({"class_id": class_ids[cls], "auroc": safe_auc((donor == cls).astype(np.int64), residual_scores[:, cls])})
    donor_specificity_auc = safe_auc(np.eye(len(class_ids))[donor].ravel(), residual_scores.ravel())
    q_cir = select(donor_residual, donor) - select(donor_residual, target)
    q_real = (select(real_swap, donor) - select(real_swap, target)) - (select(rec_fg, donor) - select(rec_fg, target))
    real_darflip = (orig_pred == target) & (swap_pred == donor)
    cir_darflip_auc = safe_auc(real_darflip.astype(np.int64), q_cir)
    corr = spearmanr(q_cir, q_real)
    rows = pd.DataFrame({
        "image_path": pairs.image_path.astype(str), "group_id": pairs.group_id.astype(str), "target": target, "donor": donor,
        "foreground_correct": fg_pred == target, "original_correct": orig_pred == target, "real_swap_darflip": real_darflip,
        "true_injected_recipient_correct": inj_pred == target, "true_injected_donor_attraction": inj_pred == donor,
        "shuffle_injected_recipient_correct": shuffle_pred == target, "shuffle_injected_donor_attraction": shuffle_pred == donor,
        "q_cir": q_cir, "q_real": q_real,
    })
    rows["true_donor_attraction"] = rows.true_injected_donor_attraction.astype(float)
    rows["shuffle_donor_attraction"] = rows.shuffle_injected_donor_attraction.astype(float)
    bootstrap = cluster_bootstrap_difference(rows, "true_donor_attraction", int(cfg["bootstrap_replicates"]), int(cfg["bootstrap_seed"]))
    valid_pair_fraction = float(np.isfinite(q_cir).mean())
    per_class_finite = [item["auroc"] for item in one_vs_rest if item["auroc"] is not None]
    positive_classes = sum(value >= float(cfg["thresholds"]["residual_donor_auroc"]) for value in per_class_finite)
    result = {
        "gate_version": cfg["gate_version"], "checkpoint": str(cfg["checkpoint_path"]), "checkpoint_variant": state.get("variant", "unknown"),
        "partition": "val", "internal_test_accessed": False, "outer_folds_accessed": False, "train_used_for_fitting": False,
        "pair_count": int(len(rows)), "unique_recipient_count": int(pairs.recipient_image_id.nunique()), "unique_track_count": int(rows.group_id.nunique()),
        "valid_pair_fraction": valid_pair_fraction, "residual_donor_auroc": donor_specificity_auc, "per_class_residual_auroc": one_vs_rest,
        "positive_class_count_at_threshold": int(positive_classes), "class_count": int(len(class_ids)),
        "foreground_injected_recipient_accuracy": float(rows.true_injected_recipient_correct.mean()), "shuffle_injected_recipient_accuracy": float(rows.shuffle_injected_recipient_correct.mean()),
        "foreground_injected_donor_attraction": float(rows.true_donor_attraction.mean()), "shuffle_injected_donor_attraction": float(rows.shuffle_donor_attraction.mean()),
        "true_minus_shuffle_donor_attraction": float(rows.true_donor_attraction.mean() - rows.shuffle_donor_attraction.mean()),
        "true_minus_shuffle_bootstrap": bootstrap, "cir_darflip_auroc": cir_darflip_auc,
        "q_cir_q_real_spearman": None if not np.isfinite(corr.statistic) else float(corr.statistic),
        "q_cir_q_real_spearman_pvalue": None if not np.isfinite(corr.pvalue) else float(corr.pvalue),
        "real_darflip_count": int(real_darflip.sum()), "real_darflip_fraction": float(real_darflip.mean()),
    }
    t = cfg["thresholds"]
    result["gate_checks"] = {
        "residual_donor_auroc": result["residual_donor_auroc"] is not None and result["residual_donor_auroc"] >= float(t["residual_donor_auroc"]),
        "cir_darflip_auroc": result["cir_darflip_auroc"] is not None and result["cir_darflip_auroc"] >= float(t["cir_darflip_auroc"]),
        "real_shuffle_ci_lower": bootstrap["ci95"][0] > float(t["residual_real_shuffle_ci_lower"]),
        "positive_class_fraction": positive_classes >= int(np.ceil(float(t["min_positive_classes_fraction"]) * len(class_ids))),
        "valid_pair_fraction": valid_pair_fraction >= float(t["min_valid_pair_fraction"]),
        "q_alignment_positive": result["q_cir_q_real_spearman"] is not None and result["q_cir_q_real_spearman"] > 0,
    }
    result["gate_pass"] = bool(all(result["gate_checks"].values()))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    atomic_csv_dump(rows, output / "cir_gate_per_pair.csv", args.overwrite)
    atomic_json_dump(result, output / "cir_gate_summary.json", args.overwrite)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
