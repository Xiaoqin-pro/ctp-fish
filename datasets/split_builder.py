"""Deterministic image- and trajectory-level split construction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from tools.hashing import sha256_file


SPLITS = ("train", "val", "test")


def f4k16t_classes(metadata: pd.DataFrame, minimum_tracks: int) -> pd.DataFrame:
    counts = metadata.groupby(["species_id", "species_name"], dropna=False).agg(num_images=("image_path", "size"), num_tracks=("group_id", "nunique")).reset_index()
    counts["included"] = counts["num_tracks"] >= minimum_tracks
    counts["exclusion_reason"] = np.where(counts["included"], "", f"fewer_than_{minimum_tracks}_tracks")
    return counts.sort_values(["species_id"]).reset_index(drop=True)


def _split_labels(indices: np.ndarray, labels: np.ndarray, seed: int, fractions: tuple[float, float, float]) -> dict[str, np.ndarray]:
    train_f, val_f, test_f = fractions
    if not np.isclose(train_f + val_f + test_f, 1.0):
        raise ValueError("Split fractions must sum to 1.")
    train_idx, held_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=val_f + test_f, random_state=seed).split(indices, labels))
    held_labels = labels[held_idx]
    val_relative = val_f / (val_f + test_f)
    val_local, test_local = next(StratifiedShuffleSplit(n_splits=1, test_size=1 - val_relative, random_state=seed + 1).split(held_idx, held_labels))
    return {"train": indices[train_idx], "val": indices[held_idx[val_local]], "test": indices[held_idx[test_local]]}


def build_image_level_split(metadata: pd.DataFrame, seed: int, fractions: tuple[float, float, float]) -> pd.DataFrame:
    indices = np.arange(len(metadata))
    parts = _split_labels(indices, metadata["species_id"].to_numpy(), seed, fractions)
    result = metadata[["image_path", "species_id", "species_name", "group_id"]].copy()
    result["split"] = ""
    for name, positions in parts.items(): result.loc[positions, "split"] = name
    return result


def build_track_level_split(metadata: pd.DataFrame, seed: int, fractions: tuple[float, float, float]) -> pd.DataFrame:
    groups = metadata[["group_id", "species_id"]].drop_duplicates().sort_values("group_id").reset_index(drop=True)
    if (groups.groupby("species_id").size() < 3).any():
        raise ValueError("Each retained class needs at least three trajectory groups.")
    parts = _split_labels(np.arange(len(groups)), groups["species_id"].to_numpy(), seed, fractions)
    groups["split"] = ""
    for name, positions in parts.items(): groups.loc[positions, "split"] = name
    result = metadata[["image_path", "species_id", "species_name", "group_id"]].merge(groups[["group_id", "split"]], on="group_id", validate="many_to_one")
    validate_track_split(result)
    return result


def validate_track_split(split: pd.DataFrame) -> None:
    crossing = split.groupby("group_id")["split"].nunique()
    if (crossing > 1).any(): raise ValueError("A group_id crosses trajectory-level splits.")
    presence = split.groupby(["species_id", "split"])["group_id"].nunique().unstack(fill_value=0)
    if not set(SPLITS).issubset(presence.columns) or (presence.loc[:, list(SPLITS)] < 1).any().any(): raise ValueError("Each class must have a group in every split.")


def build_outer_folds(metadata: pd.DataFrame, seed: int, n_splits: int = 3) -> dict:
    groups = metadata[["group_id", "species_id"]].drop_duplicates().sort_values("group_id").reset_index(drop=True)
    counts = groups.groupby("species_id").size()
    if (counts < n_splits).any(): raise ValueError("Not enough trajectory groups per class for outer folds.")
    folds = {}
    for fold, (_, test_positions) in enumerate(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(groups, groups["species_id"]), start=1): folds[str(fold)] = groups.iloc[test_positions]["group_id"].tolist()
    all_ids = [item for values in folds.values() for item in values]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(groups["group_id"]): raise RuntimeError("Outer test folds must partition all groups.")
    return {"locked": True, "split_seed": seed, "outer_folds": folds}


def write_outer_folds(payload: dict, path: Path, overwrite: bool = False) -> str:
    if path.exists() and not overwrite: raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return sha256_file(path)
