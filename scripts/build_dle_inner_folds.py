"""Build deterministic two-fold, track-disjoint development folds from train only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_ctdfs.yaml")
    parser.add_argument("--output", default="splits/f4k16t_phase2_inner_2fold.json")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    protocol = json.loads(Path(cfg["class_ids_path"]).read_text(encoding="utf-8"))
    class_ids = [str(v) for v in protocol["class_ids"]]
    metadata = pd.read_csv(cfg["metadata_path"])
    split = pd.read_csv(cfg["track_split_path"])
    if set(split.split.unique()) - {"train", "val", "test"}:
        raise ValueError("unexpected split name")
    train = metadata.merge(split.loc[split.split == "train", ["image_path", "split"]], on="image_path", validate="one_to_one")
    if set(train.species_id.astype(str)) != set(class_ids):
        raise ValueError("train classes do not match frozen class protocol")
    groups = train.groupby([train.species_id.astype(str), "group_id"], sort=True).agg(image_count=("image_path", "size")).reset_index()
    for species, subset in groups.groupby("species_id", sort=True):
        if len(subset) < 2:
            raise ValueError(f"class {species} has fewer than two train tracks")
    assignments = {0: [], 1: []}; track_counts = {0: {c: 0 for c in class_ids}, 1: {c: 0 for c in class_ids}}; image_counts = {0: 0, 1: 0}
    for species, subset in groups.groupby("species_id", sort=True):
        for row in subset.sort_values(["image_count", "group_id"], ascending=[False, True]).itertuples(index=False):
            fold = min((0, 1), key=lambda f: (track_counts[f][species], image_counts[f], f))
            assignments[fold].append(str(row.group_id)); track_counts[fold][species] += 1; image_counts[fold] += int(row.image_count)
    if set(assignments[0]) & set(assignments[1]): raise AssertionError("group overlap")
    payload = {"schema_version": "f4k16t_phase2_inner_2fold_v1", "class_ids": class_ids, "source_split": cfg["track_split_path"], "source_partition": "train", "folds": [], "official_test_accessed": False, "internal_test_accessed": False, "outer_folds_accessed": False}
    group_species = {str(row.group_id): str(row.species_id) for row in groups.itertuples(index=False)}
    for fold in (0, 1):
        selected = set(assignments[fold]); part = train[train.group_id.astype(str).isin(selected)]
        payload["folds"].append({"fold": fold, "heldout_group_ids": sorted(selected), "image_count": int(len(part)), "class_track_counts": {c: int(sum(1 for g in selected if group_species[g] == c)) for c in class_ids}, "class_image_counts": {c: int((part.species_id.astype(str) == c).sum()) for c in class_ids}})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "folds": [(x["image_count"], min(x["class_track_counts"].values())) for x in payload["folds"]]}, indent=2))


if __name__ == "__main__": main()
