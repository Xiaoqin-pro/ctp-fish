"""Build deterministic outer train/dev/test manifests from frozen trajectory IDs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cxt_fish_outer_validation_v1.yaml"


def stable_key(seed: int, fold: str, group_id: str) -> str:
    return hashlib.sha256(f"{seed}:{fold}:{group_id}".encode("utf-8")).hexdigest()


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    split_records = pd.read_csv(ROOT / cfg["track_level_dev_path"])
    metadata = pd.read_csv(ROOT / cfg["metadata_path"])
    records = metadata.merge(split_records[["image_path", "split"]], on="image_path", validate="one_to_one")
    outer = json.loads((ROOT / cfg["outer_folds_path"]).read_text(encoding="utf-8"))
    output_root = ROOT / cfg["outer_manifest_output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    seed = int(outer["split_seed"])
    fraction = float(cfg["inner_dev_fraction"])
    if not 0.0 < fraction < 0.5:
        raise ValueError("inner_dev_fraction must be in (0, 0.5).")

    all_groups = set(records["group_id"].astype(str))
    manifest = {"protocol": cfg["protocol_version"], "split_seed": seed, "folds": {}}
    for fold, values in sorted(outer["outer_folds"].items()):
        test_groups = set(map(str, values))
        if test_groups - all_groups:
            raise ValueError(f"Fold {fold} contains unknown groups.")
        train_groups = all_groups - test_groups
        dev_groups: set[str] = set()
        source = records[records["group_id"].astype(str).isin(train_groups)]
        for species, species_frame in source.groupby("species_id", sort=True):
            species_groups = sorted(set(species_frame["group_id"].astype(str)), key=lambda group: stable_key(seed, f"{fold}:{species}", group))
            if len(species_groups) < 2:
                raise ValueError(f"Fold {fold}, species {species} has fewer than two outer-train trajectories.")
            dev_count = min(max(1, round(len(species_groups) * fraction)), len(species_groups) - 1)
            dev_groups.update(species_groups[:dev_count])
        fit_groups = train_groups - dev_groups
        if not fit_groups or not dev_groups:
            raise ValueError(f"Fold {fold} has an empty inner split.")
        if fit_groups & dev_groups or fit_groups & test_groups or dev_groups & test_groups:
            raise AssertionError(f"Fold {fold} group overlap.")

        fold_root = output_root / f"fold_{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        group_series = records["group_id"].astype(str)
        subsets = {
            "outer_train": records[group_series.isin(fit_groups)],
            "inner_dev": records[group_series.isin(dev_groups)],
            "outer_test": records[group_series.isin(test_groups)],
        }
        for name, frame in subsets.items():
            frame = frame.copy()
            frame["source_split"] = frame["split"]
            frame["split"] = "train" if name == "outer_train" else ("val" if name == "inner_dev" else "test")
            frame.sort_values(["group_id", "image_path"]).to_csv(fold_root / f"{name}.csv", index=False)
        manifest["folds"][str(fold)] = {
            "outer_train_groups": len(fit_groups),
            "inner_dev_groups": len(dev_groups),
            "outer_test_groups": len(test_groups),
            "outer_train_rows": len(subsets["outer_train"]),
            "inner_dev_rows": len(subsets["inner_dev"]),
            "outer_test_rows": len(subsets["outer_test"]),
            "group_hash_seed": seed,
        }

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "outer_manifests_written", "output_root": str(output_root), "folds": manifest["folds"]}, indent=2))


if __name__ == "__main__":
    main()
