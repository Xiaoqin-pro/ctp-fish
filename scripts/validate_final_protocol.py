"""Validate the frozen CXT-Fish protocol without training or model inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cxt_fish_outer_validation_v1.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    expected = {
        "class_ids_path": "class_ids_sha256",
        "metadata_path": "metadata_sha256",
        "track_level_dev_path": "track_level_dev_sha256",
        "outer_folds_path": "outer_folds_sha256",
        "context_swap_manifest_path": "context_swap_manifest_sha256",
    }
    for path_key, hash_key in expected.items():
        path = ROOT / cfg[path_key]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != cfg[hash_key]:
            raise RuntimeError(f"{path_key} hash mismatch: {actual} != {cfg[hash_key]}")

    class_payload = json.loads((ROOT / cfg["class_ids_path"]).read_text(encoding="utf-8"))
    class_ids = class_payload.get("class_ids", class_payload)
    if len(class_ids) != 16 or len(set(map(str, class_ids))) != 16:
        raise RuntimeError("Expected exactly 16 unique frozen class IDs.")

    split = pd.read_csv(ROOT / cfg["track_level_dev_path"])
    required = {"image_path", "species_id", "group_id", "split"}
    if not required.issubset(split.columns):
        raise RuntimeError(f"Missing split columns: {sorted(required - set(split.columns))}")
    if split.image_path.duplicated().any() or split.group_id.isna().any():
        raise RuntimeError("Split contains duplicate images or missing group IDs.")
    groups = split.groupby("split")["group_id"].apply(set).to_dict()
    for left in groups:
        for right in groups:
            if left < right and groups[left] & groups[right]:
                raise RuntimeError(f"Group overlap between {left} and {right}.")

    outer = json.loads((ROOT / cfg["outer_folds_path"]).read_text(encoding="utf-8"))
    if outer.get("locked") is not True or set(outer.get("outer_folds", {})) != {"1", "2", "3"}:
        raise RuntimeError("Outer manifest is not the locked three-fold manifest.")
    folds = [set(map(str, values)) for values in outer["outer_folds"].values()]
    if any(not fold for fold in folds) or sum(map(len, folds)) != len(set().union(*folds)):
        raise RuntimeError("Outer folds overlap or contain an empty fold.")
    all_groups = set(map(str, split.group_id.unique()))
    if set().union(*folds) != all_groups:
        raise RuntimeError("Outer folds do not cover the frozen trajectory universe exactly.")

    print(json.dumps({
        "status": "protocol_valid",
        "class_count": len(class_ids),
        "development_rows": int(len(split)),
        "development_splits": {str(k): int(v) for k, v in split.groupby("split").size().items()},
        "outer_fold_group_counts": [len(fold) for fold in folds],
        "training_started": False,
        "internal_test_accessed_before_outer_confirmation": False,
        "official_test_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
