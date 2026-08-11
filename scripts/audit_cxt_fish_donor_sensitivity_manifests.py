"""Validate all fixed donor-sensitivity manifests before any inference."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (4101, 4102, 4103, 4104, 4105)


def main() -> None:
    root = ROOT / "outputs/cxt_fish/donor_sensitivity/manifests"
    checked = []
    for seed in SEEDS:
        for fold in (1, 2, 3):
            path = root / f"seed_{seed}" / f"fold_{fold}" / "outer_context_swap.csv"
            payload_path = path.with_suffix(".manifest.json")
            manifest = pd.read_csv(path, dtype=str)
            outer = pd.read_csv(ROOT / f"outputs/cxt_fish/final_outer_manifests/fold_{fold}/outer_test.csv", dtype=str)
            if len(manifest) != 2 * outer["image_path"].nunique():
                raise AssertionError(f"{path}: row count mismatch")
            if set(manifest["recipient_image_path"]) != set(outer["image_path"]):
                raise AssertionError(f"{path}: recipients differ from frozen outer-test")
            if manifest["supported"].astype(str).str.lower().ne("true").any():
                raise AssertionError(f"{path}: unsupported row")
            for recipient, rows in manifest.groupby("recipient_image_path"):
                if len(rows) != 2 or set(rows["swap_type"]) != {"same_class_cross_track", "cross_class"}:
                    raise AssertionError(f"{path}: invalid rows for {recipient}")
                if (rows["recipient_group_id"] == rows["donor_group_id"]).any():
                    raise AssertionError(f"{path}: recipient/donor group overlap")
                same = rows[rows["swap_type"] == "same_class_cross_track"].iloc[0]
                cross = rows[rows["swap_type"] == "cross_class"].iloc[0]
                if same["recipient_species_id"] != same["donor_species_id"]:
                    raise AssertionError(f"{path}: same-class mismatch")
                if cross["recipient_species_id"] == cross["donor_species_id"]:
                    raise AssertionError(f"{path}: cross-class mismatch")
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            if payload.get("official_test_accessed") is not False or payload.get("post_hoc_sensitivity") is not True:
                raise AssertionError(f"{payload_path}: invalid provenance")
            checked.append({"seed": seed, "fold": fold, "rows": len(manifest), "path": str(path)})
    print(json.dumps({"status": "complete", "manifests_checked": len(checked), "checked": checked}, indent=2))


if __name__ == "__main__":
    main()
