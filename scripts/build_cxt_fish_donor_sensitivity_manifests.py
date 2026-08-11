"""Build post-hoc donor-realization manifests without changing the primary ones."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np

from build_cxt_fish_outer_context_swap import SALT

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (4101, 4102, 4103, 4104, 4105)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_one(fold: int, seed: int, source_root: Path, output_root: Path, parent_commit: str) -> dict:
    source = source_root / f"fold_{fold}" / "outer_test.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    records = pd.read_csv(source, dtype=str).sort_values("image_path").reset_index(drop=True)
    paths = records["image_path"].to_numpy()
    groups = records["group_id"].to_numpy()
    species = records["species_id"].to_numpy()
    rows: list[dict] = []
    for row_index, item in enumerate(records.itertuples(index=False)):
        recipient = pd.Series(item._asdict())
        for kind in ("same_class_cross_track", "cross_class"):
            if kind == "same_class_cross_track":
                eligible = (species == recipient.species_id) & (groups != recipient.group_id)
            else:
                eligible = (species != recipient.species_id) & (groups != recipient.group_id)
            candidate_indices = np.flatnonzero(eligible)
            donor = None
            if len(candidate_indices):
                digest = hashlib.sha256(f"{SALT}|{seed}|{kind}|{recipient.image_path}".encode()).hexdigest()
                donor = records.iloc[candidate_indices[int(int(digest, 16) % len(candidate_indices))]]
            row = {
                "recipient_image_id": recipient.image_path,
                "recipient_image_path": recipient.image_path,
                "recipient_mask_path": recipient.mask_path,
                "recipient_group_id": recipient.group_id,
                "recipient_species_id": recipient.species_id,
                "swap_type": kind,
                "seed": seed,
                "supported": donor is not None,
            }
            if donor is not None:
                row.update({
                    "donor_image_id": donor.image_path,
                    "donor_image_path": donor.image_path,
                    "donor_mask_path": donor.mask_path,
                    "donor_group_id": donor.group_id,
                    "donor_species_id": donor.species_id,
                })
            rows.append(row)
    destination = output_root / f"seed_{seed}" / f"fold_{fold}" / "outer_context_swap.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    payload = {
        "schema_version": "cxt_fish_donor_sensitivity_v1",
        "parent_commit": parent_commit,
        "post_hoc_sensitivity": True,
        "fold": fold,
        "seed": seed,
        "rows": len(rows),
        "unique_recipients": int(pd.DataFrame(rows)["recipient_image_path"].nunique()),
        "sha256": sha256(destination),
        "recipient_source_sha256": sha256(source),
        "candidate_pool_source": str(source),
        "official_test_accessed": False,
        "primary_manifest_modified": False,
    }
    destination.with_suffix(".manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="outputs/cxt_fish/final_outer_manifests")
    parser.add_argument("--output-root", default="outputs/cxt_fish/donor_sensitivity/manifests")
    parser.add_argument("--parent-commit", default="0332cd9bf580cde12e5f82587dd7cae9dcba3b6b")
    args = parser.parse_args()
    source_root, output_root = ROOT / args.source_root, ROOT / args.output_root
    payloads = [build_one(fold, seed, source_root, output_root, args.parent_commit) for seed in SEEDS for fold in (1, 2, 3)]
    print(json.dumps({"status": "complete", "manifests": payloads}, indent=2))


if __name__ == "__main__":
    main()
