"""Create the deterministic validation-only CXT-Fish Phase 1C swap manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


SALT = "cxt_fish_phase1c_context_swap_v1"


def stable_index(value: str, modulus: int) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16) % modulus


def donor_for(recipient: pd.Series, candidates: pd.DataFrame, kind: str, seed: int) -> pd.Series:
    if kind == "same_class_cross_track":
        pool = candidates[(candidates.species_id == recipient.species_id) & (candidates.group_id != recipient.group_id)]
    elif kind == "cross_class":
        pool = candidates[(candidates.species_id != recipient.species_id) & (candidates.group_id != recipient.group_id)]
    else:
        raise ValueError(f"unknown swap type: {kind}")
    pool = pool.sort_values("image_path").reset_index(drop=True)
    if pool.empty:
        raise ValueError(f"no {kind} donor for {recipient.image_path}")
    return pool.iloc[stable_index(f"{SALT}|{seed}|{kind}|{recipient.image_path}", len(pool))]


def build_manifest(records: pd.DataFrame, seed: int) -> pd.DataFrame:
    if set(records.split.unique()) - {"val"}:
        raise ValueError("Phase 1C swap manifest may contain validation records only")
    output: list[dict] = []
    for recipient in records.sort_values("image_path").reset_index(drop=True).itertuples(index=False):
        row = pd.Series(recipient._asdict())
        for kind in ("same_class_cross_track", "cross_class"):
            donor = donor_for(row, records, kind, seed)
            output.append({
                "recipient_image_id": row.image_path,
                "recipient_image_path": row.image_path,
                "recipient_mask_path": row.mask_path,
                "recipient_group_id": row.group_id,
                "recipient_species_id": row.species_id,
                "donor_image_id": donor.image_path,
                "donor_image_path": donor.image_path,
                "donor_mask_path": donor.mask_path,
                "donor_group_id": donor.group_id,
                "donor_species_id": donor.species_id,
                "swap_type": kind,
                "seed": seed,
            })
    return pd.DataFrame(output)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_phase1c.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    split = pd.read_csv(cfg["track_split_path"])
    if "test" in set(split.split):
        split = split.loc[split.split == "val"]
    metadata = pd.read_csv(cfg["metadata_path"])
    records = metadata.merge(split[["image_path", "split"]], on="image_path", validate="one_to_one")
    if set(records.split) != {"val"}:
        raise ValueError("refusing non-validation Phase 1C manifest")
    manifest = build_manifest(records, int(cfg["swap_manifest_seed"]))
    destination = Path(args.output or cfg["swap_manifest_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(destination, index=False)
    provenance = {
        "schema_version": cfg["swap_schema_version"],
        "manifest_path": str(destination),
        "manifest_sha256": sha256(destination),
        "rows": len(manifest),
        "unique_recipients": manifest.recipient_image_id.nunique(),
        "seed": int(cfg["swap_manifest_seed"]),
        "view_construction": "recipient-mask feathered foreground + donor RGB background; resize and compose before normalization",
        "internal_test_accessed": False,
        "outer_folds_accessed": False,
    }
    destination.with_suffix(".manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
