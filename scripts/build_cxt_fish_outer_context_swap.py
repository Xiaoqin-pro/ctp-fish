"""Build a frozen context-swap manifest from one outer-test fold only."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SALT = "cxt_fish_outer_context_swap_v1"


def pick(recipient: pd.Series, candidates: pd.DataFrame, kind: str, seed: int):
    if kind == "same_class_cross_track":
        pool = candidates[(candidates.species_id == recipient.species_id) & (candidates.group_id != recipient.group_id)]
    else:
        pool = candidates[(candidates.species_id != recipient.species_id) & (candidates.group_id != recipient.group_id)]
    pool = pool.sort_values("image_path").reset_index(drop=True)
    if pool.empty:
        return None
    idx = int(hashlib.sha256(f"{SALT}|{seed}|{kind}|{recipient.image_path}".encode()).hexdigest(), 16) % len(pool)
    return pool.iloc[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_outer_validation_v1.yaml")
    parser.add_argument("--fold", choices=["1", "2", "3"], required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    root = ROOT / cfg["outer_manifest_output_root"] / f"fold_{args.fold}"
    source = root / "outer_test.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    records = pd.read_csv(source)
    if "mask_path" not in records.columns:
        metadata = pd.read_csv(ROOT / cfg["metadata_path"])
        records = records.drop(columns=["species_id", "species_name", "group_id"], errors="ignore").merge(
            metadata[["image_path", "mask_path", "species_id", "species_name", "group_id"]],
            on="image_path", validate="one_to_one")
    seed = int(cfg["outer_context_swap_manifest_seed"])
    rows = []
    for item in records.sort_values("image_path").itertuples(index=False):
        recipient = pd.Series(item._asdict())
        for kind in ("same_class_cross_track", "cross_class"):
            donor = pick(recipient, records, kind, seed)
            row = {"recipient_image_id": recipient.image_path, "recipient_image_path": recipient.image_path,
                   "recipient_mask_path": recipient.mask_path, "recipient_group_id": recipient.group_id,
                   "recipient_species_id": recipient.species_id, "swap_type": kind, "seed": seed,
                   "supported": donor is not None}
            if donor is not None:
                row.update({"donor_image_id": donor.image_path, "donor_image_path": donor.image_path,
                            "donor_mask_path": donor.mask_path, "donor_group_id": donor.group_id,
                            "donor_species_id": donor.species_id})
            rows.append(row)
    destination = root / "outer_context_swap.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    provenance = {"schema_version": "cxt_fish_outer_context_swap_v1", "fold": args.fold, "seed": seed,
                  "rows": len(rows), "supported_rows": sum(bool(row["supported"]) for row in rows),
                  "manifest_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                  "recipient_source": str(source), "donor_source": str(source),
                  "outer_train_used": False, "inner_dev_used": False, "official_test_accessed": False}
    destination.with_suffix(".manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
