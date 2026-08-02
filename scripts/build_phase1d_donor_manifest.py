"""Build the one-shot class-uniform, track-uniform Phase 1D training donor manifest."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pairs(records: pd.DataFrame, seed: int) -> pd.DataFrame:
    required = {"image_path", "mask_path", "species_id", "group_id", "split"}
    if missing := required - set(records): raise ValueError(f"missing columns: {sorted(missing)}")
    train = records.loc[records.split.eq("train")].copy()
    if train.empty or not set(records.split.unique()) <= {"train", "val", "test"}: raise ValueError("invalid or non-training source records")
    train["_species"] = train.species_id.astype(str); train["_group"] = train.group_id.astype(str)
    species = sorted(train._species.unique())
    # Pre-indexing makes the one-time manifest build linear rather than repeatedly
    # filtering all training rows for every recipient.
    by_species_group = {(key[0], key[1]): value.sort_values("image_path").reset_index(drop=True) for key, value in train.groupby(["_species", "_group"], sort=True)}
    groups_by_species = {label: sorted(train.loc[train._species.eq(label), "_group"].unique()) for label in species}
    rng = np.random.default_rng(seed); output = []
    for recipient in train.sort_values("image_path").to_dict("records"):
        donor_species_options = [label for label in species if label != recipient["_species"]]
        donor_species = donor_species_options[int(rng.integers(len(donor_species_options)))]
        donor_group_options = groups_by_species[donor_species]
        donor_group = donor_group_options[int(rng.integers(len(donor_group_options)))]
        candidates = by_species_group[(donor_species, donor_group)]
        donor = candidates.iloc[int(rng.integers(len(candidates)))]
        output.append({"recipient_image_path": recipient["image_path"], "recipient_mask_path": recipient["mask_path"], "recipient_species_id": recipient["_species"], "recipient_group_id": recipient["_group"], "donor_image_path": donor.image_path, "donor_mask_path": donor.mask_path, "donor_species_id": donor_species, "donor_group_id": donor_group, "pair_seed": seed})
    result = pd.DataFrame(output)
    if result.recipient_species_id.eq(result.donor_species_id).any() or result.recipient_group_id.eq(result.donor_group_id).any(): raise AssertionError("manifest contains an invalid donor")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/cxt_fish_phase1d.yaml"); parser.add_argument("--output", default="splits/f4k16t_phase1d_train_donors.csv"); args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); split = pd.read_csv(cfg["track_split_path"]); metadata = pd.read_csv(cfg["metadata_path"])
    if Path(cfg["track_split_path"]).resolve() == Path(cfg["outer_folds_path"]).resolve(): raise ValueError("outer folds are locked")
    records = metadata.merge(split[["image_path", "split"]], on="image_path", validate="one_to_one")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); build_pairs(records, int(cfg["pair_seed"])).to_csv(output, index=False)
    print(f"manifest={output} sha256={sha256(output)}")


if __name__ == "__main__": main()
