"""Validate the single F4K-16T class vocabulary without model evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-ids", default="splits/f4k16t_class_ids.json")
    parser.add_argument("--metadata", default="outputs/gate0/audit/f4k_metadata.csv")
    parser.add_argument("--split", default="splits/f4k16t_track_level_dev.csv")
    args = parser.parse_args()
    protocol = json.loads(Path(args.class_ids).read_text(encoding="utf-8"))
    ids = [str(value) for value in protocol["class_ids"]]
    if len(ids) != 16 or len(set(ids)) != 16:
        raise ValueError("class protocol is not exactly 16 unique IDs")
    metadata = pd.read_csv(args.metadata, usecols=["image_path", "species_id"])
    split = pd.read_csv(args.split, usecols=["image_path", "split"])
    records = metadata.merge(split, on="image_path", validate="one_to_one")
    for partition in ("train", "val"):
        found = set(records.loc[records.split == partition, "species_id"].astype(str))
        if found != set(ids):
            raise ValueError(f"{partition} class set does not match frozen protocol")
    print(json.dumps({"class_ids": ids, "class_count": 16, "train_val_match": True}, indent=2))


if __name__ == "__main__":
    main()
