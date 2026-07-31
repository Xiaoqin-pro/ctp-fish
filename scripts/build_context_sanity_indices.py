"""Build deterministic train/validation donor-mask indices for CXT-Fish."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


SALT = "cxt_fish_mask_shuffle_v1"


def _hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def build_pairs(records: pd.DataFrame, partition: str) -> pd.DataFrame:
    frame = records[records.split == partition].copy().sort_values("image_path").reset_index(drop=True)
    if frame.empty: raise ValueError(f"Empty partition: {partition}")
    output = []
    for row in frame.itertuples(index=False):
        start = _hash(f"{SALT}|{row.image_path}") % len(frame)
        donor = None
        for offset in range(len(frame)):
            candidate = frame.iloc[(start + offset) % len(frame)]
            if candidate.group_id != row.group_id:
                donor = candidate; break
        if donor is None: raise ValueError(f"No cross-track donor for {row.image_path}")
        output.append({"recipient_image_path": row.image_path, "recipient_group_id": row.group_id, "donor_image_path": donor.image_path, "donor_mask_path": donor.mask_path, "donor_group_id": donor.group_id, "split": partition, "seed": 407})
    return pd.DataFrame(output)


def _sha(path: Path) -> str:
    digest=hashlib.sha256(); digest.update(path.read_bytes()); return digest.hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-dir", default="outputs/cxt_fish/context_sanity/indices"); args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); metadata=pd.read_csv(cfg["metadata_path"]); split=pd.read_csv(cfg["track_split_path"])
    records=metadata.merge(split[["image_path","split"]], on="image_path", validate="one_to_one")
    root=Path(args.output_dir); root.mkdir(parents=True, exist_ok=True); outputs=[]
    for partition in ("train", "val"):
        destination=root/f"shuffled_mask_pairs_{partition}.csv"; build_pairs(records, partition).to_csv(destination, index=False); outputs.append(destination)
    combined=root/"shuffled_mask_pairs.csv"; pd.concat([pd.read_csv(path) for path in outputs], ignore_index=True).to_csv(combined, index=False)
    manifest={"schema_version":"cxt_fish_shuffled_mask_pairs_v1", "salt":SALT, "seed":407, "combined":{"path":str(combined),"sha256":_sha(combined),"rows":len(pd.read_csv(combined))}, "partitions":{path.stem.replace("shuffled_mask_pairs_", ""): {"path":str(path), "sha256":_sha(path), "rows":len(pd.read_csv(path))} for path in outputs}, "internal_test_read":False, "outer_folds_read":False}
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
