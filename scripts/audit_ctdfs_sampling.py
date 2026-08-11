"""Audit CT-DFS stream exposure without training or model evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.ctdfs_sampler import CTDFSDualStreamSampler


def audit_stream(records, mode: str, seed: int, epoch: int = 1) -> dict:
    sampler = CTDFSDualStreamSampler(records, "s1_track_uniform", mode, seed)
    sampler.set_epoch(epoch)
    pairs = list(iter(sampler))
    original = records.iloc[[pair[0] for pair in pairs]]
    foreground = records.iloc[[pair[1] for pair in pairs]]
    def counts(frame, column):
        return {str(k): int(v) for k, v in frame[column].astype(str).value_counts().sort_index().items()}
    return {
        "mode": mode,
        "samples": len(pairs),
        "original_class_counts": counts(original, "species_id"),
        "foreground_class_counts": counts(foreground, "species_id"),
        "original_track_counts": counts(original, "group_id"),
        "foreground_track_counts": counts(foreground, "group_id"),
        "original_unique_classes": int(original.species_id.nunique()),
        "foreground_unique_classes": int(foreground.species_id.nunique()),
        "original_unique_tracks": int(original.group_id.nunique()),
        "foreground_unique_tracks": int(foreground.group_id.nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cxt_fish_ctdfs.yaml")
    parser.add_argument("--output", default="outputs/cxt_fish/ctdfs_sampling_audit.json")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    metadata = pd.read_csv(cfg["metadata_path"])
    split = pd.read_csv(cfg["track_split_path"])
    train = metadata.merge(split.loc[split.split == "train", ["image_path", "split"]], on="image_path", validate="one_to_one")
    if train.empty or set(train.split) != {"train"}:
        raise ValueError("sampling audit requires train-only records")
    result = {
        "protocol": "ctdfs_phase1_pilot_v1",
        "seed": int(cfg["pilot_seed"]),
        "train_samples": int(len(train)),
        "train_tracks": int(train.group_id.nunique()),
        "p1": audit_stream(train, str(cfg["p1_foreground_sampler"]), int(cfg["pilot_seed"])),
        "p2": audit_stream(train, str(cfg["p2_foreground_sampler"]), int(cfg["pilot_seed"])),
        "internal_test_accessed": False,
        "outer_folds_accessed": False,
    }
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
