from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import yaml
from datasets.split_builder import build_image_level_split, build_outer_folds, build_track_level_split, write_outer_folds
from tools.hashing import sha256_file
from tools.io_utils import atomic_csv_dump, atomic_json_dump

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--overwrite",action="store_true"); args=parser.parse_args(); cfg=yaml.safe_load(Path(args.config).read_text())
    metadata=pd.read_csv(cfg["metadata_path"]); classes=pd.read_csv(cfg["f4k16t_classes_path"]); included=set(classes.loc[classes.included.astype(bool),"species_id"].astype(str)); metadata=metadata[metadata.species_id.astype(str).isin(included)].reset_index(drop=True)
    fractions=(float(cfg["train_fraction"]),float(cfg["val_fraction"]),float(cfg["test_fraction"])); seed=int(cfg["split_seed"]); out=Path(cfg["output_dir"])
    if args.dry_run: print({"retained_images":len(metadata),"retained_classes":len(included)}); return
    image=build_image_level_split(metadata,seed,fractions); track=build_track_level_split(metadata,seed,fractions)
    atomic_csv_dump(image,out/"f4k16t_image_level_dev.csv",args.overwrite); atomic_csv_dump(track,out/"f4k16t_track_level_dev.csv",args.overwrite)
    fold_path=out/"f4k16t_outer_3fold.json"; fold_sha=write_outer_folds(build_outer_folds(metadata,seed,int(cfg["outer_folds"])),fold_path,args.overwrite)
    atomic_json_dump({"image_split_sha256":sha256_file(out/"f4k16t_image_level_dev.csv"),"track_split_sha256":sha256_file(out/"f4k16t_track_level_dev.csv"),"outer_folds_sha256":fold_sha},out/"split_hashes.json",args.overwrite)
if __name__=="__main__": main()
