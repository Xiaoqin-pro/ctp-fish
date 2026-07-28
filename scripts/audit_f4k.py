"""Run strict Fish4Knowledge discovery and stop if any image cannot be parsed."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import yaml
from datasets.f4k_metadata import compile_filename_parser, discover_images, parse_image
from datasets.split_builder import f4k16t_classes
from tools.io_utils import atomic_csv_dump, atomic_json_dump

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text()); root = Path(config["data_root"])
    if not root.exists(): raise FileNotFoundError(f"Configured data_root does not exist: {root}")
    images = discover_images(root, config["image_extensions"])
    filename_parser = compile_filename_parser(config.get("filename_regex"))
    records, unresolved = [], []
    for image in images:
        try:
            record = parse_image(image, filename_parser)
            records.append(record.__dict__ | {"group_id": record.group_id})
        except Exception as exc: unresolved.append({"path": str(image), "reason": str(exc)})
    output = Path(config["output_dir"])
    if args.dry_run:
        print(json.dumps({"images_discovered": len(images), "unresolved": len(unresolved)}, indent=2)); return
    atomic_csv_dump(pd.DataFrame(unresolved), output / "unresolved_files.csv", args.overwrite)
    if unresolved: raise RuntimeError(f"{len(unresolved)} unresolved files; see {output / 'unresolved_files.csv'}")
    metadata = pd.DataFrame(records); atomic_csv_dump(metadata, output / "f4k_metadata.csv", args.overwrite)
    classes = f4k16t_classes(metadata, int(config["min_tracks_per_class"])); atomic_csv_dump(classes, output / "f4k16t_classes.csv", args.overwrite)
    class_stats = metadata.groupby(["species_id", "species_name"], as_index=False).agg(num_images=("image_path", "size"), num_tracks=("group_id", "nunique")); atomic_csv_dump(class_stats, output / "class_statistics.csv", args.overwrite)
    track_stats = metadata.groupby(["group_id", "species_id"], as_index=False).size().rename(columns={"size": "frames"}); atomic_csv_dump(track_stats, output / "track_statistics.csv", args.overwrite)
    summary = {"num_images": len(metadata), "num_classes": int(metadata.species_id.nunique()), "num_tracks": int(metadata.group_id.nunique()), "tracking_id_cross_species": bool(metadata.groupby("tracking_id_raw").species_id.nunique().gt(1).any()), "duplicate_sha_rows": int(metadata.file_sha256.duplicated(keep=False).sum()), "mask_missing_rate": 1.0}
    atomic_json_dump(summary, output / "audit_summary.json", args.overwrite)
    print(json.dumps(summary, indent=2))
if __name__ == "__main__": main()
