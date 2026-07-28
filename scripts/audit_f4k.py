"""Run strict Fish4Knowledge discovery and stop if any image cannot be parsed."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from datasets.f4k_metadata import compile_filename_parser, discover_images, parse_image, scan_f4k_groundtruth
from datasets.split_builder import f4k16t_classes
from tools.io_utils import atomic_csv_dump, atomic_json_dump


def _plot_audit(class_stats: pd.DataFrame, track_stats: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frames = track_stats.frames.to_numpy()
    charts = [
        ("class_image_counts.png", class_stats, "species_name", "num_images", "Images per class"),
        ("class_track_counts.png", class_stats, "species_name", "num_tracks", "Trajectories per class"),
    ]
    for name, frame, category, value, title in charts:
        axis=frame.sort_values(value).plot.barh(x=category,y=value,legend=False,figsize=(9,6),title=title); axis.figure.tight_layout(); axis.figure.savefig(output/name,dpi=160); plt.close(axis.figure)
    axis=plt.figure(figsize=(8,5)).gca(); axis.hist(frames,bins=50); axis.set(xlabel="Frames per trajectory",ylabel="Tracks",title="Trajectory length distribution"); axis.figure.tight_layout(); axis.figure.savefig(output/"track_length_distribution.png",dpi=160); plt.close(axis.figure)
    axis=class_stats.plot.scatter(x="num_tracks",y="num_images",figsize=(7,5),title="Images versus trajectories per class"); axis.figure.tight_layout(); axis.figure.savefig(output/"images_vs_tracks_per_class.png",dpi=160); plt.close(axis.figure)
    joined=track_stats.merge(class_stats[["species_id","species_name"]],on="species_id"); groups=[joined.loc[joined.species_id==key,"frames"].to_numpy() for key in class_stats.species_id]; axis=plt.figure(figsize=(10,6)).gca(); axis.boxplot(groups,tick_labels=class_stats.species_id,showfliers=False); axis.set(xlabel="Species id",ylabel="Frames per trajectory",title="Trajectory length by class"); axis.figure.tight_layout(); axis.figure.savefig(output/"track_length_by_class.png",dpi=160); plt.close(axis.figure)


def _write_report(summary: dict, classes: pd.DataFrame, destination: Path) -> None:
    included=classes[classes.included]
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text("# Fish4Knowledge Gate-0 data audit\n\n"+f"- Images: {summary['num_images']:,}\n- Classes: {summary['num_classes']}\n- Trajectories: {summary['num_tracks']:,}\n- Missing-mask rate: {summary['mask_missing_rate']:.2%}\n- Exact duplicate-image rows: {summary['duplicate_sha_rows']:,}\n- F4K-16T retained classes: {len(included)}\n\n"+"Trajectory-group isolation only prevents direct continuous-trajectory overlap. It does not establish camera, site, time, or scene independence.\n",encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text()); root = Path(config["data_root"])
    if not root.exists(): raise FileNotFoundError(f"Configured data_root does not exist: {root}")
    if config.get("parser_kind") == "official_groundtruth_v1":
        parsed, unresolved = scan_f4k_groundtruth(root); records = [record.__dict__ | {"group_id": record.group_id} for record in parsed]; images = [Path(record.image_path) for record in parsed]
    else:
        images = discover_images(root, config["image_extensions"]); filename_parser = compile_filename_parser(config.get("filename_regex")); records, unresolved = [], []
        for image in images:
            try:
                record = parse_image(image, filename_parser); records.append(record.__dict__ | {"group_id": record.group_id})
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
    summary = {"num_images": len(metadata), "num_classes": int(metadata.species_id.nunique()), "num_tracks": int(metadata.group_id.nunique()), "tracking_id_cross_species": bool(metadata.groupby("tracking_id_raw").species_id.nunique().gt(1).any()), "duplicate_sha_rows": int(metadata.file_sha256.duplicated(keep=False).sum()), "mask_missing_rate": float(metadata.mask_path.isna().mean())}
    atomic_json_dump(summary, output / "audit_summary.json", args.overwrite)
    _plot_audit(class_stats, track_stats, output)
    _write_report(summary, classes, Path("reports/f4k_data_audit.md"))
    print(json.dumps(summary, indent=2))
if __name__ == "__main__": main()
