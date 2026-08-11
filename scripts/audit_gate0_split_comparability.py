"""Audit, without retraining, the comparability of the frozen Gate-0 splits."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def split_stats(path: Path, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    rows = []
    for split, part in frame.groupby("split", sort=True):
        rows.append({"protocol": protocol, "split": split, "images": len(part), "groups": part.group_id.nunique(), "classes": part.species_id.nunique()})
    per_class = frame.groupby(["split", "species_id"], sort=True).agg(images=("image_path", "size"), groups=("group_id", "nunique")).reset_index()
    per_class["protocol"] = protocol
    return pd.DataFrame(rows), per_class


def main() -> None:
    summaries, per_class = [], []
    for protocol, filename in (("image-level", "splits/f4k16t_image_level_dev.csv"), ("group-disjoint", "splits/f4k16t_track_level_dev.csv")):
        summary, classes = split_stats(ROOT / filename, protocol)
        summaries.append(summary); per_class.append(classes)
    summary = pd.concat(summaries, ignore_index=True)
    classes = pd.concat(per_class, ignore_index=True)
    classes["proportion"] = classes.groupby(["protocol", "split"]).images.transform(lambda x: x / x.sum())
    classes.to_csv(ROOT / "experiments/gate0_split_comparability.csv", index=False)
    cfg = json.loads("{}")
    rows = ["# Gate-0 split comparability audit", "", "This is a frozen, no-training audit of the existing image-level and group-disjoint development manifests. Splits were not changed and the observed performance difference is not decomposed causally.", "", "## Partition sizes", "", "| protocol | split | images | groups | classes |", "|---|---|---:|---:|---:|"]
    for r in summary.itertuples(index=False): rows.append(f"| {r.protocol} | {r.split} | {r.images} | {r.groups} | {r.classes} |")
    rows += ["", "## Training/evaluation configuration", "", "- Architecture: ResNet-18", "- Initialization: ImageNet", "- Input: 224 px", "- Optimizer: AdamW", "- Learning rate: 3e-4", "- Weight decay: 1e-4", "- Maximum epochs: 40; early stopping patience: 7", "- Seeds: 3407, 2026, 17", "- Checkpoint metric: protocol-local test metrics were not used for selection; existing Gate-0 metadata is retained as the source.", "", "## Interpretation", "", "The two protocols yield materially different performance estimates. Any difference reflects the complete protocol change, including partition composition and group crossing, and is not interpreted as a causal estimate of same-group crossing alone.", "", "- Official TEST accessed: false."]
    (ROOT / "reports/gate0_split_comparability_audit.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(summary), "per_class_rows": len(classes), "official_test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
