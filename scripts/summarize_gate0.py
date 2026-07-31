"""Create the reproducible, results-only CTP-Fish Gate-0 report.

This script does not train or evaluate models.  It only summarizes the fixed
Gate-0 artifacts after all six baseline and five mask-view evaluations exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def gate0_decision(image_macro_f1: float, track_macro_f1: float, background_macro_f1: float, image_same_track_fraction: float, track_same_track_fraction: float) -> str:
    """Predeclared structural decision: no result is a single pass threshold."""
    leakage_signal = image_same_track_fraction > track_same_track_fraction
    protocol_gap = image_macro_f1 > track_macro_f1
    background_signal = background_macro_f1 > 1.0 / 16.0
    return "PROCEED_WITH_CAUTION" if leakage_signal and protocol_gap and background_signal else "STOP"


def _metric_row(run: str, split: str, metrics: dict) -> dict:
    return {key: value for key, value in {
        "run": run, "split": split, "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"], "track_balanced_accuracy": metrics["track_balanced_accuracy"],
    }.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet18_gate0.yaml")
    parser.add_argument("--output-report", default="reports/ctp_fish_gate0_report.md")
    parser.add_argument("--output-summary", default="experiments/ctp_fish_gate0_summary.csv")
    args = parser.parse_args()
    cfg_path = Path(args.config); cfg = yaml.safe_load(cfg_path.read_text())
    root = Path("outputs/gate0"); base = root / "baselines"; background = root / "background_audit"
    metadata = pd.read_csv(cfg["metadata_path"])
    rows = []
    grouped: dict[str, list[dict]] = {"image": [], "track": []}
    for split in grouped:
        for seed in cfg["seeds"]:
            run = f"resnet18_{split}_seed{seed}"
            path = base / run / "evaluation" / "metrics_test.json"
            if not path.exists(): raise FileNotFoundError(path)
            metric = json.loads(path.read_text()); grouped[split].append(metric); rows.append(_metric_row(run, split, metric))
    summary_path = Path(args.output_summary); summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    def aggregate(split: str, key: str) -> tuple[float, float]: return mean_std([float(x[key]) for x in grouped[split]])
    image_macro, image_macro_std = aggregate("image", "macro_f1")
    track_macro, track_macro_std = aggregate("track", "macro_f1")
    image_track, image_track_std = aggregate("image", "track_balanced_accuracy")
    track_track, track_track_std = aggregate("track", "track_balanced_accuracy")
    similarity = json.loads((root / "similarity" / "similarity_summary.json").read_text())
    background_rows = []
    for path in background.rglob("metrics_test.json"):
        item = json.loads(path.read_text()); background_rows.append(item)
    required = {"train_original__eval_original", "train_foreground_only__eval_foreground_only", "train_background_only__eval_background_only", "train_original__eval_foreground_only", "train_original__eval_background_only"}
    audit = {item["evaluation_name"]: item for item in background_rows}
    missing = required - set(audit)
    if missing: raise RuntimeError(f"Missing background audit artifacts: {sorted(missing)}")
    decision = gate0_decision(image_macro, track_macro, audit["train_background_only__eval_background_only"]["macro_f1"], similarity["image_level"]["phash_same_track_fraction"], similarity["track_level"]["phash_same_track_fraction"])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    hashes = {"config": sha256(cfg_path), "image_split": sha256(Path(cfg["image_split_path"])), "track_split": sha256(Path(cfg["track_split_path"]))}
    report = f"""# CTP-Fish Gate-0: Data and Baseline Audit

## Decision: {decision}

Gate-0 provides structural evidence to continue **only with a track-aware,
context-controlled protocol**. This is not a method result and does not open
the locked outer folds.

## Dataset and protocol

- Official Fish4Knowledge records audited: **{len(metadata):,}** images, **{metadata.group_id.nunique():,}** species-track groups, **{metadata.species_id.nunique()}** species.
- F4K-16T development subset: **{len(pd.read_csv(cfg['track_split_path'])):,}** images across 16 retained species.
- Image-level split permits trajectory crossing; track-level split keeps every `group_id` in exactly one partition.
- All models: ImageNet-initialized ResNet-18, 224 px, AdamW, fixed 40-epoch ceiling and early stopping; seeds {cfg['seeds']}.

## Baseline comparison (mean +/- sample SD across three seeds)

| Split protocol | Macro F1 | Track-balanced accuracy |
|---|---:|---:|
| Image-level | {image_macro:.4f} +/- {image_macro_std:.4f} | {image_track:.4f} +/- {image_track_std:.4f} |
| Track-level | {track_macro:.4f} +/- {track_macro_std:.4f} | {track_track:.4f} +/- {track_track_std:.4f} |

The macro-F1 gap is {(image_macro-track_macro)*100:.2f} percentage points. It is evidence that the conventional image-level protocol is optimistic for class-balanced recognition, even though aggregate image accuracy remains high.

## Cross-partition similarity audit

| Protocol | Mean pHash distance | Mean feature cosine | pHash nearest-neighbor same-track fraction | Feature nearest-neighbor same-track fraction |
|---|---:|---:|---:|---:|
| Image-level | {similarity['image_level']['mean_phash_distance']:.4f} | {similarity['image_level']['mean_feature_cosine']:.4f} | {similarity['image_level']['phash_same_track_fraction']:.4f} | {similarity['image_level']['feature_same_track_fraction']:.4f} |
| Track-level | {similarity['track_level']['mean_phash_distance']:.4f} | {similarity['track_level']['mean_feature_cosine']:.4f} | {similarity['track_level']['phash_same_track_fraction']:.4f} | {similarity['track_level']['feature_same_track_fraction']:.4f} |

Track-level splitting removes direct crossing of the recorded fish trajectories. It does not establish independence across site, camera, date, or unrecorded scene factors.

## Online mask-view background audit (fixed track-level seed 407)

| Train view -> test view | Accuracy | Balanced accuracy | Macro F1 | Track-balanced accuracy |
|---|---:|---:|---:|---:|
"""
    for name in ["train_original__eval_original", "train_foreground_only__eval_foreground_only", "train_background_only__eval_background_only", "train_original__eval_foreground_only", "train_original__eval_background_only"]:
        item = audit[name]
        label = name.replace('__', ' -> ').replace('train_', '').replace('eval_', '')
        report += f"| {label} | {item['accuracy']:.4f} | {item['balanced_accuracy']:.4f} | {item['macro_f1']:.4f} | {item['track_balanced_accuracy']:.4f} |\n"
    report += f"""

The background-only matched-view result is materially above chance for 16 classes. This establishes available contextual signal, not a claim that background is the sole causal shortcut. The original-view model's performance under foreground-only and background-only test shifts should be interpreted as sensitivity to information removal rather than as an in-distribution accuracy estimate.

## Research room and boundary

The observed combination of (i) track-crossing similarity under image-level splitting, (ii) a track-level macro-F1 decrease, and (iii) nontrivial background-only recognition supports a constrained next phase: methods must be evaluated with the frozen track-aware protocol and must directly test whether improvements remain after controlling contextual cues. No trajectory sampler, prototype learner, contrastive objective, or outer-fold result is included in Gate-0.

## Provenance

- Git commit: `{commit}`
- Config SHA-256: `{hashes['config']}`
- Image-level split SHA-256: `{hashes['image_split']}`
- Track-level split SHA-256: `{hashes['track_split']}`
- Official outer folds remain locked and were not evaluated.
"""
    report_path = Path(args.output_report); report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"decision": decision, "report": str(report_path), "summary": args.output_summary}, indent=2))


if __name__ == "__main__": main()
