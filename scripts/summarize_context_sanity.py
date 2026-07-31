"""Freeze the CXT-Fish train/validation-only context sanity audit report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def load_metrics(root: Path, condition: str) -> dict:
    if condition == "geometry_only": path=root/condition/"metrics_val.json"
    else: path=root/f"resnet18_track_seed407_{condition}"/"evaluation_context_sanity_val"/"metrics_val.json"
    if not path.exists(): raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",default="outputs/cxt_fish/context_sanity"); parser.add_argument("--report",default="reports/cxt_fish_context_sanity_report.md"); parser.add_argument("--summary",default="experiments/cxt_fish_context_sanity_summary.csv"); args=parser.parse_args()
    root=Path(args.root); conditions=["mask_only","geometry_only","constant_fill_background","inpainted_background","shuffled_mask_background"]
    metrics={condition:load_metrics(root,condition) for condition in conditions}
    rows=[{"condition":condition,**{key:metrics[condition][key] for key in ("accuracy","balanced_accuracy","macro_f1","weighted_f1")},"partition":"val","seed":407} for condition in conditions]
    summary=Path(args.summary); summary.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(summary,index=False)
    index_manifest=root/"indices"/"manifest.json"; manifest=json.loads(index_manifest.read_text())
    constant=metrics["constant_fill_background"]["macro_f1"]; inpaint=metrics["inpainted_background"]["macro_f1"]; shuffled=metrics["shuffled_mask_background"]["macro_f1"]
    report=f"""# CXT-Fish context-shortcut sanity audit

## Scope

This fixed-seed diagnostic uses only the frozen F4K-16T track-level `train`
and `val` partitions. It does not read the internal test partition or locked
outer folds. The visual conditions each use a separately trained ResNet-18;
geometry-only uses the preregistered six-feature logistic regression.

## Validation results

| Condition | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
"""
    for row in rows: report+=f"| {row['condition']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} | {row['weighted_f1']:.4f} |\n"
    report+=f"""

## Interpretation fixed by this audit

- `mask_only` reaches macro F1 {metrics['mask_only']['macro_f1']:.4f}, whereas
  `geometry_only` reaches {metrics['geometry_only']['macro_f1']:.4f}. Thus,
  detailed silhouette/shape information is highly discriminative, while the
  predefined box-scale-position features alone are not a sufficient account.
- `constant_fill_background` reaches {constant:.4f};
  `shuffled_mask_background` reaches {shuffled:.4f} (difference
  {(shuffled-constant):+.4f}). Replacing the fish-hole shape with a
  cross-track donor mask does not materially remove the available signal in
  this fixed diagnostic.
- `inpainted_background` remains high at {inpaint:.4f}, but is
  {(constant-inpaint):.4f} below the constant-fill condition. Therefore the
  original background-only signal is consistent with both residual
  environmental/camera context and some fish-removal artifact. Telea
  inpainting is itself a diagnostic transformation and not a reconstruction of
  the true seabed.

The manuscript terminology is therefore fixed to **contextual shortcuts**:
the evidence does not justify attributing the effect solely to background,
nor does it establish causal environmental influence or scene independence.

## Consequence

The next authorized stage is Phase 1A's orthogonal sampling and class-prior
baselines. Cross-track contrast, foreground context loss, prototype learning,
memory banks, internal-test access and outer-fold access remain out of scope
until Phase 1A is frozen.

## Provenance

- Seed: `407`
- Shuffled-mask index SHA-256: `{manifest['combined']['sha256']}`
- Index rows: `{manifest['combined']['rows']}` (train + validation only)
- Internal test read: `{manifest['internal_test_read']}`
- Outer folds read: `{manifest['outer_folds_read']}`
- Report input summary SHA-256: `{sha256(summary)}`
"""
    destination=Path(args.report); destination.parent.mkdir(parents=True,exist_ok=True); destination.write_text(report,encoding="utf-8")
    print(json.dumps({"report":str(destination),"summary":str(summary),"internal_test_read":False,"outer_folds_read":False},indent=2))


if __name__=="__main__": main()
