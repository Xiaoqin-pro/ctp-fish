"""Zero-training TRAFS risk Gate on the frozen F0 validation evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--risk", default="outputs/cxt_fish/phase1e/val_track_risk.csv"); parser.add_argument("--evaluation", default="outputs/cxt_fish/phase1a_replication/A1_s1_ce_seed3407/evaluation_phase1c_f0_seed3407/per_image_val.csv"); parser.add_argument("--output", default="outputs/cxt_fish/phase1e/risk_gate"); args = parser.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    risk = pd.read_csv(args.risk); evaluation = pd.read_csv(args.evaluation); frame = risk.merge(evaluation, on="image_path", validate="one_to_one", suffixes=("", "_eval")); frame["swap_failure"] = frame.original.eq(frame.target) & ~frame.cross_swap.eq(frame.target); frame["dar_flip"] = frame.original.eq(frame.target) & frame.cross_swap.eq(frame.cross_donor_target); frame["risk_group"] = pd.qcut(frame.image_risk.rank(method="first"), 3, labels=["low", "mid", "high"]); frame.to_csv(out / "per_image_gate.csv", index=False)
    def auc(score, label): return float(roc_auc_score(label.astype(int), score)) if label.nunique() == 2 else None
    group = frame.groupby(["species_id", "group_id"], as_index=False).agg(image_risk=("image_risk", "median"), track_risk=("track_risk", "first"), swap_failure=("swap_failure", "mean"), dar_flip=("dar_flip", "mean")); group["within_class_track_risk"] = group.groupby("species_id")["track_risk"].rank(method="average", pct=True); group["risk_group"] = pd.qcut(group.image_risk.rank(method="first"), 3, labels=["low", "mid", "high"]); group.to_csv(out / "per_track_gate.csv", index=False)
    summary = {"image_rows": len(frame), "tracks": len(group), "image_risk_auroc": auc(frame.image_risk, frame.swap_failure), "track_risk_auroc": auc(group.track_risk, group.swap_failure.gt(0)), "within_class_track_risk_auroc": auc(group.within_class_track_risk, group.swap_failure.gt(0)), "risk_groups": frame.groupby("risk_group", observed=False).agg(swap_failure=("swap_failure", "mean"), dar_flip=("dar_flip", "mean"), n=("image_path", "size")).reset_index().to_dict("records"), "per_species": frame.groupby(["species_id", "risk_group"], observed=False).agg(swap_failure=("swap_failure", "mean"), dar_flip=("dar_flip", "mean"), n=("image_path", "size")).reset_index().to_dict("records")}
    high_low = {row["risk_group"]: row for row in summary["risk_groups"]}; summary["high_minus_low_swap_failure"] = high_low.get("high", {}).get("swap_failure", 0) - high_low.get("low", {}).get("swap_failure", 0); summary["high_minus_low_dar_flip"] = high_low.get("high", {}).get("dar_flip", 0) - high_low.get("low", {}).get("dar_flip", 0); summary["species_positive_direction_count"] = sum(float(item["swap_failure"]) > float(high_low.get("low", {"swap_failure": 0})["swap_failure"]) for item in summary["per_species"] if item["risk_group"] == "high"); summary["gate_pass"] = bool(summary["high_minus_low_swap_failure"] > 0 and summary["image_risk_auroc"] is not None and summary["image_risk_auroc"] >= 0.60 and summary["species_positive_direction_count"] >= 3)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
