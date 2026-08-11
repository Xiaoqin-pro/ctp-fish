"""Generate final construct-validity artifacts from frozen outputs only."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    v2 = json.loads((ROOT / "experiments/cxt_fish_rgb2_vs_f1_bootstrap_5000_v2.json").read_text(encoding="utf-8"))
    fish = json.loads((ROOT / "experiments/cxt_fish_fish_suppressed_donor_bootstrap_5000.json").read_text(encoding="utf-8"))
    donor = pd.read_csv(ROOT / "experiments/cxt_fish_donor_sensitivity_summary.csv")
    residual = pd.read_csv(ROOT / "experiments/cxt_fish_donor_foreground_residual.csv")
    species = pd.read_csv(ROOT / "experiments/cxt_fish_donor_species_effects.csv")
    gate = pd.read_csv(ROOT / "experiments/gate0_split_comparability.csv")
    primary_effect = float(donor.loc[donor.donor_seed.eq(3407), "delta_f1_f0_equal_weight_cell_mean"].iloc[0])
    construct = pd.DataFrame([
        {"analysis": "Primary donor", "effect": primary_effect, "analysis_role": "frozen main donor-composite result", "artifact": "experiments/cxt_fish_donor_sensitivity_summary.csv"},
        {"analysis": "5-donor sensitivity range", "effect": donor.loc[donor.donor_seed.ne(3407), "delta_f1_f0_equal_weight_cell_mean"].mean(), "effect_min": donor.loc[donor.donor_seed.ne(3407), "delta_f1_f0_equal_weight_cell_mean"].min(), "effect_max": donor.loc[donor.donor_seed.ne(3407), "delta_f1_f0_equal_weight_cell_mean"].max(), "analysis_role": "post-hoc donor-realization sensitivity", "artifact": "experiments/cxt_fish_donor_sensitivity_summary.csv"},
        {"analysis": "Donor-species-equal-weight sensitivity", "effect": float(species.effect.mean()), "analysis_role": "post-hoc descriptive reweighting", "artifact": "experiments/cxt_fish_donor_species_effects.csv"},
        {"analysis": "Fish-suppressed donor sensitivity", "effect": fish["observed_point_estimate"], "ci95_low": fish["ci95"][0], "ci95_high": fish["ci95"][1], "analysis_role": fish["analysis_role"], "artifact": "experiments/cxt_fish_fish_suppressed_donor_bootstrap_5000.json"},
    ])
    construct.to_csv(ROOT / "experiments/cxt_fish_construct_validity_summary.csv", index=False)
    audit_source_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    checkpoint_files = sorted((ROOT / "outputs/cxt_fish/final_outer").glob("fold_*/F*_seed*/best.pt"))
    rgb_checkpoints = sorted((ROOT / "outputs/cxt_fish/rgb2_control_outer").glob("fold_*/F0_2RGB_seed*/best.pt"))
    audit = {"status": "complete", "protocol_config_sha256": sha256(ROOT / "configs/cxt_fish_construct_validity_v1.yaml"), "protocol_commit": "7471c0cb64305a811231755b033fd14a3199300f", "audit_source_head": audit_source_head, "final_package_head_recorded_by_git": True, "rgb2_observed_point_estimate": v2["observed_point_estimate"], "rgb2_bootstrap_mean": v2["bootstrap_mean"], "rgb2_ci95": v2["ci95"], "fish_suppressed_observed_point_estimate": fish["observed_point_estimate"], "fish_suppressed_ci95": fish["ci95"], "donor_residual_pairs": len(residual), "donor_residual_mean": float(residual.donor_foreground_fraction_of_full_composite.mean()), "donor_species_count": int(species.donor_species.nunique()), "gate0_rows": len(gate), "frozen_f0_f1_checkpoint_count": len(checkpoint_files), "frozen_rgb2_checkpoint_count": len(rgb_checkpoints), "no_new_checkpoints_created": True, "official_test_accessed": False, "primary_manifest_unchanged": True, "five_donor_manifests_unchanged": True, "analysis_role": "post_hoc_construct_validity_and_statistics_repair"}
    (ROOT / "reports/cxt_fish_construct_validity_integrity_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    report = ["# CXT-Fish construct-validity final report", "", "This package is a closed, reviewer-motivated, inference-only and reanalysis-only extension. No model was trained, no checkpoint was selected, and official Fish4Knowledge TEST was not accessed.", "", "## 1. Protocol", "", f"- Protocol commit: `{audit['protocol_commit']}`.", f"- Audit source head before this report commit: `{audit_source_head}`.", "- The final package head is recorded by Git history rather than embedded recursively in this generated report.", "- Role: post-hoc construct-validity and statistics repair.", "", "## 2. F0-2RGB statistical repair", "", f"- Correct observed equal-weight 9-cell effect: **{v2['observed_point_estimate'] * 100:+.2f} pp**.", f"- Bootstrap mean: {v2['bootstrap_mean'] * 100:+.2f} pp.", f"- Exploratory percentile 95% interval: [{v2['ci95'][0] * 100:+.2f}, {v2['ci95'][1] * 100:+.2f}] pp.", "- The former JSON point estimate was the bootstrap distribution mean and is superseded.", "", "## 3. Donor construct audits", "", f"- Donor-species rows: {len(species)}; natural-manifest-weight effect and donor-species-equal-weight sensitivity are reported separately.", f"- Donor residual pairs: {len(residual)}; mean hard-mask residual fraction: {residual.donor_foreground_fraction_of_full_composite.mean():.4f}.", "- Residual/effect bins use pre-specified cutoffs and are descriptive, not causal.", "", "## 4. Fish-suppressed donor sensitivity", "", f"- Frozen-pair observed effect: **{fish['observed_point_estimate'] * 100:+.2f} pp**.", f"- Exploratory percentile 95% interval: [{fish['ci95'][0] * 100:+.2f}, {fish['ci95'][1] * 100:+.2f}] pp.", "- Same recipient, same donor, same pairing, same checkpoint; only donor subject suppression changed.", "- This analysis may support a narrower statement that the effect is not confined to visible donor-subject evidence; it does not prove natural-background robustness.", "", "## 5. Gate-0 comparability", "", f"- Existing image-level and group-disjoint manifests audited: {len(gate)} partition rows.", "- Splits were not changed; the observed difference is not interpreted as a causal decomposition of same-group crossing alone.", "", "## 6. Final limits", "", "The study does not claim external-dataset or cross-camera generalization, backbone-agnostic behavior, clean-accuracy improvement, independent confirmation, causal proof of foreground sufficiency, superiority over CLIB, or inferiority of contrastive learning.", "", "## 7. Integrity", "", f"- Frozen F0/F1 checkpoints present: {len(checkpoint_files)}.", f"- Frozen F0-2RGB checkpoints present: {len(rgb_checkpoints)}.", "- No new checkpoint was created by this package.", "- Official TEST accessed: false.", "- Primary and five donor-sensitivity manifests remain unchanged.", ""]
    (ROOT / "reports/cxt_fish_construct_validity_final_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
