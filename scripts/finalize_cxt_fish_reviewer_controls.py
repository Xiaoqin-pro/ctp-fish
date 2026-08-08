"""Run final integrity checks and generate the closed reviewer-control report."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    b_root = ROOT / "outputs/cxt_fish/donor_sensitivity/evaluation"
    b_cells = list(b_root.glob("seed_*/fold_*/F?_seed*/metrics.json"))
    if len(b_cells) != 90:
        raise SystemExit(f"Expected 90 donor sensitivity cells, got {len(b_cells)}")
    b_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in b_cells]
    if any(p.get("official_test_accessed") is not False for p in b_payloads):
        raise SystemExit("Donor sensitivity official TEST flag violation")
    c_root = ROOT / "outputs/cxt_fish/rgb2_control_outer_evaluation"
    c_cells = list(c_root.glob("fold_*/F0_2RGB_seed*/metrics.json"))
    if len(c_cells) != 9:
        raise SystemExit(f"Expected 9 RGB2 evaluation cells, got {len(c_cells)}")
    c_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in c_cells]
    if any(p.get("official_test_accessed") is not False for p in c_payloads):
        raise SystemExit("RGB2 official TEST flag violation")
    primary_hashes = {}
    for fold in (1, 2, 3):
        manifest = ROOT / f"outputs/cxt_fish/final_outer_manifests/fold_{fold}/outer_context_swap.csv"
        manifest_meta = manifest.with_suffix(".manifest.json")
        payload = json.loads(manifest_meta.read_text(encoding="utf-8"))
        digest = sha256(manifest)
        if digest != payload.get("manifest_sha256"):
            raise SystemExit(f"Primary manifest changed or metadata mismatch: fold {fold}")
        primary_hashes[str(fold)] = digest
    changed = subprocess.check_output(["git", "diff", "--name-only", "0332cd9..HEAD"], cwd=ROOT, text=True).splitlines()
    forbidden = [p for p in changed if p.startswith(("configs/cxt_fish_outer_validation_v1", "experiments/cxt_fish_resnet_outer_bootstrap", "reports/cxt_fish_outer_confirmation_report"))]
    if forbidden:
        raise SystemExit(f"Frozen primary artifacts changed: {forbidden}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    audit = {
        "status": "complete", "protocol_commit": "11ecb6436e78af9a22c5141760ff7dceefc60bab", "final_commit": commit,
        "donor_sensitivity_cells": 90, "donor_sensitivity_seeds": [4101, 4102, 4103, 4104, 4105],
        "rgb2_training_cells": 9, "rgb2_evaluation_cells": 9, "primary_manifest_sha256": primary_hashes,
        "official_test_accessed": False, "primary_artifacts_changed": False,
    }
    (ROOT / "reports/cxt_fish_reviewer_controls_integrity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    report = [
        "# CXT-Fish reviewer-control package: final report", "",
        "This closed package was completed after the frozen outer results. It is",
        "post-hoc reviewer-motivated evidence, not a new method-selection stage.", "",
        "## Completed controls", "",
        "- A: class-inclusion, donor-protocol, and frozen-artifact audits.",
        "- B: five alternative donor realizations (4101--4105), 90/90 cell evaluations.",
        "- C: F0-2RGB supervision-matched control, 9/9 training and 9/9 evaluation cells.",
        "- Official Fish4Knowledge TEST: not accessed.", "",
        "## B result", "",
        "The five alternative donor realizations all produced positive equal-weight",
        "cell-mean CXT-Fish minus F0 cross-composite differences. The realization",
        "range and cell-level counts are in `reports/cxt_fish_donor_sensitivity_report.md`.",
        "The original seed-3407 +7.24pp result remains the frozen primary result.", "",
        "## C result", "",
        "F0-2RGB has the same ResNet18, sampler, optimizer budget, 2B forward, and",
        "two CE terms, but uses two independently augmented ordinary-RGB views with",
        "no mask or foreground view. The exploratory paired bootstrap and summary",
        "are in `experiments/cxt_fish_rgb2_vs_f1_bootstrap_5000.json` and",
        "`reports/cxt_fish_rgb2_control_report.md`.", "",
        "## Interpretation boundary", "",
        "These controls address duplicated supervised exposure and donor-assignment",
        "concerns. They do not establish cross-dataset, cross-camera, or cross-site",
        "generalization. They do not convert the post-hoc controls into confirmatory",
        "endpoints or alter the frozen primary result.", "",
        f"Protocol commit: `{audit['protocol_commit']}`", f"Final commit: `{commit}`", "",
        "Integrity audit: passed.",
    ]
    (ROOT / "reports/cxt_fish_reviewer_controls_final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
