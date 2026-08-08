import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.recompute_cxt_fish_rgb2_bootstrap_v2 import _bootstrap, _observed


def _frame(groups, targets, predictions):
    return pd.DataFrame({"group_id": groups, "target": targets, "prediction": predictions})


def test_rgb2_observed_point_is_paired_cell_mean_and_method_mean_difference():
    summary = pd.DataFrame({"method": ["CXT-Fish"] * 9 + ["F0_2RGB"] * 9, "cross_composite_macro_f1": [0.8] * 9 + [0.7] * 9})
    assert np.isclose(_observed(summary), 0.1)


def test_rgb2_bootstrap_is_deterministic_and_finite():
    cells = []
    for fold in (1, 2, 3):
        for seed in (3407, 2026, 17):
            base = _frame(["g1", "g1", "g2", "g2"], [0, 0, 1, 1], [0, 0, 1, 1]).rename(columns={"prediction": "cross_swap"})
            changed = base.copy(); changed["cross_swap"] = [0, 1, 1, 1]
            cells.append({"fold": fold, "seed": seed, "rgb": base, "cxt": changed})
    a = _bootstrap(cells, 20, 3410); b = _bootstrap(cells, 20, 3410)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()


def test_v2_artifact_has_observed_point_separate_from_bootstrap_mean():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "experiments/cxt_fish_rgb2_vs_f1_bootstrap_5000_v2.json").read_text(encoding="utf-8"))
    assert "observed_point_estimate" in payload and "bootstrap_mean" in payload
    assert payload["analysis_role"] == "exploratory_post_hoc_mechanism_control"
    assert payload["official_test_accessed"] is False


def test_fish_suppressed_outputs_use_all_18_cells_and_false_official_flag():
    root = Path(__file__).resolve().parents[1]
    files = list((root / "outputs/cxt_fish/fish_suppressed_donor_evaluation").glob("fold_*/F*_seed*/metrics.json"))
    assert len(files) == 18
    assert all(json.loads(p.read_text(encoding="utf-8"))["official_test_accessed"] is False for p in files)


def test_primary_manifests_and_checkpoints_are_present():
    root = Path(__file__).resolve().parents[1]
    assert len(list((root / "outputs/cxt_fish/final_outer_manifests").glob("fold_*/outer_context_swap.csv"))) == 3
    assert len(list((root / "outputs/cxt_fish/final_outer").glob("fold_*/F*_seed*/best.pt"))) == 18


def test_construct_integrity_audit_locks_official_test():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "reports/cxt_fish_construct_validity_integrity_audit.json").read_text(encoding="utf-8"))
    assert payload["official_test_accessed"] is False
    assert payload["no_new_checkpoints_created"] is True
