from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_donor_sensitivity_protocol_matches_frozen_seed_set():
    cfg = yaml.safe_load(
        (ROOT / "configs" / "cxt_fish_reviewer_controls_v1.yaml").read_text(encoding="utf-8")
    )
    sensitivity = cfg["donor_sensitivity"]
    assert sensitivity["primary_frozen_seed"] == 3407
    assert sensitivity["alternative_seeds"] == [4101, 4102, 4103, 4104, 4105]
    assert sensitivity["algorithm"] == "frozen_outer_algorithm_except_seed"
    assert sensitivity["primary_manifest_unchanged"] is True
    assert sensitivity["inference_only"] is True
