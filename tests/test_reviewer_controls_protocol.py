from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_protocol():
    return yaml.safe_load(
        (ROOT / "configs" / "cxt_fish_reviewer_controls_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_reviewer_controls_are_post_hoc_and_do_not_unlock_official_test():
    cfg = load_protocol()
    assert cfg["parent_commit"] == "0332cd9bf580cde12e5f82587dd7cae9dcba3b6b"
    assert cfg["status"]["post_hoc_after_frozen_outer_results"] is True
    assert cfg["status"]["main_method_frozen"] is True
    assert cfg["official_test_accessed"] is False
    assert cfg["integrity"]["official_test_accessed"] is False


def test_donor_sensitivity_seeds_and_primary_manifest_are_frozen():
    cfg = load_protocol()["donor_sensitivity"]
    assert cfg["primary_frozen_seed"] == 3407
    assert cfg["alternative_seeds"] == [4101, 4102, 4103, 4104, 4105]
    assert cfg["primary_manifest_unchanged"] is True
    assert cfg["no_manifest_selection"] is True
    assert cfg["all_realizations_reported"] is True
    assert cfg["bootstrap_replicates"] == 5000


def test_two_rgb_control_is_exactly_the_single_new_training_control():
    cfg = load_protocol()["two_rgb_control"]
    assert cfg["id"] == "F0_2RGB"
    assert cfg["folds"] == [1, 2, 3]
    assert cfg["seeds"] == [3407, 2026, 17]
    assert cfg["recipient_batch_size"] == 64
    assert cfg["concatenated_model_batch_size"] == 128
    assert cfg["foreground_or_mask_used"] is False
    assert cfg["augmentation"]["independent_view_draws"] is True
    assert cfg["max_epochs"] == 40
    assert cfg["early_stopping_patience"] == 7
    assert cfg["no_hyperparameter_search"] is True
    assert cfg["no_outer_evaluation_until_all_9_training_cells_complete"] is True


def test_reviewer_package_does_not_change_frozen_primary_analysis():
    cfg = load_protocol()["analysis_rules"]
    assert cfg["frozen_f0_f1_unchanged"] is True
    assert cfg["donor_sensitivity_does_not_redefine_primary"] is True
    assert cfg["two_rgb_does_not_select_final_method"] is True
    assert cfg["donor_sensitivity_no_additional_realizations"] is True
