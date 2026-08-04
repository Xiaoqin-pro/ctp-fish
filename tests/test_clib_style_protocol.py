from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cxt_fish_clib_style_outer_v1.yaml"
CONFIG_V11 = ROOT / "configs" / "cxt_fish_clib_style_outer_v1_1.yaml"


def load_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_clib_style_protocol_is_frozen_and_external_only():
    cfg = load_config()
    assert cfg["protocol_version"] == "cxt_fish_clib_style_outer_v1"
    assert cfg["seeds"] == [3407, 2026, 17]
    assert cfg["outer_folds"] == 3
    assert cfg["outer_test_access"] == "final_evaluation_only"
    assert cfg["access_policy"]["official_test_accessed"] is False
    assert cfg["access_policy"]["calibration_accessed"] is False
    assert cfg["prohibitions"]


def test_clib_style_views_and_contrastive_rule_are_fixed():
    cfg = load_config()
    views = cfg["views"]
    assert views["construction"] == "paper_faithful_clib_style"
    assert views["subject"] == "center_crop"
    assert views["background"] == "four_corner_mosaic"
    assert views["ratio"] == 0.25
    assert views["mask_used_for_training_views"] is False
    contrastive = cfg["contrastive"]
    assert contrastive["temperature"] == 0.07
    assert contrastive["positive_pair"] == "original_subject_same_sample"
    assert contrastive["projection_head"] == "512-256-relu-128-l2"


def test_clib_style_run_matrix_is_exactly_nine_cells():
    cfg = load_config()
    assert len(cfg["seeds"]) * int(cfg["outer_folds"]) == 9
    assert cfg["evaluation"]["bootstrap_replicates"] == 1000
    assert "cross_class_swap_macro_f1" in cfg["evaluation"]["metrics"]


def test_clib_style_driver_records_frozen_provenance():
    driver = (ROOT / "scripts" / "train_cxt_fish_clib_style_outer.py").read_text(encoding="utf-8")
    for field in ("initialization_sha256", "outer_train_sha256", "inner_dev_sha256", "view_schema_sha256"):
        assert field in driver
    assert "outer_test_loaded" in driver


def test_clib_style_v11_is_explicitly_adapted_and_uses_two_augmentations():
    cfg = yaml.safe_load(CONFIG_V11.read_text(encoding="utf-8"))
    assert cfg["protocol_version"] == "cxt_fish_clib_style_outer_v1_1"
    assert cfg["views"]["construction"] == "deterministic_clib_inspired_interpretation"
    assert cfg["views"]["official_formula_verified"] is False
    assert cfg["pretrain_augmentation"]["crop_scale"] == [0.2, 1.0]
    assert cfg["classifier_augmentation"]["crop_scale"] == [0.08, 1.0]
    assert cfg["training"]["scheduler"] == "cosine_annealing_each_stage"
