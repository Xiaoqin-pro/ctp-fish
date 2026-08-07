from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_rgb2_config_freezes_the_single_post_hoc_control():
    cfg = yaml.safe_load((ROOT / "configs/cxt_fish_rgb2_control_v1.yaml").read_text(encoding="utf-8"))
    assert cfg["control_type"] == "post_hoc_two_rgb_supervision_matched"
    assert cfg["mask_used"] is False
    assert cfg["foreground_used"] is False
    assert cfg["recipient_batch_size"] == 64
    assert cfg["model_batch_after_concat"] == 128
    assert cfg["epochs"] == 40
    assert cfg["early_stopping_patience"] == 7
    assert cfg["official_test_accessed"] is False


def test_rgb2_queue_has_exactly_nine_cells_and_no_outer_evaluation():
    queue = (ROOT / "scripts/run_cxt_fish_rgb2_outer_queue.ps1").read_text(encoding="utf-8")
    assert "$seeds = @(3407, 2026, 17)" in queue
    assert "foreach ($fold in @(1, 2, 3))" in queue
    assert "train_cxt_fish_rgb2_outer.py" in queue
    assert "evaluate_cxt_fish" not in queue
