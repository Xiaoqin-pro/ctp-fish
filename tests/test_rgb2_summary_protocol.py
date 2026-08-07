from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rgb2_summary_is_post_hoc_and_requires_nine_paired_cells():
    text = (ROOT / "scripts/summarize_cxt_fish_rgb2_control.py").read_text(encoding="utf-8")
    assert "post-hoc supervision- and compute-matched mechanism control" in text
    assert "bootstrap_cross_delta" in text
    assert "official_test_accessed" in text
    assert "for fold in (1, 2, 3)" in text
    assert "for seed in SEEDS" in text
