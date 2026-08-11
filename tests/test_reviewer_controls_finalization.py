from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_finalization_checks_frozen_primary_and_new_cell_counts():
    text = (ROOT / "scripts/finalize_cxt_fish_reviewer_controls.py").read_text(encoding="utf-8")
    assert "len(b_cells) != 90" in text
    assert "len(c_cells) != 9" in text
    assert "primary_manifest_sha256" in text
    assert "primary_artifacts_changed" in text
    assert "official_test_accessed" in text
