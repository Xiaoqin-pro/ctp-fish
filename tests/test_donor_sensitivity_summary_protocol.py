from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_summary_script_reports_all_five_alternative_realizations():
    text = (ROOT / "scripts/summarize_cxt_fish_donor_sensitivity.py").read_text(encoding="utf-8")
    assert "4101, 4102, 4103, 4104, 4105" in text
    assert "primary_3407" in text
    assert "favorable_cells" in text
    assert "do not redefine the primary endpoint" in text
