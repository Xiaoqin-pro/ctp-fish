from pathlib import Path


def test_phase1b_evaluator_is_validation_only():
    text = (Path(__file__).resolve().parents[1] / "scripts" / "evaluate_cxt_phase1b.py").read_text(encoding="utf-8")
    assert 'records[records.split == "val"]' in text
    assert 'partition": "val"' in text
    assert "internal_test_accessed\": False" in text
