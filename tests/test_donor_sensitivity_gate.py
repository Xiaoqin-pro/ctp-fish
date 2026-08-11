from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_donor_sensitivity_gate_requires_exactly_90_cells():
    text = (ROOT / "scripts/audit_cxt_fish_donor_sensitivity_complete.py").read_text(encoding="utf-8")
    assert "len(records) != 90" in text
    assert "4101, 4102, 4103, 4104, 4105" in text
    assert "checkpoint_sha256" in text
    assert "donor_manifest_sha256" in text
