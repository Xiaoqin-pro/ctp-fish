from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_donor_sensitivity_queue_has_exactly_90_cells():
    queue = (ROOT / "scripts" / "run_cxt_fish_donor_sensitivity_queue.ps1").read_text(encoding="utf-8")
    assert "4101, 4102, 4103, 4104, 4105" in queue
    assert '"F0", "F1"' in queue
    assert "foreach ($modelSeed in $modelSeeds)" in queue
    assert "if ($LASTEXITCODE -ne 0)" in queue
