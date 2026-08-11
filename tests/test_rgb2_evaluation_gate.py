from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rgb2_evaluation_requires_completion_gate_and_has_nine_cell_queue():
    script = (ROOT / "scripts/evaluate_cxt_fish_rgb2_outer.py").read_text(encoding="utf-8")
    queue = (ROOT / "scripts/run_cxt_fish_rgb2_evaluation_queue.ps1").read_text(encoding="utf-8")
    assert "reports/cxt_fish_rgb2_training_complete.json" in script
    assert 'gate_payload.get("cells") != 9' in script
    assert "evaluate_cxt_fish_rgb2_outer.py" in queue
    assert "foreach ($fold in @(1, 2, 3))" in queue
