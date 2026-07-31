from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_background_audit.py"
    spec = importlib.util.spec_from_file_location("background_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_background_audit_plan_is_fixed_and_track_only() -> None:
    module = _load_script_module()
    plan = module.audit_plan("configs/resnet18_gate0.yaml", "python")
    train = [entry for entry in plan if entry["kind"] == "train"]
    evaluation = [entry for entry in plan if entry["kind"] == "evaluate"]
    assert [entry["view"] for entry in train] == ["original", "foreground_only", "background_only"]
    assert len(evaluation) == 5
    assert all("--split track" in entry["command"] for entry in plan)
    assert all("--seed 407" in entry["command"] for entry in train)
    assert not any("outer" in entry["command"].lower() for entry in plan)

