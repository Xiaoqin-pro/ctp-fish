from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_gate0.py"
    spec = importlib.util.spec_from_file_location("gate0_summary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_gate0_decision_requires_all_predeclared_structural_signals() -> None:
    module = _module()
    assert module.gate0_decision(0.99, 0.95, 0.80, 0.25, 0.0) == "PROCEED_WITH_CAUTION"
    assert module.gate0_decision(0.95, 0.95, 0.80, 0.25, 0.0) == "STOP"
