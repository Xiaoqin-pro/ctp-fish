from __future__ import annotations

import importlib.util
from pathlib import Path


def test_context_sanity_summary_module_loads() -> None:
    path=Path(__file__).resolve().parents[1]/"scripts"/"summarize_context_sanity.py"
    spec=importlib.util.spec_from_file_location("context_sanity_summary",path); module=importlib.util.module_from_spec(spec)
    assert spec and spec.loader; spec.loader.exec_module(module)
    assert callable(module.load_metrics)
