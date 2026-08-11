from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _module():
    path=Path(__file__).resolve().parents[1]/"scripts"/"build_context_sanity_indices.py"
    spec=importlib.util.spec_from_file_location("context_indices",path); module=importlib.util.module_from_spec(spec)
    assert spec and spec.loader; spec.loader.exec_module(module); return module


def test_shuffled_mask_pairs_are_cross_track_and_deterministic():
    records=pd.DataFrame({"image_path":["a","b","c"],"mask_path":["ma","mb","mc"],"group_id":["g1","g1","g2"],"split":["train","train","train"]})
    module=_module(); first=module.build_pairs(records,"train"); second=module.build_pairs(records,"train")
    assert first.equals(second)
    assert (first.recipient_group_id != first.donor_group_id).all()
