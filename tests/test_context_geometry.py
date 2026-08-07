from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


def _module():
    path=Path(__file__).resolve().parents[1]/"scripts"/"run_context_geometry_audit.py"
    spec=importlib.util.spec_from_file_location("context_geometry",path); module=importlib.util.module_from_spec(spec)
    assert spec and spec.loader; spec.loader.exec_module(module); return module


def test_geometry_features_are_normalized(tmp_path):
    mask=np.zeros((10,20),dtype=np.uint8); mask[2:6,5:15]=255; path=tmp_path/"mask.png"; Image.fromarray(mask).save(path)
    values=_module().geometry_features(str(path))
    assert np.allclose(values,[0.5,0.4,0.2,0.5,0.4,2.5])
