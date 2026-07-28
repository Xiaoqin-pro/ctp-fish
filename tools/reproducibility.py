"""Reproducibility helpers shared by Gate-0 scripts."""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment_summary() -> dict[str, Any]:
    return {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
