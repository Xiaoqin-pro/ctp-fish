"""Atomic output helpers. Existing results are protected by default."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def _prepare(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing result: {path}. Use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_json_dump(value: Any, path: Path, overwrite: bool = False) -> None:
    _prepare(path, overwrite)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv_dump(frame: pd.DataFrame, path: Path, overwrite: bool = False) -> None:
    _prepare(path, overwrite)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as handle:
        frame.to_csv(handle.name, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)
