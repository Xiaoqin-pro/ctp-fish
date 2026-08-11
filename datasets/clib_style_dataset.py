"""Dataset for the frozen CLIB-style three-view baseline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from datasets.clib_style_views import center_subject_view, four_corner_context_mosaic


class CLIBStyleDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform=None, ratio: float = 0.25, class_ids=None):
        self.records = records.reset_index(drop=True).copy()
        self.transform = transform
        self.ratio = float(ratio)
        self.class_ids = class_ids or sorted(self.records.species_id.astype(str).unique())
        self.class_to_index = {str(value): index for index, value in enumerate(self.class_ids)}
        split = self.records.get("split", pd.Series(["train"] * len(self.records)))
        if set(split.astype(str)) - {"train"}:
            raise ValueError("CLIB-style training dataset accepts train records only")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        original = Image.open(Path(row.image_path)).convert("RGB")
        views = (
            original,
            center_subject_view(original, self.ratio),
            four_corner_context_mosaic(original, self.ratio),
        )
        if self.transform is not None:
            views = tuple(self.transform(view) for view in views)
        return views + (self.class_to_index[str(row.species_id)], str(row.image_path), str(row.group_id))
