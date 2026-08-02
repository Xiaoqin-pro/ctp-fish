"""Independent original/foreground views for CT-DFS."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from datasets.context_views import foreground_blur_view
from datasets.phase1c_dataset import MEAN, PairedTrainTransform


class CTDFSDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform: PairedTrainTransform, class_ids: list[str], *, blur_kernel: int, blur_sigma: float, feather_radius: int):
        self.records = records.reset_index(drop=True).copy()
        if set(self.records.split.unique()) != {"train"}:
            raise ValueError("CT-DFS training dataset accepts train records only")
        self.transform = transform
        self.class_to_index = {label: index for index, label in enumerate(class_ids)}
        self.kwargs = {"blur_kernel": blur_kernel, "blur_sigma": blur_sigma, "feather_radius": feather_radius}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, pair: tuple[int, int]):
        original_index, foreground_index = pair
        original_row = self.records.iloc[int(original_index)]
        foreground_row = self.records.iloc[int(foreground_index)]
        original = Image.open(Path(original_row.image_path)).convert("RGB")
        foreground_source = Image.open(Path(foreground_row.image_path)).convert("RGB")
        foreground_mask = Image.open(Path(foreground_row.mask_path)).convert("L")
        foreground = foreground_blur_view(foreground_source, foreground_mask, **self.kwargs)
        # Independent streams receive independent random geometry/photometry.
        original_view = self.transform(original, original)
        _, foreground_view = self.transform(foreground_source, foreground)
        return (
            original_view,
            foreground_view,
            self.class_to_index[str(original_row.species_id)],
            self.class_to_index[str(foreground_row.species_id)],
            str(original_row.image_path),
            str(foreground_row.image_path),
            str(original_row.group_id),
            str(foreground_row.group_id),
        )
