"""Plain image classification dataset; no trajectory method enters Gate-0."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from datasets.mask_variants import apply_mask_variant


class F4KDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform=None, mask_variant: str = "original", class_ids: list[str] | None = None) -> None:
        self.records = records.reset_index(drop=True).copy(); self.transform = transform; self.mask_variant = mask_variant
        self.class_ids = class_ids or sorted(self.records.species_id.astype(str).unique())
        self.class_to_index = {value: index for index, value in enumerate(self.class_ids)}

    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        image = Image.open(Path(row.image_path)).convert("RGB")
        if self.mask_variant != "original":
            if not isinstance(row.get("mask_path"), str): raise ValueError("Mask view requested without a matched official mask.")
            donor_mask = Image.open(Path(row.donor_mask_path)) if self.mask_variant == "shuffled_mask_background" and isinstance(row.get("donor_mask_path"), str) else None
            image = apply_mask_variant(image, Image.open(Path(row.mask_path)), self.mask_variant, donor_mask=donor_mask)
        if self.transform: image = self.transform(image)
        return image, self.class_to_index[str(row.species_id)], str(row.image_path), str(row.group_id)
