"""Official-mask three-view dataset for route-C contrastive baseline."""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, functional as TF
from torchvision.transforms import RandomResizedCrop

from datasets.mask_variants import apply_mask_variant


MEAN, STD = [.485, .456, .406], [.229, .224, .225]


class MaskContrastiveTransform:
    def __init__(self, image_size: int, crop_scale: tuple[float, float] = (0.2, 1.0)) -> None:
        self.image_size = image_size
        self.crop_scale = crop_scale
        self.jitter = ColorJitter(.1, .1, .1, .05)

    def __call__(self, *views: Image.Image) -> tuple[torch.Tensor, ...]:
        top, left, height, width = RandomResizedCrop.get_params(views[0], scale=self.crop_scale, ratio=(3 / 4, 4 / 3))
        values = [TF.resized_crop(view, top, left, height, width, (self.image_size, self.image_size), Image.Resampling.BILINEAR) for view in views]
        if random.random() < .5:
            values = [TF.hflip(view) for view in values]
        order, brightness, contrast, saturation, hue = self.jitter.get_params(self.jitter.brightness, self.jitter.contrast, self.jitter.saturation, self.jitter.hue)
        def apply(view: Image.Image) -> Image.Image:
            for function_id in order:
                if function_id == 0: view = TF.adjust_brightness(view, brightness)
                elif function_id == 1: view = TF.adjust_contrast(view, contrast)
                elif function_id == 2: view = TF.adjust_saturation(view, saturation)
                else: view = TF.adjust_hue(view, hue)
            return view
        return tuple(TF.normalize(TF.to_tensor(apply(view)), MEAN, STD) for view in values)


class MaskContrastiveDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform: MaskContrastiveTransform, class_ids: list[str], fill_value: int = 128) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.transform = transform
        self.class_to_index = {str(label): index for index, label in enumerate(class_ids)}
        self.fill_value = int(fill_value)
        if set(self.records.get("split", pd.Series(["train"] * len(self.records))).astype(str)) - {"train"}:
            raise ValueError("Mask contrastive training dataset accepts train records only")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        image = Image.open(Path(row.image_path)).convert("RGB")
        mask = Image.open(Path(row.mask_path)).convert("L")
        foreground = apply_mask_variant(image, mask, "foreground_only", fill=self.fill_value)
        non_primary = apply_mask_variant(image, mask, "background_only", fill=self.fill_value)
        original, foreground, non_primary = self.transform(image, foreground, non_primary)
        return original, foreground, non_primary, self.class_to_index[str(row.species_id)], str(row.image_path), str(row.group_id)
