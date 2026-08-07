"""Paired RGB/foreground views with identical train-time geometry for Phase 1C."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, functional as TF
from torchvision.transforms import RandomResizedCrop

from datasets.context_views import foreground_blur_view


MEAN, STD = [.485, .456, .406], [.229, .224, .225]


class PairedTrainTransform:
    def __init__(self, image_size: int) -> None:
        self.image_size = image_size
        self.jitter = ColorJitter(.1, .1, .1, .05)

    def __call__(self, original: Image.Image, foreground: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        top, left, height, width = RandomResizedCrop.get_params(original, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3))
        values = [TF.resized_crop(view, top, left, height, width, (self.image_size, self.image_size), Image.Resampling.BILINEAR) for view in (original, foreground)]
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


class Phase1CDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform: PairedTrainTransform, class_ids: list[str], *, blur_kernel: int, blur_sigma: float, feather_radius: int) -> None:
        self.records = records.reset_index(drop=True).copy()
        if set(self.records.split.unique()) - {"train", "val"}:
            raise ValueError("Phase 1C dataset rejects test and outer-fold records")
        self.transform = transform; self.class_to_index = {label: index for index, label in enumerate(class_ids)}
        self.kwargs = {"blur_kernel": blur_kernel, "blur_sigma": blur_sigma, "feather_radius": feather_radius}

    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        image = Image.open(Path(row.image_path)).convert("RGB")
        mask = Image.open(Path(row.mask_path)).convert("L")
        fg = foreground_blur_view(image, mask, **self.kwargs)
        original, foreground = self.transform(image, fg)
        return original, foreground, self.class_to_index[str(row.species_id)], str(row.image_path), str(row.group_id)
