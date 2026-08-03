"""DLE training views with RGB, foreground and mask geometry synchronized."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, RandomResizedCrop, functional as TF

from datasets.context_views import foreground_blur_view
from datasets.phase1c_dataset import MEAN, STD


class DLETrainTransform:
    def __init__(self, image_size: int) -> None:
        self.image_size = image_size
        self.jitter = ColorJitter(.1, .1, .1, .05)

    def __call__(self, image: Image.Image, foreground: Image.Image, mask: Image.Image):
        top, left, height, width = RandomResizedCrop.get_params(image, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3))
        image = TF.resized_crop(image, top, left, height, width, (self.image_size, self.image_size), Image.Resampling.BILINEAR)
        foreground = TF.resized_crop(foreground, top, left, height, width, (self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = TF.resized_crop(mask, top, left, height, width, (self.image_size, self.image_size), Image.Resampling.NEAREST)
        if random.random() < .5:
            image, foreground, mask = TF.hflip(image), TF.hflip(foreground), TF.hflip(mask)
        order, brightness, contrast, saturation, hue = self.jitter.get_params(self.jitter.brightness, self.jitter.contrast, self.jitter.saturation, self.jitter.hue)
        for function_id in order:
            if function_id == 0: image, foreground = TF.adjust_brightness(image, brightness), TF.adjust_brightness(foreground, brightness)
            elif function_id == 1: image, foreground = TF.adjust_contrast(image, contrast), TF.adjust_contrast(foreground, contrast)
            elif function_id == 2: image, foreground = TF.adjust_saturation(image, saturation), TF.adjust_saturation(foreground, saturation)
            else: image, foreground = TF.adjust_hue(image, hue), TF.adjust_hue(foreground, hue)
        rgb = lambda view: TF.normalize(TF.to_tensor(view), MEAN, STD)
        mask_tensor = TF.to_tensor(mask).clamp(0.0, 1.0)
        return rgb(image), rgb(foreground), mask_tensor


class DLEDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform: DLETrainTransform, class_ids: list[str], *, blur_kernel: int, blur_sigma: float, feather_radius: int):
        self.records = records.reset_index(drop=True).copy(); self.transform = transform
        self.class_to_index = {label: index for index, label in enumerate(class_ids)}
        self.kwargs = {"blur_kernel": blur_kernel, "blur_sigma": blur_sigma, "feather_radius": feather_radius}
        if set(self.records.split.unique()) - {"train"}: raise ValueError("DLE training dataset accepts train records only")

    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        image = Image.open(Path(row.image_path)).convert("RGB")
        mask = Image.open(Path(row.mask_path)).convert("L")
        foreground = foreground_blur_view(image, mask, **self.kwargs)
        original, foreground, mask_tensor = self.transform(image, foreground, mask)
        return original, foreground, mask_tensor, self.class_to_index[str(row.species_id)], str(row.image_path), str(row.group_id)
