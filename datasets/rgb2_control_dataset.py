"""Two independently augmented ordinary-RGB views for F0-2RGB."""
from __future__ import annotations

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


MEAN, STD = [.485, .456, .406], [.229, .224, .225]


def standard_rgb_transform(image_size: int):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.1, .1, .1, .05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class RGB2ControlDataset(Dataset):
    def __init__(self, records: pd.DataFrame, image_size: int, class_ids: list[str]):
        self.records = records.reset_index(drop=True).copy()
        if set(self.records.split.astype(str).unique()) - {"train", "val"}:
            raise ValueError("F0-2RGB dataset rejects outer-test records")
        self.class_to_index = {str(label): index for index, label in enumerate(class_ids)}
        self.view1_transform = standard_rgb_transform(image_size)
        self.view2_transform = standard_rgb_transform(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        image = Image.open(row.image_path).convert("RGB")
        view1 = self.view1_transform(image)
        view2 = self.view2_transform(image)
        return view1, view2, self.class_to_index[str(row.species_id)], str(row.image_path), str(row.group_id)
