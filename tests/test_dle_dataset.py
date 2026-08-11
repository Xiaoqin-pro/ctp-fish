from PIL import Image
import numpy as np
import torch

from datasets.dle_dataset import DLETrainTransform


def test_dle_transform_keeps_mask_synchronized_shape_and_range():
    image = Image.fromarray(np.full((64, 80, 3), 128, dtype=np.uint8))
    foreground = image.copy()
    mask = Image.fromarray(np.pad(np.ones((20, 20), dtype=np.uint8) * 255, ((20, 24), (30, 30))))
    original, fg, transformed_mask = DLETrainTransform(32)(image, foreground, mask)
    assert original.shape == fg.shape == (3, 32, 32)
    assert transformed_mask.shape == (1, 32, 32)
    assert float(transformed_mask.min()) >= 0.0 and float(transformed_mask.max()) <= 1.0
    assert torch.isfinite(transformed_mask).all()
