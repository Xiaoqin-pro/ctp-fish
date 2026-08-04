import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from datasets.mask_contrastive_dataset import MaskContrastiveDataset, MaskContrastiveTransform
from losses.mask_subject_nonprimary import subject_nonprimary_infonce


def _records(tmp_path):
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[7:17, 11:21] = [240, 80, 20]
    mask = np.zeros((24, 32), dtype=np.uint8); mask[7:17, 11:21] = 255
    image_path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    Image.fromarray(image).save(image_path); Image.fromarray(mask).save(mask_path)
    return pd.DataFrame([{"image_path": str(image_path), "mask_path": str(mask_path), "species_id": "1", "group_id": "g", "split": "train"}])


def test_mask_dataset_returns_aligned_three_views(tmp_path):
    dataset = MaskContrastiveDataset(_records(tmp_path), MaskContrastiveTransform(16), ["1"])
    item = dataset[0]
    assert len(item) == 6
    assert all(tuple(value.shape) == (3, 16, 16) for value in item[:3])


def test_mask_dataset_rejects_non_train_records(tmp_path):
    with pytest.raises(ValueError, match="train records only"):
        MaskContrastiveDataset(_records(tmp_path).assign(split="val"), MaskContrastiveTransform(16), ["1"])


def test_route_c_loss_is_finite_and_differentiable():
    embeddings = torch.randn(4, 3, 128, requires_grad=True)
    loss = subject_nonprimary_infonce(embeddings, 0.07)
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None


def test_route_c_loss_rejects_empty_negative_batch():
    with pytest.raises(ValueError, match="at least two"):
        subject_nonprimary_infonce(torch.randn(1, 3, 8), 0.07)
