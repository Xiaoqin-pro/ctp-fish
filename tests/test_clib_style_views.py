import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from datasets.clib_style_dataset import CLIBStyleDataset
from datasets.clib_style_views import center_subject_view, four_corner_context_mosaic
from losses.clib_style_contrastive import paired_subject_infonce


def _image():
    values = np.arange(32 * 24 * 3, dtype=np.uint8).reshape(24, 32, 3)
    return Image.fromarray(values, mode="RGB")


def test_clib_style_views_are_deterministic_and_preserve_canvas():
    image = _image()
    for builder in (center_subject_view, four_corner_context_mosaic):
        first = np.asarray(builder(image, 0.25))
        second = np.asarray(builder(image, 0.25))
        assert first.shape == (24, 32, 3)
        assert np.array_equal(first, second)


def test_dataset_returns_three_views_and_rejects_non_train_records(tmp_path):
    image_path = tmp_path / "x.png"
    _image().save(image_path)
    records = pd.DataFrame([{"image_path": str(image_path), "species_id": "1", "group_id": "g", "split": "train"}])
    views = CLIBStyleDataset(records, transform=lambda x: torch.tensor(np.asarray(x)).permute(2, 0, 1))
    item = views[0]
    assert len(item) == 6 and all(tuple(t.shape) == (3, 24, 32) for t in item[:3])
    with pytest.raises(ValueError, match="train records only"):
        CLIBStyleDataset(records.assign(split="val"))


def test_paired_subject_infonce_is_finite_and_differentiable():
    embeddings = torch.randn(4, 3, 128, requires_grad=True)
    loss = paired_subject_infonce(embeddings, 0.07)
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None


def test_paired_subject_infonce_rejects_empty_negative_batch():
    with pytest.raises(ValueError, match="at least two"):
        paired_subject_infonce(torch.randn(1, 3, 8), 0.07)
