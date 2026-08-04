import numpy as np
import pandas as pd
import pytest
import torch
import cv2
from PIL import Image

from datasets.mask_contrastive_dataset import MaskContrastiveDataset, MaskContrastiveTransform
from datasets.context_views import foreground_blur_view, non_primary_blur_view
from losses.mask_subject_nonprimary import subject_nonprimary_infonce
from scripts.evaluate_cxt_fish_mask_contrastive_outer import validate_state


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


def test_route_c_views_match_cxt_fish_blur_and_complement():
    image = Image.fromarray(np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3))
    mask = Image.fromarray(np.pad(np.ones((10, 10), dtype=np.uint8) * 255, ((7, 7), (11, 11))))
    foreground = np.asarray(foreground_blur_view(image, mask, blur_kernel=21, blur_sigma=5.0, feather_radius=3))
    non_primary = np.asarray(non_primary_blur_view(image, mask, blur_kernel=21, blur_sigma=5.0, feather_radius=3))
    assert foreground.shape == non_primary.shape == (24, 32, 3)
    blurred = cv2.GaussianBlur(np.asarray(image), (21, 21), sigmaX=5.0, sigmaY=5.0)
    assert np.array_equal(foreground[0, 0], blurred[0, 0])
    assert np.array_equal(non_primary[0, 0], image.getpixel((0, 0)))
    assert not np.array_equal(foreground[12, 16], non_primary[12, 16])


def test_route_c_loss_manual_positive_indices_and_shuffled_positive():
    base = torch.eye(8)
    embeddings = torch.zeros(2, 3, 8)
    embeddings[0, 0] = base[0]; embeddings[0, 1] = base[0]; embeddings[0, 2] = base[1]
    embeddings[1, 0] = base[2]; embeddings[1, 1] = base[2]; embeddings[1, 2] = base[3]
    aligned = subject_nonprimary_infonce(embeddings, 0.07)
    shuffled = embeddings[:, [0, 2, 1]]
    assert aligned < subject_nonprimary_infonce(shuffled, 0.07)


def test_route_c_evaluator_rejects_wrong_checkpoint_metadata(tmp_path):
    cfg = tmp_path / "cfg.yaml"; train = tmp_path / "train.csv"; dev = tmp_path / "dev.csv"; schema = tmp_path / "schema.py"
    for path, text in ((cfg, "cfg"), (train, "train"), (dev, "dev"), (schema, "schema")): path.write_text(text, encoding="utf-8")
    state = {"stage": "classifier", "fold": "1", "seed": 3407, "config_sha256": "bad", "outer_train_sha256": "bad", "inner_dev_sha256": "bad", "class_ids": ["1"], "outer_test_accessed": False, "official_test_accessed": False}
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_state(state, fold="1", seed=3407, cfg_path=cfg, train_path=train, dev_path=dev, class_ids=["1"])
