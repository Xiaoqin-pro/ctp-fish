from pathlib import Path

import torch
import yaml

from models.mobilenet_context import MobileNetV3LargeContext


ROOT = Path(__file__).resolve().parents[1]


def test_mobilenet_transfer_protocol_is_fixed():
    cfg = yaml.safe_load((ROOT / "configs/cxt_fish_mobilenet_outer_v1.yaml").read_text())
    assert cfg["backbone"] == "mobilenet_v3_large"
    assert cfg["initialization"] == "ImageNet1K_V2"
    assert cfg["seeds"] == [3407]
    assert cfg["outer_folds"] == [1, 2, 3]
    assert [item["id"] for item in cfg["methods"]] == ["MV0", "MV1"]
    assert [item["foreground_ce_weight"] for item in cfg["methods"]] == [0.0, 1.0]
    assert cfg["foreground_blur_kernel"] == 21
    assert cfg["foreground_blur_sigma"] == 5.0
    assert cfg["mask_feather_radius"] == 3


def test_mobilenet_context_forward_shape():
    model = MobileNetV3LargeContext(16).eval()
    with torch.no_grad():
        logits, feature = model(torch.zeros(1, 3, 224, 224))
    assert tuple(logits.shape) == (1, 16)
    assert tuple(feature.shape) == (1, 960)
    assert all(parameter.requires_grad for parameter in model.parameters())
