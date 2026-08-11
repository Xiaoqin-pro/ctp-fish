"""MobileNetV3-Large classifier exposing its pooled feature for CXT-Fish."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class MobileNetV3LargeContext(nn.Module):
    """ImageNet-V2 MobileNetV3-Large with the same ``(logits, feature)`` API as ResNet18Context."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.encoder = base.features
        self.pool = base.avgpool
        self.pre_classifier = base.classifier[0]
        self.activation = base.classifier[1]
        self.dropout = base.classifier[2]
        self.classifier = nn.Linear(base.classifier[3].in_features, num_classes)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.pool(self.encoder(image)).flatten(1)
        hidden = self.activation(self.pre_classifier(feature))
        logits = self.classifier(self.dropout(hidden))
        return logits, feature


def build_mobilenet_v3_large(num_classes: int) -> MobileNetV3LargeContext:
    return MobileNetV3LargeContext(num_classes)
