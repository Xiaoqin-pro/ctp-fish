"""ResNet-18 classifier exposing its fixed pooled feature for Phase 1C."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Context(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.encoder(image).flatten(1)
        return self.classifier(feature), feature
