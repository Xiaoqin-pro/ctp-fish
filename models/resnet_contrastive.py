"""ImageNet ResNet-18 classifier with a training-only projection head."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Contrastive(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(backbone.fc.in_features, num_classes)
        self.projector = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 128))

    def forward(self, images: torch.Tensor):
        features = self.encoder(images).flatten(1)
        return self.classifier(features), F.normalize(self.projector(features), dim=1)
