"""The only Gate-0 classifier: ImageNet-initialised torchvision ResNet18."""
from __future__ import annotations

from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = __import__("torch").nn.Linear(model.fc.in_features, num_classes)
    return model
