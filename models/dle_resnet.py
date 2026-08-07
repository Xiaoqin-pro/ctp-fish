"""DLE-Fish spatial evidence primitives for a ResNet-18 classifier."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class DLEResNet18(nn.Module):
    def __init__(self, num_classes: int, evidence_temperature: float = 1.0, *, pretrained: bool = True) -> None:
        super().__init__()
        if evidence_temperature <= 0:
            raise ValueError("evidence temperature must be positive")
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.classifier = nn.Linear(base.fc.in_features, num_classes)
        self.evidence_temperature = float(evidence_temperature)

    def feature_map(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); return self.layer4(x)

    def forward(self, image: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        feature_map = self.feature_map(image)
        global_feature = feature_map.mean(dim=(2, 3))
        global_logits = self.classifier(global_feature)
        result = {"feature_map": feature_map, "global_feature": global_feature, "global_logits": global_logits}
        if mask is None:
            return result
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must have shape [B,1,H,W]")
        mask_small = F.interpolate(mask.float(), size=feature_map.shape[-2:], mode="area").clamp(0.0, 1.0)
        mask_mass = mask_small.sum(dim=(1, 2, 3))
        masked_feature = (feature_map * mask_small).sum(dim=(2, 3)) / (mask_mass.unsqueeze(1) + 1e-6)
        masked_logits = self.classifier(masked_feature)
        cam = torch.einsum("oc,bchw->bohw", self.classifier.weight, feature_map)
        positive_evidence = F.softplus(cam / self.evidence_temperature)
        result.update({"mask_small": mask_small, "mask_mass": mask_mass, "masked_feature": masked_feature, "masked_logits": masked_logits, "class_activation": cam, "positive_evidence": positive_evidence})
        return result


def positive_class_evidence_concentration(output: dict[str, torch.Tensor], target: torch.Tensor, *, min_mass: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    mask_mass = output["mask_mass"]
    evidence = output["positive_evidence"].gather(1, target[:, None, None, None].expand(-1, 1, *output["positive_evidence"].shape[-2:])).squeeze(1)
    inside = (output["mask_small"].squeeze(1) * evidence).sum(dim=(1, 2)) / (evidence.sum(dim=(1, 2)) + 1e-6)
    valid = mask_mass >= float(min_mass)
    if valid.any():
        loss = (1.0 - inside[valid]).mean()
    else:
        loss = output["feature_map"].sum() * 0.0
    return loss, valid
