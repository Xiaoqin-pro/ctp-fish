"""Frozen Phase 1C context objective definitions."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def context_consistency_loss(original_feature: torch.Tensor, foreground_feature: torch.Tensor) -> torch.Tensor:
    if original_feature.shape != foreground_feature.shape:
        raise ValueError("context consistency features must share a shape")
    return (1.0 - F.cosine_similarity(F.normalize(original_feature, dim=1), F.normalize(foreground_feature.detach(), dim=1), dim=1)).mean()


def variant_losses(variant: str) -> dict[str, bool]:
    if variant not in {"f1", "f2", "f3"}:
        raise ValueError("Phase 1C trainable variants are f1/f2/f3")
    return {"foreground_ce": variant in {"f1", "f3"}, "consistency": variant in {"f2", "f3"}}
