"""Supervised contrastive losses with an explicit positive-pair definition."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def positive_mask(targets: torch.Tensor, groups: torch.Tensor, mode: str) -> torch.Tensor:
    """Return same-class non-self (standard) or cross-track positive pairs."""
    if mode not in {"standard", "cross_track"}:
        raise ValueError(f"Unsupported positive-pair mode: {mode}")
    if targets.ndim != 1 or groups.ndim != 1 or targets.shape != groups.shape:
        raise ValueError("targets and groups must be aligned one-dimensional tensors.")
    mask = targets[:, None].eq(targets[None, :])
    mask.fill_diagonal_(False)
    if mode == "cross_track":
        mask &= groups[:, None].ne(groups[None, :])
    return mask


def supervised_contrastive_loss(features: torch.Tensor, targets: torch.Tensor, groups: torch.Tensor, temperature: float, mode: str):
    """Return loss plus audit statistics; zero-positive anchors are excluded."""
    if features.ndim != 2:
        raise ValueError("features must have shape [batch, embedding_dim].")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    features = F.normalize(features, dim=1)
    positives = positive_mask(targets, groups, mode)
    logits = features @ features.T / float(temperature)
    non_self = ~torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    logits = logits.masked_fill(~non_self, float("-inf"))
    log_denom = torch.logsumexp(logits, dim=1)
    positive_count = positives.sum(dim=1)
    eligible = positive_count > 0
    if not bool(eligible.any()):
        return features.sum() * 0.0, {"eligible_anchors": 0, "mean_positive_count": 0.0}
    positive_log_prob = (logits - log_denom[:, None]).masked_fill(~positives, 0.0)
    per_anchor = -positive_log_prob.sum(dim=1) / positive_count.clamp_min(1)
    return per_anchor[eligible].mean(), {
        "eligible_anchors": int(eligible.sum().item()),
        "mean_positive_count": float(positive_count[eligible].float().mean().item()),
    }
