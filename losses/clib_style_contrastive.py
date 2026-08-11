"""Three-view InfoNCE primitive for the CLIB-style fairness baseline."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def paired_subject_infonce(embeddings: torch.Tensor, temperature: float = 0.07):
    """Contrast original/subject pairs; every other view is a negative."""
    if embeddings.ndim != 3 or embeddings.shape[1] != 3:
        raise ValueError("embeddings must have shape [batch, 3, dim]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch = embeddings.shape[0]
    if batch < 2:
        raise ValueError("at least two samples are required for negatives")
    normalized = F.normalize(embeddings, dim=-1)
    anchors = normalized[:, :2].reshape(batch * 2, -1)
    candidates = normalized.reshape(batch * 3, -1)
    logits = anchors @ candidates.T / float(temperature)
    positive_indices = torch.stack(
        (torch.arange(batch, device=embeddings.device) * 3 + 1,
         torch.arange(batch, device=embeddings.device) * 3), dim=1
    ).reshape(-1)
    anchor_indices = torch.arange(batch * 2, device=embeddings.device)
    logits[anchor_indices, anchor_indices + (anchor_indices // 2)] = float("-inf")
    return F.cross_entropy(logits, positive_indices)
