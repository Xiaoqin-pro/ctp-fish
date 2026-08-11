"""Train-only-prior Balanced Softmax classification objective for Phase 1A."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts: torch.Tensor) -> None:
        super().__init__()
        if class_counts.ndim != 1 or not torch.isfinite(class_counts).all() or (class_counts <= 0).any(): raise ValueError("Balanced Softmax requires finite positive class counts.")
        self.register_buffer("log_prior", class_counts.float().log())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.log_prior.numel(): raise ValueError("Logit dimension does not match class-prior dimension.")
        return F.cross_entropy(logits+self.log_prior, targets)
