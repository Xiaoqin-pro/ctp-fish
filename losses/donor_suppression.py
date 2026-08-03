"""Targeted donor-logit suppression used only by frozen Phase 1D D2."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def donor_suppression_loss(logits: torch.Tensor, recipient: torch.Tensor, donor: torch.Tensor, margin: float = 0.0) -> tuple[torch.Tensor, dict[str, float]]:
    if logits.ndim != 2 or logits.shape[0] != recipient.numel() or recipient.shape != donor.shape:
        raise ValueError("logits, recipient and donor shapes are incompatible")
    rows = torch.arange(logits.shape[0], device=logits.device)
    recipient_logit, donor_logit = logits[rows, recipient], logits[rows, donor]
    terms = F.relu(float(margin) + donor_logit - recipient_logit)
    return terms.mean(), {"activation_fraction": float(terms.gt(0).float().mean().detach()), "mean_recipient_minus_donor": float((recipient_logit - donor_logit).mean().detach())}
