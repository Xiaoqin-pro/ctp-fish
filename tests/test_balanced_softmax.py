from __future__ import annotations

import pytest
import torch

from losses.balanced_softmax import BalancedSoftmaxLoss


def test_balanced_softmax_adds_train_prior_to_logits():
    loss=BalancedSoftmaxLoss(torch.tensor([1.0,4.0])); logits=torch.zeros(1,2); target=torch.tensor([1])
    assert loss(logits,target) < torch.nn.functional.cross_entropy(logits,target)


def test_balanced_softmax_rejects_nonpositive_counts():
    with pytest.raises(ValueError): BalancedSoftmaxLoss(torch.tensor([1.0,0.0]))
