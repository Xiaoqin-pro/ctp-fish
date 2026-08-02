from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.recalibrate_phase1c_bn import recalibrate


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__(); self.bn = nn.BatchNorm1d(2); self.linear = nn.Linear(2, 1)
    def forward(self, x): return self.linear(self.bn(x))


def test_recalibration_updates_only_bn_statistics():
    model = TinyModel(); before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    loader = DataLoader(TensorDataset(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.zeros(2)), batch_size=2)
    # The production loader yields four values; adapt the toy batches to that protocol.
    batches = [(x, y, ["a"], ["g"]) for x, y in loader]
    count = recalibrate(model, batches, torch.device("cpu"))
    assert count == 1
    assert not torch.equal(model.bn.running_mean, before["bn.running_mean"])
    assert torch.equal(model.linear.weight, before["linear.weight"])
    assert torch.equal(model.linear.bias, before["linear.bias"])
