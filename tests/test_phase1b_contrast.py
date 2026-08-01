from __future__ import annotations

import pandas as pd
import pytest
import torch

from datasets.phase1b_sampler import StructuredTrackBatchSampler
from losses.supervised_contrastive import positive_mask, supervised_contrastive_loss
from scripts.train_cxt_phase1b import variant_spec


def _records():
    rows = []
    for label in range(4):
        for track in range(3):
            for frame in range(2):
                rows.append({"species_id": str(label), "group_id": f"{label}_{track}", "image_path": f"{label}_{track}_{frame}"})
    return pd.DataFrame(rows)


def test_structured_sampler_has_exact_p_q_k_and_is_deterministic():
    first = StructuredTrackBatchSampler(_records(), p=4, q=3, k=2, seed=3407, num_batches=2)
    second = StructuredTrackBatchSampler(_records(), p=4, q=3, k=2, seed=3407, num_batches=2)
    first.set_epoch(3); second.set_epoch(3)
    first_batches, second_batches = list(first), list(second)
    assert first_batches == second_batches
    records = _records()
    for batch in first_batches:
        frame = records.iloc[batch]
        assert len(batch) == 24 and frame.species_id.nunique() == 4
        assert all(group.group_id.nunique() == 3 for _, group in frame.groupby("species_id"))
        assert all(len(group) == 2 for _, group in frame.groupby(["species_id", "group_id"]))


def test_structured_sampler_rejects_validation_or_test_records():
    records = _records(); records["split"] = "train"; records.loc[0, "split"] = "val"
    with pytest.raises(ValueError, match="train records only"):
        StructuredTrackBatchSampler(records, p=4, q=3, k=2, seed=1, num_batches=1)


def test_variants_share_classification_and_projection_configuration():
    c0, c1, c2 = (variant_spec(name, 0.1) for name in ("c0", "c1", "c2"))
    assert {item["classification_loss"] for item in (c0, c1, c2)} == {"ce"}
    assert {item["projection_head"] for item in (c0, c1, c2)} == {"512-256-128"}
    assert c0["contrastive_weight"] == 0.0 and c1["contrastive_weight"] == c2["contrastive_weight"] == 0.1


def test_standard_and_cross_track_positive_masks_are_exact():
    targets = torch.tensor([0, 0, 0, 0, 0, 0])
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    standard = positive_mask(targets, groups, "standard")
    cross = positive_mask(targets, groups, "cross_track")
    assert standard.sum(dim=1).tolist() == [5] * 6
    assert cross.sum(dim=1).tolist() == [4] * 6
    assert not bool(cross[0, 1]) and bool(cross[0, 2])


def test_cross_track_loss_is_finite_with_eligible_anchors():
    features = torch.randn(24, 128, requires_grad=True)
    targets = torch.arange(4).repeat_interleave(6)
    groups = torch.arange(12).repeat_interleave(2)
    loss, stats = supervised_contrastive_loss(features, targets, groups, 0.1, "cross_track")
    assert torch.isfinite(loss) and stats["eligible_anchors"] == 24 and stats["mean_positive_count"] == 4.0
    loss.backward(); assert features.grad is not None


def test_no_positive_anchor_returns_graph_connected_zero():
    features = torch.randn(3, 8, requires_grad=True)
    loss, stats = supervised_contrastive_loss(features, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), 0.1, "cross_track")
    assert loss.item() == 0.0 and stats["eligible_anchors"] == 0
    loss.backward(); assert features.grad is not None
