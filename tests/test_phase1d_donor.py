from __future__ import annotations

import pandas as pd
import torch
import numpy as np
from PIL import Image

from datasets.donor_context import donor_aware_swap, donor_background
from losses.donor_suppression import donor_suppression_loss
from scripts.build_phase1d_donor_manifest import build_pairs


def test_manifest_is_deterministic_and_cross_species_cross_track():
    rows = []
    for species in ("a", "b", "c"):
        for track in ("one", "two"):
            rows.append({"image_path": f"{species}_{track}.png", "mask_path": f"{species}_{track}_m.png", "species_id": species, "group_id": f"{species}:{track}", "split": "train"})
    frame = pd.DataFrame(rows)
    first, second = build_pairs(frame, 3407), build_pairs(frame, 3407)
    pd.testing.assert_frame_equal(first, second)
    assert first.recipient_species_id.ne(first.donor_species_id).all()
    assert first.recipient_group_id.ne(first.donor_group_id).all()


def test_donor_fish_is_inpainted_before_composition():
    recipient = Image.new("RGB", (8, 8), "blue"); recipient_mask = Image.new("L", (8, 8), 0); recipient_mask.paste(255, (3, 3, 5, 5))
    donor_rgb = np.full((8, 8, 3), [0, 128, 0], dtype=np.uint8); donor_rgb[:3, :3] = [255, 0, 0]
    donor = Image.fromarray(donor_rgb); donor_mask = Image.new("L", (8, 8), 0); donor_mask.paste(255, (0, 0, 3, 3))
    background = donor_background(donor, donor_mask)
    assert background.getpixel((1, 1)) != donor.getpixel((1, 1))
    swapped = donor_aware_swap(recipient, recipient_mask, donor, donor_mask, feather_radius=0)
    assert swapped.getpixel((4, 4)) == recipient.getpixel((4, 4))


def test_donor_suppression_has_expected_margin_direction():
    logits = torch.tensor([[0.0, 1.0, 3.0]], requires_grad=True); recipient = torch.tensor([2]); donor = torch.tensor([1])
    loss, stats = donor_suppression_loss(logits, recipient, donor)
    assert loss.item() == 0.0 and stats["activation_fraction"] == 0.0
    logits = torch.tensor([[0.0, 3.0, 1.0]], requires_grad=True)
    loss, stats = donor_suppression_loss(logits, recipient, donor)
    loss.backward()
    assert loss.item() == 2.0 and stats["activation_fraction"] == 1.0
    assert logits.grad[0, 1] > 0 and logits.grad[0, 2] < 0
