from __future__ import annotations

import pandas as pd
import numpy as np
from PIL import Image

from datasets.context_views import context_swap_view, foreground_blur_view
from scripts.build_phase1c_context_swap_manifest import build_manifest


def records() -> pd.DataFrame:
    return pd.DataFrame({
        "image_path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "mask_path": ["a.png", "b.png", "c.png", "d.png"],
        "species_id": ["1", "1", "2", "2"],
        "group_id": ["g1", "g2", "g3", "g4"],
        "split": ["val"] * 4,
    })


def test_manifest_is_deterministic_and_respects_donor_constraints():
    first, second = build_manifest(records(), 3407), build_manifest(records(), 3407)
    pd.testing.assert_frame_equal(first, second)
    same = first[first.swap_type == "same_class_cross_track"]
    cross = first[first.swap_type == "cross_class"]
    assert (same.recipient_species_id == same.donor_species_id).all()
    assert (cross.recipient_species_id != cross.donor_species_id).all()
    assert (first.recipient_group_id != first.donor_group_id).all()
    assert first.recipient_image_id.nunique() == 4


def test_foreground_view_preserves_center_and_changes_background():
    pixels = np.zeros((15, 15, 3), dtype=np.uint8); pixels[:, :7] = (20, 100, 200); pixels[:, 7:] = (200, 20, 30)
    image = Image.fromarray(pixels)
    mask = Image.new("L", (15, 15), 0); mask.putpixel((7, 7), 255)
    result = foreground_blur_view(image, mask, blur_kernel=5, blur_sigma=1.0, feather_radius=0)
    assert result.getpixel((7, 7)) == image.getpixel((7, 7))
    assert result.getpixel((6, 7)) != image.getpixel((6, 7))


def test_swap_keeps_recipient_foreground_and_uses_donor_background():
    recipient, donor = Image.new("RGB", (9, 9), (10, 20, 30)), Image.new("RGB", (9, 9), (200, 100, 50))
    mask = Image.new("L", (9, 9), 0); mask.putpixel((4, 4), 255)
    result = context_swap_view(recipient, mask, donor, feather_radius=0)
    assert result.getpixel((4, 4)) == recipient.getpixel((4, 4))
    assert result.getpixel((0, 0)) == donor.getpixel((0, 0))
