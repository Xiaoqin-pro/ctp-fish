"""Frozen donor-aware counterfactual context construction for CXT-Fish Phase 1D."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from datasets.context_views import feathered_mask


SCHEMA_VERSION = "cxt_fish_phase1d_donor_context_v1"


def donor_background(image: Image.Image, mask: Image.Image, *, dilation_iterations: int = 1, inpaint_radius: float = 3.0) -> Image.Image:
    """Remove the donor fish with the frozen Gate-0 Telea procedure."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    raw = np.asarray(mask.convert("L").resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST), dtype=np.uint8)
    binary = (raw > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hole = cv2.dilate(binary, kernel, iterations=dilation_iterations)
    return Image.fromarray(cv2.inpaint(rgb, hole, inpaint_radius, cv2.INPAINT_TELEA))


def donor_aware_swap(recipient: Image.Image, recipient_mask: Image.Image, donor: Image.Image, donor_mask: Image.Image, *, feather_radius: int, dilation_iterations: int = 1, inpaint_radius: float = 3.0) -> Image.Image:
    """Compose recipient fish over an inpainted, donor-fish-free donor background."""
    target = np.asarray(recipient.convert("RGB"), dtype=np.uint8)
    background = np.asarray(donor_background(donor, donor_mask, dilation_iterations=dilation_iterations, inpaint_radius=inpaint_radius).resize((target.shape[1], target.shape[0]), Image.Resampling.BILINEAR), dtype=np.uint8)
    alpha = feathered_mask(recipient_mask, (target.shape[1], target.shape[0]), feather_radius)[..., None]
    return Image.fromarray(np.round(alpha * target + (1.0 - alpha) * background).astype(np.uint8))
