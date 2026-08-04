"""Fixed foreground and context-swap image constructions for CXT-Fish Phase 1C."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


SCHEMA_VERSION = "cxt_fish_phase1c_context_views_v1"


def feathered_mask(mask: Image.Image, size: tuple[int, int], radius: int) -> np.ndarray:
    """Return a deterministic [0, 1] alpha mask on the requested width/height canvas."""
    width, height = size
    raw = np.asarray(mask.convert("L").resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8)
    alpha = (raw > 0).astype(np.float32)
    if radius < 0:
        raise ValueError("mask feather radius must be non-negative")
    if radius:
        kernel = 2 * radius + 1
        alpha = cv2.GaussianBlur(alpha, (kernel, kernel), sigmaX=0, sigmaY=0)
    return np.clip(alpha, 0.0, 1.0)


def foreground_blur_view(image: Image.Image, mask: Image.Image, *, blur_kernel: int, blur_sigma: float, feather_radius: int) -> Image.Image:
    """Keep official-mask foreground pixels and replace the background by fixed Gaussian blur."""
    if blur_kernel < 3 or blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be an odd integer >= 3")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    blurred = cv2.GaussianBlur(rgb, (blur_kernel, blur_kernel), sigmaX=blur_sigma, sigmaY=blur_sigma)
    alpha = feathered_mask(mask, (rgb.shape[1], rgb.shape[0]), feather_radius)[..., None]
    return Image.fromarray(np.round(alpha * rgb + (1.0 - alpha) * blurred).astype(np.uint8))


def non_primary_blur_view(image: Image.Image, mask: Image.Image, *, blur_kernel: int, blur_sigma: float, feather_radius: int) -> Image.Image:
    """Keep non-primary pixels and blur the annotated primary fish region."""
    if blur_kernel < 3 or blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be an odd integer >= 3")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    blurred = cv2.GaussianBlur(rgb, (blur_kernel, blur_kernel), sigmaX=blur_sigma, sigmaY=blur_sigma)
    alpha = feathered_mask(mask, (rgb.shape[1], rgb.shape[0]), feather_radius)[..., None]
    return Image.fromarray(np.round((1.0 - alpha) * rgb + alpha * blurred).astype(np.uint8))


def context_swap_view(recipient: Image.Image, recipient_mask: Image.Image, donor: Image.Image, *, feather_radius: int) -> Image.Image:
    """Keep recipient fish and use donor RGB as background, on the recipient canvas before normalization."""
    target = np.asarray(recipient.convert("RGB"), dtype=np.uint8)
    donor_rgb = np.asarray(donor.convert("RGB").resize((target.shape[1], target.shape[0]), Image.Resampling.BILINEAR), dtype=np.uint8)
    alpha = feathered_mask(recipient_mask, (target.shape[1], target.shape[0]), feather_radius)[..., None]
    return Image.fromarray(np.round(alpha * target + (1.0 - alpha) * donor_rgb).astype(np.uint8))
