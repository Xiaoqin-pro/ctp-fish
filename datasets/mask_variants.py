"""Deterministic online mask views for Gate-0 and CXT-Fish diagnostics."""
from __future__ import annotations

import numpy as np
from PIL import Image
import cv2


MASK_VARIANTS = ("original", "foreground_only", "background_only", "mask_only", "constant_fill_background", "inpainted_background", "shuffled_mask_background")


def _foreground(mask: Image.Image, height: int, width: int) -> np.ndarray:
    value = np.asarray(mask.convert("L")) > 0
    if value.shape != (height, width): raise ValueError("Mask and image dimensions differ.")
    return value


def apply_mask_variant(image: Image.Image, mask: Image.Image, variant: str, fill: int = 128, donor_mask: Image.Image | None = None) -> Image.Image:
    rgb = np.asarray(image.convert("RGB")).copy()
    foreground = _foreground(mask, *rgb.shape[:2])
    if variant == "original": return Image.fromarray(rgb)
    if variant == "foreground_only": rgb[~foreground] = fill
    elif variant in ("background_only", "constant_fill_background"): rgb[foreground] = fill
    elif variant == "mask_only": return Image.fromarray(np.repeat((foreground.astype(np.uint8) * 255)[..., None], 3, axis=2))
    elif variant == "inpainted_background":
        binary=(foreground.astype(np.uint8) * 255)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated=cv2.dilate(binary, kernel, iterations=1)
        rgb=cv2.inpaint(rgb, dilated, 3, cv2.INPAINT_TELEA)
    elif variant == "shuffled_mask_background":
        if donor_mask is None: raise ValueError("Shuffled-mask view requires donor_mask.")
        donor=np.asarray(donor_mask.convert("L").resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)) > 0
        rgb[donor] = fill
    else: raise ValueError(f"Unknown mask variant: {variant}")
    return Image.fromarray(rgb)
