"""Online foreground/background views for the frozen Gate-0 audit."""
from __future__ import annotations

import numpy as np
from PIL import Image


def apply_mask_variant(image: Image.Image, mask: Image.Image, variant: str, fill: int = 128) -> Image.Image:
    rgb = np.asarray(image.convert("RGB")).copy()
    foreground = np.asarray(mask.convert("L")) > 0
    if foreground.shape != rgb.shape[:2]: raise ValueError("Mask and image dimensions differ.")
    if variant == "original": return Image.fromarray(rgb)
    if variant == "foreground_only": rgb[~foreground] = fill
    elif variant == "background_only": rgb[foreground] = fill
    else: raise ValueError(f"Unknown mask variant: {variant}")
    return Image.fromarray(rgb)
