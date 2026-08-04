"""Deterministic CLIB-style subject/context views.

These views intentionally do not use the official fish masks.  They are a
paper-faithful *style* baseline: a central subject crop and a four-corner
context mosaic are generated from the RGB image, then the normal training
transform is applied independently to each view.
"""
from __future__ import annotations

from PIL import Image


SCHEMA_VERSION = "clib_style_views_v1"


def _crop_box(width: int, height: int, ratio: float, left: int, top: int) -> tuple[int, int, int, int]:
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    crop_w = max(1, round(width * ratio))
    crop_h = max(1, round(height * ratio))
    return left, top, min(width, left + crop_w), min(height, top + crop_h)


def center_subject_view(image: Image.Image, ratio: float = 0.25) -> Image.Image:
    """Return a center crop resized to the original RGB canvas size."""
    source = image.convert("RGB")
    width, height = source.size
    crop_w = max(1, round(width * ratio))
    crop_h = max(1, round(height * ratio))
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    return source.crop(_crop_box(width, height, ratio, left, top)).resize(
        (width, height), Image.Resampling.BILINEAR
    )


def four_corner_context_mosaic(image: Image.Image, ratio: float = 0.25) -> Image.Image:
    """Return a deterministic 2x2 mosaic of the four corner crops.

    ``ratio`` is the retained side fraction for each source corner crop.  The
    four crops are resized into equal quadrants of the original canvas.
    """
    source = image.convert("RGB")
    width, height = source.size
    crop_w = max(1, round(width * ratio))
    crop_h = max(1, round(height * ratio))
    boxes = (
        _crop_box(width, height, ratio, 0, 0),
        _crop_box(width, height, ratio, width - crop_w, 0),
        _crop_box(width, height, ratio, 0, height - crop_h),
        _crop_box(width, height, ratio, width - crop_w, height - crop_h),
    )
    half_w, half_h = max(1, width // 2), max(1, height // 2)
    tiles = [source.crop(box).resize((half_w, half_h), Image.Resampling.BILINEAR) for box in boxes]
    mosaic = Image.new("RGB", (half_w * 2, half_h * 2))
    for tile, location in zip(tiles, ((0, 0), (half_w, 0), (0, half_h), (half_w, half_h))):
        mosaic.paste(tile, location)
    return mosaic.resize((width, height), Image.Resampling.BILINEAR)
