"""Strict discovery and metadata parsing for a real Fish4Knowledge layout.

The parser deliberately has no guessed filename convention. A named-group regular
expression is supplied only after the local dataset layout has been audited.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from tools.hashing import sha256_file


REQUIRED_GROUPS = {"species_id", "tracking_id"}


@dataclass(frozen=True)
class ParsedRecord:
    image_path: str
    mask_path: str | None
    species_id: str
    species_name: str
    tracking_id_raw: str
    fish_id_raw: str | None
    filename: str
    file_sha256: str
    width: int
    height: int

    @property
    def group_id(self) -> str:
        return f"{self.species_id}::{self.tracking_id_raw}"


def discover_images(root: Path, extensions: Iterable[str]) -> list[Path]:
    allowed = {suffix.lower() for suffix in extensions}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed)


def compile_filename_parser(pattern: str | None) -> re.Pattern[str]:
    if not pattern:
        raise ValueError("filename_regex is required after inspecting the real dataset layout.")
    parser = re.compile(pattern)
    missing = REQUIRED_GROUPS - set(parser.groupindex)
    if missing:
        raise ValueError(f"filename_regex lacks required named groups: {sorted(missing)}")
    return parser


def parse_image(path: Path, parser: re.Pattern[str], mask_path: Path | None = None) -> ParsedRecord:
    match = parser.search(path.name)
    if match is None:
        raise ValueError(f"Filename did not match filename_regex: {path}")
    fields = match.groupdict()
    with Image.open(path) as image:
        width, height = image.size
    return ParsedRecord(
        image_path=str(path), mask_path=str(mask_path) if mask_path else None,
        species_id=fields["species_id"], species_name=fields.get("species_name") or fields["species_id"],
        tracking_id_raw=fields["tracking_id"], fish_id_raw=fields.get("fish_id"),
        filename=path.name, file_sha256=sha256_file(path), width=width, height=height,
    )
