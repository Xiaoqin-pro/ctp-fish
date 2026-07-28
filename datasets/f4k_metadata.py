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

F4K_SPECIES = {
    "01": "Dascyllus reticulatus", "02": "Plectroglyphidodon dickii", "03": "Chromis chrysura",
    "04": "Amphiprion clarkii", "05": "Chaetodon lunulatus", "06": "Chaetodon trifascialis",
    "07": "Myripristis kuntee", "08": "Acanthurus nigrofuscus", "09": "Hemigymnus fasciatus",
    "10": "Neoniphon sammara", "11": "Abudefduf vaigiensis", "12": "Canthigaster valentini",
    "13": "Pomacentrus moluccensis", "14": "Zebrasoma scopas", "15": "Hemigymnus melapterus",
    "16": "Lutjanus fulvus", "17": "Scolopsis bilineata", "18": "Scaridae",
    "19": "Pempheris vanicolensis", "20": "Zanclus cornutus", "21": "Neoglyphidodon nigroris",
    "22": "Balistapus undulatus", "23": "Siganus fuscescens",
}


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


def scan_f4k_groundtruth(root: Path) -> tuple[list[ParsedRecord], list[dict[str, str]]]:
    """Parse the official archive layout after it has been explicitly verified.

    Expected paths are `fish_image/fish_XX/fish_<tracking>_<fishid>.png` and matching
    `mask_image/mask_XX/mask_<tracking>_<fishid>.png`. A mismatch is unresolved, never
    silently omitted.
    """
    image_root, mask_root = root / "fish_image", root / "mask_image"
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError("Expected official fish_image/ and mask_image/ directories.")
    name_pattern = re.compile(r"^fish_(?P<tracking_id>\d+)_(?P<fish_id>\d+)\.png$")
    records: list[ParsedRecord] = []; unresolved: list[dict[str, str]] = []
    for species_id, species_name in F4K_SPECIES.items():
        folder = image_root / f"fish_{species_id}"
        if not folder.is_dir():
            unresolved.append({"path": str(folder), "reason": "missing official species image directory"}); continue
        for image in sorted(folder.glob("*.png")):
            match = name_pattern.match(image.name)
            if not match:
                unresolved.append({"path": str(image), "reason": "unexpected official filename"}); continue
            mask = mask_root / f"mask_{species_id}" / image.name.replace("fish_", "mask_", 1)
            if not mask.is_file():
                unresolved.append({"path": str(image), "reason": f"missing paired mask: {mask}"}); continue
            with Image.open(image) as opened: width, height = opened.size
            records.append(ParsedRecord(str(image), str(mask), species_id, species_name, match["tracking_id"], match["fish_id"], image.name, sha256_file(image), width, height))
    return records, unresolved
