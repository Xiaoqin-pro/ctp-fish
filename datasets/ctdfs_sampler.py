"""Deterministic independent two-stream sampling for CT-DFS."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class CTDFSDualStreamSampler(Sampler[tuple[int, int]]):
    """Yield (original-index, foreground-index) pairs with fixed stream modes."""

    MODES = {"s1_track_uniform", "s2_class_uniform", "s3_class_track_uniform"}

    def __init__(self, records, original_mode: str, foreground_mode: str, seed: int, num_samples: int | None = None):
        if original_mode not in self.MODES or foreground_mode not in self.MODES:
            raise ValueError("unsupported CT-DFS sampling mode")
        self.records = records.reset_index(drop=True)
        self.original_mode = original_mode
        self.foreground_mode = foreground_mode
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(num_samples or len(self.records))
        self.by_track: dict[str, list[int]] = defaultdict(list)
        self.by_class: dict[str, list[int]] = defaultdict(list)
        self.tracks_by_class: dict[str, list[str]] = defaultdict(list)
        for index, row in self.records.iterrows():
            label, group = str(row.species_id), str(row.group_id)
            self.by_track[group].append(index)
            self.by_class[label].append(index)
        for group, indices in self.by_track.items():
            self.tracks_by_class[str(self.records.iloc[indices[0]].species_id)].append(group)
        self.tracks = sorted(self.by_track)
        self.classes = sorted(self.by_class)
        if not self.tracks or not self.classes:
            raise ValueError("CT-DFS sampler requires non-empty classes and tracks")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def _draw(self, rng: np.random.Generator, mode: str) -> int:
        if mode == "s1_track_uniform":
            group = str(rng.choice(self.tracks))
            return int(rng.choice(self.by_track[group]))
        if mode == "s2_class_uniform":
            label = str(rng.choice(self.classes))
            return int(rng.choice(self.by_class[label]))
        label = str(rng.choice(self.classes))
        group = str(rng.choice(sorted(self.tracks_by_class[label])))
        return int(rng.choice(self.by_track[group]))

    def __iter__(self):
        # Separate deterministic streams make P1/P2 differ only in the
        # foreground index distribution; original indices are identical.
        original_rng = np.random.default_rng(self.seed + self.epoch * 2 + 0)
        foreground_rng = np.random.default_rng(self.seed + self.epoch * 2 + 1)
        for _ in range(self.num_samples):
            yield self._draw(original_rng, self.original_mode), self._draw(foreground_rng, self.foreground_mode)


def stream_exposure(records, mode: str, seed: int, epoch: int = 1) -> dict:
    sampler = CTDFSDualStreamSampler(records, mode, mode, seed)
    sampler.set_epoch(epoch)
    indices = list(iter(sampler))
    # For audit only, inspect the corresponding stream in the generated pairs.
    selected = [left if mode == sampler.original_mode else right for left, right in indices]
    frame = records.iloc[selected]
    return {
        "mode": mode,
        "samples": len(selected),
        "class_counts": {str(k): int(v) for k, v in frame.species_id.astype(str).value_counts().sort_index().items()},
        "track_counts": {str(k): int(v) for k, v in frame.group_id.astype(str).value_counts().sort_index().items()},
        "unique_classes": int(frame.species_id.nunique()),
        "unique_tracks": int(frame.group_id.nunique()),
    }
