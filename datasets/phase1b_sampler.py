"""Deterministic P-by-Q-by-K track-structured training batches."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class StructuredTrackBatchSampler(Sampler[list[int]]):
    """Draw P classes, Q distinct tracks/class, and K frames/track."""

    def __init__(self, records, p: int, q: int, k: int, seed: int, num_batches: int) -> None:
        self.records = records.reset_index(drop=True).copy()
        if "split" in self.records.columns and set(self.records["split"].astype(str)) != {"train"}:
            raise ValueError("Phase 1B structured sampler accepts train records only.")
        self.p, self.q, self.k = int(p), int(q), int(k)
        self.seed, self.num_batches, self.epoch = int(seed), int(num_batches), 0
        if min(self.p, self.q, self.k, self.num_batches) <= 0:
            raise ValueError("P, Q, K, and num_batches must be positive.")
        self.frames_by_track: dict[str, list[int]] = defaultdict(list)
        self.tracks_by_class: dict[str, list[str]] = defaultdict(list)
        for index, row in self.records.iterrows():
            self.frames_by_track[str(row.group_id)].append(int(index))
        for group, frames in self.frames_by_track.items():
            label = str(self.records.iloc[frames[0]].species_id)
            self.tracks_by_class[label].append(group)
        self.classes = sorted(label for label, tracks in self.tracks_by_class.items() if len(tracks) >= self.q)
        if len(self.classes) < self.p:
            raise ValueError("Not enough classes with Q distinct tracks for a structured batch.")

    @property
    def batch_size(self) -> int:
        return self.p * self.q * self.k

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.num_batches):
            selected_classes = rng.choice(self.classes, size=self.p, replace=False)
            batch: list[int] = []
            for label in selected_classes:
                tracks = rng.choice(sorted(self.tracks_by_class[str(label)]), size=self.q, replace=False)
                for group in tracks:
                    frames = self.frames_by_track[str(group)]
                    batch.extend(int(index) for index in rng.choice(frames, size=self.k, replace=len(frames) < self.k))
            yield batch
