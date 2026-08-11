"""Deterministic, interpretable Phase 1A sampling distributions."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class Phase1ASampler(Sampler[int]):
    """Sample indices while isolating frame, track, and class frequency effects."""

    def __init__(self, records, mode: str, seed: int, num_samples: int | None = None) -> None:
        if mode not in {"s1_track_uniform", "s2_class_uniform", "s3_class_track_uniform"}: raise ValueError(f"Unsupported Phase 1A sampler: {mode}")
        self.mode=mode; self.seed=int(seed); self.epoch=0; self.num_samples=int(num_samples or len(records)); self.records=records.reset_index(drop=True)
        self.by_track=defaultdict(list); self.by_class=defaultdict(list); self.tracks_by_class=defaultdict(list)
        for index,row in self.records.iterrows():
            label=str(row.species_id); group=str(row.group_id); self.by_track[group].append(index); self.by_class[label].append(index)
        for group,indices in self.by_track.items(): self.tracks_by_class[str(self.records.iloc[indices[0]].species_id)].append(group)
        self.tracks=sorted(self.by_track); self.classes=sorted(self.by_class)
        if not self.tracks or not self.classes: raise ValueError("Phase 1A sampler requires non-empty classes and tracks.")

    def set_epoch(self, epoch: int) -> None: self.epoch=int(epoch)
    def __len__(self) -> int: return self.num_samples

    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch)
        for _ in range(self.num_samples):
            if self.mode == "s1_track_uniform": yield int(rng.choice(self.by_track[str(rng.choice(self.tracks))]))
            elif self.mode == "s2_class_uniform": yield int(rng.choice(self.by_class[str(rng.choice(self.classes))]))
            else:
                label=str(rng.choice(self.classes)); group=str(rng.choice(sorted(self.tracks_by_class[label]))); yield int(rng.choice(self.by_track[group]))
