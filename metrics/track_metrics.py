"""Trajectory-balanced metrics and track-unit bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd


def track_balanced_accuracy(predictions: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    required = {"group_id", "correct"}
    if not required.issubset(predictions): raise ValueError(f"Missing columns: {required - set(predictions.columns)}")
    per_track = predictions.groupby("group_id", as_index=False)["correct"].mean().rename(columns={"correct": "track_accuracy"})
    return float(per_track.track_accuracy.mean()), per_track


def cluster_bootstrap_mean(per_track: pd.DataFrame, seed: int, samples: int = 1000) -> tuple[float, float]:
    values = per_track.track_accuracy.to_numpy(float)
    if len(values) == 0: raise ValueError("No tracks for bootstrap.")
    rng = np.random.default_rng(seed); means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)])
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))
