from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.ctdfs_sampler import CTDFSDualStreamSampler


def records():
    return pd.DataFrame([
        {"species_id": "1", "group_id": "1::a"}, {"species_id": "1", "group_id": "1::a"},
        {"species_id": "1", "group_id": "1::b"}, {"species_id": "1", "group_id": "1::b"},
        {"species_id": "2", "group_id": "2::a"}, {"species_id": "2", "group_id": "2::a"},
        {"species_id": "2", "group_id": "2::b"}, {"species_id": "2", "group_id": "2::b"},
    ])


def test_dual_sampler_is_deterministic():
    a = CTDFSDualStreamSampler(records(), "s1_track_uniform", "s3_class_track_uniform", 3407)
    b = CTDFSDualStreamSampler(records(), "s1_track_uniform", "s3_class_track_uniform", 3407)
    assert list(a) == list(b)


def test_p2_foreground_has_both_classes_and_tracks():
    sampler = CTDFSDualStreamSampler(records(), "s1_track_uniform", "s3_class_track_uniform", 3407, num_samples=200)
    pairs = list(sampler)
    selected = records().iloc[[right for _, right in pairs]]
    assert selected.species_id.nunique() == 2
    assert selected.group_id.nunique() == 4


def test_original_and_foreground_modes_are_explicit():
    sampler = CTDFSDualStreamSampler(records(), "s1_track_uniform", "s3_class_track_uniform", 17)
    assert sampler.original_mode == "s1_track_uniform"
    assert sampler.foreground_mode == "s3_class_track_uniform"
