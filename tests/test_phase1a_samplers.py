from __future__ import annotations

import pandas as pd

from datasets.phase1a_samplers import Phase1ASampler


def _records():
    return pd.DataFrame({"species_id":["a","a","a","b","b"],"group_id":["a1","a1","a2","b1","b1"]})


def test_phase1a_sampler_is_deterministic_per_epoch():
    first=Phase1ASampler(_records(),"s3_class_track_uniform",407,num_samples=30); second=Phase1ASampler(_records(),"s3_class_track_uniform",407,num_samples=30)
    first.set_epoch(2); second.set_epoch(2)
    assert list(first) == list(second)


def test_track_uniform_sampler_can_sample_short_track_equally():
    sampler=Phase1ASampler(_records(),"s1_track_uniform",9,num_samples=1000)
    values=list(sampler); assert any(index == 2 for index in values)
