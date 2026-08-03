import json
from pathlib import Path


def test_inner_folds_are_group_disjoint_and_cover_all_classes():
    path = Path("splits/f4k16t_phase2_inner_2fold.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["official_test_accessed"] is False
    assert payload["outer_folds_accessed"] is False
    left = set(payload["folds"][0]["heldout_group_ids"])
    right = set(payload["folds"][1]["heldout_group_ids"])
    assert left.isdisjoint(right)
    for fold in payload["folds"]:
        assert min(fold["class_track_counts"].values()) >= 2
        assert set(fold["class_track_counts"]) == set(payload["class_ids"])
