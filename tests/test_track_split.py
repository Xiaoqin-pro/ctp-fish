import pandas as pd
from datasets.split_builder import build_outer_folds, build_track_level_split, validate_track_split

def metadata():
    rows=[]
    for species in ["a","b"]:
        for track in range(15):
            for frame in range(2): rows.append({"image_path":f"{species}-{track}-{frame}","species_id":species,"species_name":species,"group_id":f"{species}::{track}"})
    return pd.DataFrame(rows)

def test_track_split_never_crosses_and_is_repeatable():
    first=build_track_level_split(metadata(),3407,(.7,.15,.15)); second=build_track_level_split(metadata(),3407,(.7,.15,.15))
    validate_track_split(first); assert first.equals(second)

def test_outer_folds_partition_groups():
    folds=build_outer_folds(metadata(),3407,3)["outer_folds"]; ids=[x for fold in folds.values() for x in fold]
    assert len(ids)==len(set(ids))==30
