from datasets.f4k_metadata import ParsedRecord

def test_group_id_is_class_scoped():
    base=dict(image_path="a",mask_path=None,species_name="x",fish_id_raw=None,filename="a",file_sha256="h",width=1,height=1)
    a=ParsedRecord(species_id="1",tracking_id_raw="9",**base); b=ParsedRecord(species_id="2",tracking_id_raw="9",**base)
    assert a.group_id == "1::9" and a.group_id != b.group_id
