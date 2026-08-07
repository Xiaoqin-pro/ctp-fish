import json
from pathlib import Path


def test_class_protocol_is_sixteen_unique_ids():
    path = Path(__file__).parents[1] / "splits" / "f4k16t_class_ids.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["class_count"] == 16
    assert len(payload["class_ids"]) == 16
    assert len(set(payload["class_ids"])) == 16

