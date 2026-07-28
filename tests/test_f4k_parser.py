import pytest
from datasets.f4k_metadata import compile_filename_parser

def test_parser_requires_named_groups():
    with pytest.raises(ValueError): compile_filename_parser(r"(?P<species_id>\d+)")

def test_parser_accepts_required_groups():
    parser=compile_filename_parser(r"(?P<species_id>\d+)_(?P<tracking_id>t\d+)")
    assert parser.search("5_t10.jpg") is not None
