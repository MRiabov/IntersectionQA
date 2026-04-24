import pytest

from intersectionqa.schema import Hashes, LabelPolicy, Transform
from intersectionqa.sources.synthetic import fixture_geometry_records


def test_transform_requires_xyz_rotation_order():
    with pytest.raises(Exception):
        Transform(
            translation=(0.0, 0.0, 0.0),
            rotation_xyz_deg=(0.0, 0.0, 0.0),
            rotation_order="ZYX",
        )


def test_hashes_require_sha256_prefix():
    with pytest.raises(ValueError):
        Hashes(
            source_code_hash="abc",
            object_hash=None,
            transform_hash=None,
            geometry_hash=None,
            config_hash=None,
            prompt_hash=None,
        )


def test_synthetic_geometry_records_match_schema():
    records = fixture_geometry_records(LabelPolicy(), "sha256:" + "0" * 64)
    assert {record.labels.relation for record in records} >= {
        "disjoint",
        "touching",
        "near_miss",
        "intersecting",
        "contained",
    }
    assert all(record.diagnostics.label_status == "ok" for record in records)
