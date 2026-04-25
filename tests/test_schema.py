import pytest

from intersectionqa.config import DatasetConfig
from intersectionqa.enums import TaskType
from intersectionqa.export.jsonl import validate_rows
from intersectionqa.prompts.materialize import materialize_rows
from intersectionqa.schema import PublicTaskRow
from intersectionqa.schema import Hashes, LabelPolicy, Transform
from intersectionqa.sources.synthetic import fixture_geometry_records
from intersectionqa.splits.grouped import assign_geometry_splits


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


def test_public_repair_direction_row_validates_answer_against_metadata():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    splits = assign_geometry_splits(records, config.seed)
    row = materialize_rows(records, splits, [TaskType.REPAIR_DIRECTION])[0]
    data = row.model_dump(mode="json")
    data["answer"] = "-z" if row.answer != "-z" else "+x"

    with pytest.raises(ValueError, match="repair_direction answer"):
        PublicTaskRow.model_validate(data)


def test_public_repair_direction_row_validates_vector_direction():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    splits = assign_geometry_splits(records, config.seed)
    row = materialize_rows(records, splits, [TaskType.REPAIR_DIRECTION])[0]
    data = row.model_dump(mode="json")
    selected_direction = data["metadata"]["selected_direction"]
    data["metadata"]["selected_translation_vector_mm"] = [0.0, 0.0, 0.0]
    data["metadata"]["selected_translation_vector_mm"][0 if selected_direction != "+x" else 1] = 1.0
    data["metadata"]["selected_magnitude_mm"] = 1.0

    with pytest.raises(ValueError, match="repair_direction vector"):
        PublicTaskRow.model_validate(data)


def test_public_repair_direction_row_requires_selected_vector_to_match_candidate():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    splits = assign_geometry_splits(records, config.seed)
    row = materialize_rows(records, splits, [TaskType.REPAIR_DIRECTION])[0]
    data = row.model_dump(mode="json")
    direction = data["metadata"]["selected_direction"]
    candidate = next(
        item
        for item in data["metadata"]["candidate_moves"]
        if item["direction"] == direction
    )
    axis = {"+x": 0, "-x": 0, "+y": 1, "-y": 1, "+z": 2, "-z": 2}[direction]
    sign = 1.0 if direction.startswith("+") else -1.0
    data["metadata"]["selected_translation_vector_mm"] = [0.0, 0.0, 0.0]
    data["metadata"]["selected_translation_vector_mm"][axis] = sign * (
        candidate["magnitude_mm"] + 1.0
    )
    data["metadata"]["selected_magnitude_mm"] = candidate["magnitude_mm"] + 1.0

    with pytest.raises(ValueError, match="selected vector"):
        PublicTaskRow.model_validate(data)


def test_public_repair_translation_row_validates_answer_against_metadata():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    splits = assign_geometry_splits(records, config.seed)
    row = materialize_rows(records, splits, [TaskType.REPAIR_TRANSLATION])[0]
    data = row.model_dump(mode="json")
    data["answer"] = data["answer"].replace(" ", "  ")

    with pytest.raises(ValueError, match="repair_translation answer"):
        PublicTaskRow.model_validate(data)


def test_validate_rows_recomputes_repair_metadata_from_stored_bboxes():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    splits = assign_geometry_splits(records, config.seed)
    row = materialize_rows(records, splits, [TaskType.REPAIR_DIRECTION])[0]
    data = row.model_dump(mode="json")
    candidate = next(
        item
        for item in data["metadata"]["candidate_moves"]
        if item["direction"] != data["metadata"]["selected_direction"]
    )
    candidate["magnitude_mm"] += 1.0
    axis = {"+x": 0, "-x": 0, "+y": 1, "-y": 1, "+z": 2, "-z": 2}[candidate["direction"]]
    sign = 1.0 if candidate["direction"].startswith("+") else -1.0
    candidate["translation_vector_mm"][axis] = sign * candidate["magnitude_mm"]
    tampered = PublicTaskRow.model_validate(data)

    with pytest.raises(ValueError, match="candidate magnitude"):
        validate_rows([tampered])
