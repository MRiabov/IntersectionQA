from collections import Counter

import pytest

from intersectionqa.config import DatasetConfig, SmokeConfig
from intersectionqa.enums import Relation, TaskType
from intersectionqa.export.balance import DEFAULT_RELATION_TARGETS, proportional_cap_counts
from intersectionqa.export.jsonl import (
    read_jsonl,
    read_metadata,
    read_object_validation_manifest,
    read_source_manifest,
    write_jsonl,
)
from intersectionqa.export.parquet import read_parquet_rows
from intersectionqa.pipeline import build_smoke_rows, validate_dataset_dir, write_smoke_dataset


def test_jsonl_export_round_trips(tmp_path):
    config = DatasetConfig(
        output_dir=tmp_path,
        smoke=SmokeConfig(include_cadevolve_if_available=False),
    )
    rows, report = build_smoke_rows(config)
    path = tmp_path / "rows.jsonl"
    assert write_jsonl(rows, path) == len(rows)
    loaded = read_jsonl(path)
    assert [row.id for row in loaded] == [row.id for row in rows]
    assert report.leakage_audit_status == "pass"


def test_smoke_export_writes_manifests(tmp_path):
    config = DatasetConfig(
        output_dir=tmp_path,
        smoke=SmokeConfig(include_cadevolve_if_available=False),
    )
    report = write_smoke_dataset(config)
    assert (tmp_path / "source_manifest.json").exists()
    assert (tmp_path / "failure_manifest.jsonl").exists()
    assert (tmp_path / "object_validation_manifest.jsonl").exists()
    assert (tmp_path / "split_manifest.json").exists()
    assert (tmp_path / "parquet_manifest.json").exists()
    assert (tmp_path / "class_balance_report.json").exists()
    assert (tmp_path / "DATASET_CARD.md").exists()
    assert (tmp_path / "parquet" / "train.parquet").exists()
    assert report.source_manifest_hash.startswith("sha256:")
    assert report.object_validation_records == 3
    assert len(validate_dataset_dir(tmp_path)) == 21
    validations = read_object_validation_manifest(tmp_path / "object_validation_manifest.jsonl")
    assert len(validations) == 3
    assert all(record.valid for record in validations)
    source_manifest = read_source_manifest(tmp_path / "source_manifest.json")
    assert source_manifest is not None
    assert source_manifest.sources[0].source == "cadevolve"
    metadata = read_metadata(tmp_path / "metadata.json")
    assert metadata is not None
    assert metadata.dataset_name == "IntersectionQA"
    parquet_rows = read_parquet_rows(tmp_path / "parquet" / "train.parquet")
    assert parquet_rows
    assert {"id", "prompt", "answer", "labels_json"} <= set(parquet_rows[0])


def test_smoke_export_can_opt_into_repair_direction(tmp_path):
    config = DatasetConfig(
        output_dir=tmp_path,
        smoke=SmokeConfig(
            include_cadevolve_if_available=False,
            task_types=[
                TaskType.BINARY_INTERFERENCE,
                TaskType.REPAIR_DIRECTION,
            ],
        ),
    )

    report = write_smoke_dataset(config)
    rows = validate_dataset_dir(tmp_path)
    repair_rows = [row for row in rows if row.task_type == TaskType.REPAIR_DIRECTION]
    metadata = read_metadata(tmp_path / "metadata.json")

    assert repair_rows
    assert report.task_counts["repair_direction"] == len(repair_rows)
    assert metadata is not None
    assert "repair_direction" in metadata.task_types
    assert metadata.counts.by_task["repair_direction"] == len(repair_rows)
    assert any("conservative AABB-separating" in item for item in metadata.known_limitations)
    assert all(row.id.startswith("intersectionedit_repair_direction_") for row in repair_rows)
    assert all(
        row.metadata["repair_policy"] == "conservative_aabb_separating_translation_v01"
        for row in repair_rows
    )


def test_class_balance_groups_repair_direction_with_single_geometry_tasks():
    config = DatasetConfig(
        smoke=SmokeConfig(
            include_cadevolve_if_available=False,
            task_types=[
                TaskType.BINARY_INTERFERENCE,
                TaskType.REPAIR_DIRECTION,
            ],
        ),
    )

    rows, _ = build_smoke_rows(config)
    by_geometry: dict[str, set[TaskType]] = {}
    for row in rows:
        by_geometry.setdefault(row.geometry_ids[0], set()).add(row.task_type)

    for row in rows:
        if row.task_type == TaskType.REPAIR_DIRECTION:
            assert TaskType.BINARY_INTERFERENCE in by_geometry[row.geometry_ids[0]]


def test_public_row_limit_caps_exported_rows(tmp_path):
    config = DatasetConfig(
        output_dir=tmp_path,
        smoke=SmokeConfig(
            include_cadevolve_if_available=False,
            public_row_limit=10,
        ),
    )

    report = write_smoke_dataset(config)
    rows = validate_dataset_dir(tmp_path)
    answer_counts = Counter((row.task_type, row.answer) for row in rows)

    assert report.task_rows == 10
    assert len(rows) == 10
    assert sum(report.task_counts.values()) == 10
    assert answer_counts[("binary_interference", "yes")] == answer_counts[("binary_interference", "no")]


def test_public_row_limit_fails_when_underproduced(tmp_path):
    config = DatasetConfig(
        output_dir=tmp_path,
        smoke=SmokeConfig(
            include_cadevolve_if_available=False,
            public_row_limit=100,
        ),
    )

    with pytest.raises(ValueError, match="public_row_limit=100"):
        write_smoke_dataset(config)


def test_proportional_cap_counts_is_idempotent_for_balanced_counts():
    available = {
        Relation.INTERSECTING: 1749,
        Relation.DISJOINT: 1312,
        Relation.NEAR_MISS: 656,
        Relation.TOUCHING: 656,
    }

    assert proportional_cap_counts(available, DEFAULT_RELATION_TARGETS) == available
