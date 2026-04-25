from collections import Counter

import pytest

from intersectionqa.config import DatasetConfig, SmokeConfig
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
