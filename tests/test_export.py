from intersectionqa.config import DatasetConfig
from intersectionqa.export.jsonl import read_jsonl, read_object_validation_manifest, write_jsonl
from intersectionqa.pipeline import build_smoke_rows, validate_dataset_dir, write_smoke_dataset


def test_jsonl_export_round_trips(tmp_path):
    config = DatasetConfig(output_dir=tmp_path)
    rows, report = build_smoke_rows(config)
    path = tmp_path / "rows.jsonl"
    assert write_jsonl(rows, path) == len(rows)
    loaded = read_jsonl(path)
    assert [row.id for row in loaded] == [row.id for row in rows]
    assert report.leakage_audit_status == "pass"


def test_smoke_export_writes_manifests(tmp_path):
    config = DatasetConfig(output_dir=tmp_path)
    report = write_smoke_dataset(config)
    assert (tmp_path / "source_manifest.json").exists()
    assert (tmp_path / "failure_manifest.jsonl").exists()
    assert (tmp_path / "object_validation_manifest.jsonl").exists()
    assert (tmp_path / "split_manifest.json").exists()
    assert report.source_manifest_hash.startswith("sha256:")
    assert report.object_validation_records == 3
    assert len(validate_dataset_dir(tmp_path)) == 21
    validations = read_object_validation_manifest(tmp_path / "object_validation_manifest.jsonl")
    assert len(validations) == 3
    assert all(record.valid for record in validations)
