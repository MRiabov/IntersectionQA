"""End-to-end smoke pipeline composition."""

from __future__ import annotations

from pathlib import Path

from intersectionqa.enums import AuditStatus
from intersectionqa.config import DatasetConfig
from intersectionqa.export.jsonl import (
    build_metadata,
    read_failure_manifest,
    read_metadata,
    read_object_validation_manifest,
    read_source_manifest,
    source_manifest_hash,
    validate_rows,
    write_failure_manifest,
    write_metadata,
    write_schema,
    write_split_files,
    write_source_manifest,
    write_jsonl_like,
)
from intersectionqa.prompts.materialize import materialize_rows
from intersectionqa.schema import (
    GeometryRecord,
    PublicTaskRow,
    SmokeDatasetReport,
    SmokeGeometryReport,
    SmokeRowsReport,
    SourceManifest,
    SourceManifestEntry,
)
from intersectionqa.sources.cadevolve import CadevolveTarLoader
from intersectionqa.sources.synthetic import fixture_geometry_records, synthetic_source_records
from intersectionqa.sources.validation import validate_source_object
from intersectionqa.splits.grouped import (
    DEFAULT_SPLITS,
    assign_geometry_splits,
    audit_group_leakage,
    split_manifest,
)


def build_smoke_geometry(config: DatasetConfig) -> tuple[list[GeometryRecord], SmokeGeometryReport]:
    cadevolve = CadevolveTarLoader(config.cadevolve_archive, config.config_hash).load(
        limit=config.smoke.geometry_limit
    )
    records: list[GeometryRecord] = []
    if config.smoke.use_synthetic_fixtures:
        records.extend(fixture_geometry_records(config.label_policy, config.config_hash))
    report = SmokeGeometryReport(
        cadevolve_archive_members_scanned=cadevolve.scanned_count,
        cadevolve_source_records_loaded=len(cadevolve.records),
        cadevolve_source_failures=len(cadevolve.failures),
        geometry_records=len(records),
        synthetic_fixture_count=len(records),
        label_policy=config.label_policy,
        seed=config.seed,
        config_hash=config.config_hash,
        source_manifest=_source_manifest(
            config, cadevolve.scanned_count, len(cadevolve.records), len(records)
        ),
    )
    return records, report


def build_smoke_rows(config: DatasetConfig) -> tuple[list[PublicTaskRow], SmokeRowsReport]:
    geometry_records, report = build_smoke_geometry(config)
    split_by_geometry_id = assign_geometry_splits(geometry_records, config.seed)
    rows = materialize_rows(geometry_records, split_by_geometry_id, config.smoke.task_types)
    validate_rows(rows)
    audit = audit_group_leakage(rows)
    return rows, SmokeRowsReport(
        **report.model_dump(),
        task_rows=len(rows),
        task_counts=_counts(row.task_type for row in rows),
        relation_counts=_counts(row.labels.relation for row in rows),
        split_counts=_counts(row.split for row in rows),
        leakage_audit_status=audit.status,
        leakage_violation_count=audit.violation_count,
    )


def write_smoke_dataset(config: DatasetConfig) -> SmokeDatasetReport:
    rows, report = build_smoke_rows(config)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split_summary = write_split_files(rows, output_dir)
    write_schema(output_dir / "schema.json")
    object_validations = [
        validate_source_object(
            source,
            config_hash=config.config_hash,
            validated_at_version="intersectionqa:v0.1-smoke",
        )
        for source in synthetic_source_records()
    ]
    write_jsonl_like(object_validations, output_dir / "object_validation_manifest.jsonl")
    manifest = split_manifest(rows)
    (output_dir / "split_manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    source_manifest = report.source_manifest
    source_manifest.sources[1].object_validation_records = len(object_validations)
    source_hash = source_manifest_hash(source_manifest)
    write_source_manifest(source_manifest, output_dir / "source_manifest.json")
    write_failure_manifest([], output_dir / "failure_manifest.jsonl")
    metadata = build_metadata(
        dataset_version=config.dataset_version,
        config_hash=config.config_hash,
        source_manifest_hash=source_hash,
        label_policy=config.label_policy,
        splits=split_summary,
        rows=rows,
        license=config.license,
    )
    write_metadata(metadata, output_dir / "metadata.json")
    smoke_report = SmokeDatasetReport(
        **report.model_dump(),
        source_manifest_hash=source_hash,
        object_validation_records=len(object_validations),
        output_dir=str(output_dir),
    )
    (output_dir / "smoke_report.json").write_text(
        smoke_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return smoke_report


def _source_manifest(
    config: DatasetConfig,
    cadevolve_scanned_count: int,
    cadevolve_source_count: int,
    synthetic_fixture_count: int,
) -> SourceManifest:
    return SourceManifest(
        dataset_version=config.dataset_version,
        config_hash=config.config_hash,
        sources=[
            SourceManifestEntry(
                source="cadevolve",
                archive_path=str(config.cadevolve_archive) if config.cadevolve_archive else None,
                archive_available=bool(
                    config.cadevolve_archive and config.cadevolve_archive.exists()
                ),
                archive_members_scanned=cadevolve_scanned_count,
                source_records_loaded=cadevolve_source_count,
                execution_policy="not_executed_in_mvp_smoke_process",
            ),
            SourceManifestEntry(
                source="synthetic",
                purpose="golden_and_smoke_fallback_only",
                fixture_count=synthetic_fixture_count,
                generator_id="gen_synthetic_primitives_v01",
            ),
        ],
    )


def validate_dataset_dir(path: Path) -> list[PublicTaskRow]:
    rows: list[PublicTaskRow] = []
    split_files = {f"{split}.jsonl" for split in DEFAULT_SPLITS}
    for file_path in sorted(path.glob("*.jsonl")):
        if file_path.name not in split_files:
            continue
        from intersectionqa.export.jsonl import read_jsonl

        rows.extend(read_jsonl(file_path))
    read_object_validation_manifest(path / "object_validation_manifest.jsonl")
    read_failure_manifest(path / "failure_manifest.jsonl")
    read_source_manifest(path / "source_manifest.json")
    read_metadata(path / "metadata.json")
    validate_rows(rows)
    audit = audit_group_leakage(rows)
    if audit.status != AuditStatus.PASS:
        raise ValueError(f"group leakage audit failed: {audit.violations}")
    return rows


def _counts(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:  # type: ignore[assignment]
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))
