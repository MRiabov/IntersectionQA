"""End-to-end smoke pipeline composition."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

from intersectionqa.enums import AuditStatus, FailureStage
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
from intersectionqa.export.dataset_card import write_dataset_card
from intersectionqa.export.parquet import write_parquet_files
from intersectionqa.generation.cadevolve_assemblies import generate_cadevolve_geometry_records
from intersectionqa.generation.geometry_cache import GeometryLabelCache
from intersectionqa.prompts.materialize import materialize_rows
from intersectionqa.schema import (
    GeometryRecord,
    FailureRecord,
    ObjectValidationRecord,
    PublicTaskRow,
    SmokeDatasetReport,
    SmokeGeometryReport,
    SmokeRowsReport,
    SourceManifest,
    SourceManifestEntry,
    SourceObjectRecord,
)
from intersectionqa.sources.cadevolve import CadevolveTarLoader
from intersectionqa.sources.synthetic import fixture_geometry_records, synthetic_source_records
from intersectionqa.sources.validation import validate_source_objects_bounded
from intersectionqa.sources.validation_cache import (
    ObjectValidationCache,
    object_validation_cache_key,
)
from intersectionqa.splits.grouped import (
    DEFAULT_SPLITS,
    assign_geometry_splits,
    audit_group_leakage,
    split_manifest,
)


@dataclass(frozen=True)
class _SmokeGeometryArtifacts:
    records: list[GeometryRecord]
    report: SmokeGeometryReport
    source_records: list[SourceObjectRecord]
    object_validations: list[ObjectValidationRecord]
    failures: list[FailureRecord]


def build_smoke_geometry(config: DatasetConfig) -> tuple[list[GeometryRecord], SmokeGeometryReport]:
    artifacts = _build_smoke_geometry_artifacts(config)
    return artifacts.records, artifacts.report


def _build_smoke_geometry_artifacts(config: DatasetConfig) -> _SmokeGeometryArtifacts:
    started = time.monotonic()
    _progress(
        "loading CADEvolve sources: "
        f"source_dir={config.smoke.cadevolve_source_dir}, "
        f"archive={config.cadevolve_archive}, "
        f"source_cache_root={config.smoke.cadevolve_source_cache_root}, "
        f"limit={config.smoke.object_validation_limit}"
    )
    cadevolve = _load_cadevolve_for_smoke(config)
    _progress(
        f"loaded CADEvolve sources: {len(cadevolve.records)} records "
        f"from {cadevolve.scanned_count} scanned members, elapsed={_elapsed(started)}"
    )
    records: list[GeometryRecord] = []
    source_records: list[SourceObjectRecord] = []
    object_validations: list[ObjectValidationRecord] = []
    failures: list[FailureRecord] = [*cadevolve.failures]

    if config.smoke.include_cadevolve_if_available:
        cadevolve_validations = _validate_sources_for_smoke(config, cadevolve.records)
        source_records.extend(cadevolve.records)
        object_validations.extend(cadevolve_validations)
        failures.extend(_object_validation_failures(cadevolve.records, cadevolve_validations))
        validations_by_id = {validation.object_id: validation for validation in cadevolve_validations}
        cadevolve_generation = generate_cadevolve_geometry_records(
            cadevolve.records,
            validations_by_id,
            policy=config.label_policy,
            config_hash=config.config_hash,
            max_records=config.smoke.geometry_limit,
            geometry_cache=_geometry_label_cache_for_smoke(config),
            relation_balance=config.smoke.balance_geometry_relations,
            candidate_pool_multiplier=config.smoke.geometry_candidate_pool_multiplier,
        )
        records.extend(cadevolve_generation.records)
        failures.extend(cadevolve_generation.failures)
        _progress(
            "CADEvolve geometry generation complete: "
            f"{len(cadevolve_generation.records)} records, "
            f"{len(cadevolve_generation.failures)} generation failures, "
            f"elapsed={_elapsed(started)}"
        )

    synthetic_fixture_count = 0
    if config.smoke.use_synthetic_fixtures:
        remaining = max(0, config.smoke.geometry_limit - len(records))
        if remaining:
            synthetic_records = fixture_geometry_records(config.label_policy, config.config_hash)[:remaining]
            synthetic_fixture_count = len(synthetic_records)
            synthetic_sources = synthetic_source_records()
            source_records.extend(synthetic_sources)
            object_validations.extend(_validate_sources_for_smoke(config, synthetic_sources))
            records.extend(synthetic_records)

    source_manifest = _source_manifest(
        config,
        cadevolve.scanned_count,
        len(cadevolve.records),
        synthetic_fixture_count,
    )
    _set_source_validation_counts(source_manifest, source_records)
    report = SmokeGeometryReport(
        cadevolve_archive_members_scanned=cadevolve.scanned_count,
        cadevolve_source_records_loaded=len(cadevolve.records),
        cadevolve_source_failures=len(cadevolve.failures),
        geometry_records=len(records),
        synthetic_fixture_count=synthetic_fixture_count,
        label_policy=config.label_policy,
        seed=config.seed,
        config_hash=config.config_hash,
        source_manifest=source_manifest,
    )
    _progress(f"smoke geometry build complete: {len(records)} records, elapsed={_elapsed(started)}")
    return _SmokeGeometryArtifacts(
        records=records,
        report=report,
        source_records=source_records,
        object_validations=object_validations,
        failures=failures,
    )


def build_smoke_rows(config: DatasetConfig) -> tuple[list[PublicTaskRow], SmokeRowsReport]:
    artifacts = _build_smoke_geometry_artifacts(config)
    split_by_geometry_id = assign_geometry_splits(artifacts.records, config.seed)
    rows = materialize_rows(artifacts.records, split_by_geometry_id, config.smoke.task_types)
    validate_rows(rows)
    audit = audit_group_leakage(rows)
    return rows, SmokeRowsReport(
        **artifacts.report.model_dump(),
        task_rows=len(rows),
        task_counts=_counts(row.task_type for row in rows),
        relation_counts=_counts(row.labels.relation for row in rows),
        split_counts=_counts(row.split for row in rows),
        leakage_audit_status=audit.status,
        leakage_violation_count=audit.violation_count,
    )


def write_smoke_dataset(config: DatasetConfig) -> SmokeDatasetReport:
    started = time.monotonic()
    artifacts = _build_smoke_geometry_artifacts(config)
    split_by_geometry_id = assign_geometry_splits(artifacts.records, config.seed)
    rows = materialize_rows(artifacts.records, split_by_geometry_id, config.smoke.task_types)
    validate_rows(rows)
    audit = audit_group_leakage(rows)
    _progress(
        f"materialized rows: {len(rows)} task rows from {len(artifacts.records)} geometry records, "
        f"leakage={audit.status}, elapsed={_elapsed(started)}"
    )
    report = SmokeRowsReport(
        **artifacts.report.model_dump(),
        task_rows=len(rows),
        task_counts=_counts(row.task_type for row in rows),
        relation_counts=_counts(row.labels.relation for row in rows),
        split_counts=_counts(row.split for row in rows),
        leakage_audit_status=audit.status,
        leakage_violation_count=audit.violation_count,
    )
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split_summary = write_split_files(rows, output_dir)
    write_schema(output_dir / "schema.json")
    write_jsonl_like(artifacts.object_validations, output_dir / "object_validation_manifest.jsonl")
    manifest = split_manifest(rows)
    (output_dir / "split_manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    source_manifest = report.source_manifest
    source_hash = source_manifest_hash(source_manifest)
    write_source_manifest(source_manifest, output_dir / "source_manifest.json")
    write_failure_manifest(artifacts.failures, output_dir / "failure_manifest.jsonl")
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
    write_dataset_card(output_dir)
    parquet_counts = write_parquet_files(rows, output_dir / "parquet")
    (output_dir / "parquet_manifest.json").write_text(
        json.dumps({"files": parquet_counts, "compression": "zstd"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    smoke_report = SmokeDatasetReport(
        **report.model_dump(),
        source_manifest_hash=source_hash,
        object_validation_records=len(artifacts.object_validations),
        output_dir=str(output_dir),
    )
    (output_dir / "smoke_report.json").write_text(
        smoke_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    _progress(f"dataset export complete: output_dir={output_dir}, elapsed={_elapsed(started)}")
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
                archive_path=_cadevolve_archive_path_for_manifest(_cadevolve_source_dir_for_smoke(config)),
                archive_available=bool(
                    config.cadevolve_archive and config.cadevolve_archive.exists()
                ),
                source_dir=str(_cadevolve_source_dir_for_smoke(config))
                if _cadevolve_source_dir_for_smoke(config) is not None
                else None,
                archive_members_scanned=cadevolve_scanned_count,
                source_records_loaded=cadevolve_source_count,
                execution_policy="cadquery_validation_and_bbox_guided_geometry_generation_enabled",
            ),
            SourceManifestEntry(
                source="synthetic",
                purpose="golden_and_smoke_fallback_only",
                fixture_count=synthetic_fixture_count,
                generator_id="gen_synthetic_primitives_v01",
            ),
        ],
    )


def _load_cadevolve_for_smoke(config: DatasetConfig):
    limit = config.smoke.object_validation_limit
    offset = config.smoke.object_validation_offset
    member_index_cache_dir = (
        config.smoke.source_member_index_cache_dir
        if config.smoke.use_source_member_index_cache
        else None
    )
    extracted_source_cache_dir = (
        config.smoke.extracted_source_cache_dir
        if config.smoke.use_extracted_source_cache
        else None
    )
    return CadevolveTarLoader(
        config.cadevolve_archive,
        config.config_hash,
        member_index_cache_dir=member_index_cache_dir,
        extracted_source_cache_dir=extracted_source_cache_dir,
        extracted_source_cache_root=config.smoke.cadevolve_source_cache_root,
        source_dir=_cadevolve_source_dir_for_smoke(config),
    ).load(limit=limit, offset=offset)


def _cadevolve_source_dir_for_smoke(config: DatasetConfig) -> Path | None:
    return config.smoke.cadevolve_source_dir or config.smoke.cadevolve_source_cache_root


def _cadevolve_archive_path_for_manifest(source_dir: Path | None) -> str | None:
    if source_dir is None:
        return None
    manifest_path = source_dir / "extraction_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    archive = manifest.get("archive")
    if isinstance(archive, dict) and isinstance(archive.get("path"), str):
        return archive["path"]
    return None


def _geometry_label_cache_for_smoke(config: DatasetConfig) -> GeometryLabelCache | None:
    if not config.smoke.use_geometry_label_cache:
        return None
    return GeometryLabelCache(config.smoke.geometry_label_cache_dir)


def _validate_sources_for_smoke(
    config: DatasetConfig,
    source_records: list[SourceObjectRecord],
) -> list[ObjectValidationRecord]:
    started = time.monotonic()
    total = len(source_records)
    validation_version = "intersectionqa:v0.1-smoke"
    cache = (
        ObjectValidationCache(config.smoke.object_validation_cache_dir)
        if config.smoke.use_object_validation_cache
        else None
    )
    if total:
        cache_status = f"cache={config.smoke.object_validation_cache_dir}" if cache else "cache=disabled"
        _progress(
            f"validating source objects: 0/{total}, {cache_status}, "
            f"workers={config.smoke.object_validation_worker_count}"
        )
    records: list[ObjectValidationRecord | None] = [None] * total
    missing: list[tuple[int, SourceObjectRecord, str]] = []
    cache_hits = 0
    for index, source in enumerate(source_records, start=1):
        cache_key = object_validation_cache_key(
            source,
            timeout_seconds=config.smoke.object_validation_timeout_seconds,
            validated_at_version=validation_version,
        )
        cached = cache.get(cache_key) if cache else None
        if cached is not None:
            records[index - 1] = _object_validation_with_config_hash(cached, config.config_hash)
            cache_hits += 1
        else:
            missing.append((index - 1, source, cache_key))

    if missing:
        worker_count = max(1, config.smoke.object_validation_worker_count)

        def progress(completed: int, missing_total: int) -> None:
            overall_completed = cache_hits + completed
            if overall_completed == total or overall_completed % 10 == 0 or completed == missing_total:
                _progress(
                    f"validating source objects: {overall_completed}/{total}, "
                    f"cache_hits={cache_hits}, workers={worker_count}, elapsed={_elapsed(started)}"
                )

        validated_missing = validate_source_objects_bounded(
            [source for _, source, _ in missing],
            config_hash=config.config_hash,
            validated_at_version=validation_version,
            timeout_seconds=config.smoke.object_validation_timeout_seconds,
            worker_count=worker_count,
            progress=progress,
        )
        for (index, _source, cache_key), record in zip(missing, validated_missing, strict=True):
            records[index] = record
            if cache is not None:
                cache.set(cache_key, record)
    else:
        _progress(
            f"validating source objects: {total}/{total}, cache_hits={cache_hits}, "
            f"elapsed={_elapsed(started)}"
        )

    completed_records = [record for record in records if record is not None]
    if total:
        valid_count = sum(1 for record in completed_records if record.valid)
        _progress(
            f"validating source objects complete: {len(completed_records)}/{total}, "
            f"valid={valid_count}, invalid={len(completed_records) - valid_count}, "
            f"cache_hits={cache_hits}, elapsed={_elapsed(started)}"
        )
    return completed_records


def _object_validation_with_config_hash(
    record: ObjectValidationRecord,
    config_hash: str,
) -> ObjectValidationRecord:
    if record.hashes.config_hash == config_hash:
        return record
    return record.model_copy(
        update={"hashes": record.hashes.model_copy(update={"config_hash": config_hash})}
    )


def _set_source_validation_counts(
    source_manifest: SourceManifest,
    source_records: list[SourceObjectRecord],
) -> None:
    counts = _counts(record.source for record in source_records)
    for entry in source_manifest.sources:
        entry.object_validation_records = counts.get(entry.source, 0)


def _object_validation_failures(
    sources: list[SourceObjectRecord],
    validations: object,
) -> list[FailureRecord]:
    source_by_id = {source.object_id: source for source in sources}
    failures: list[FailureRecord] = []
    for index, validation in enumerate(validations, start=1):  # type: ignore[assignment]
        if validation.valid:
            continue
        source = source_by_id.get(validation.object_id)
        failures.append(
            FailureRecord(
                failure_id=f"fail_{index:06d}",
                stage=FailureStage.OBJECT_VALIDATION,
                source=source.source if source else None,
                source_id=source.source_id if source else None,
                object_id=validation.object_id,
                geometry_id=None,
                failure_reason=validation.failure_reason,
                error_summary=f"Object validation failed: {validation.failure_reason}",
                retry_count=0,
                hashes=validation.hashes,
            )
        )
    return failures


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


def _progress(message: str) -> None:
    print(f"[intersectionqa] {message}", file=sys.stderr, flush=True)


def _elapsed(started: float) -> str:
    return f"{time.monotonic() - started:.1f}s"
