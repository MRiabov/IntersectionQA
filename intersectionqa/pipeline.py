"""End-to-end smoke pipeline composition."""

from __future__ import annotations

import json
from pathlib import Path

from intersectionqa.config import DatasetConfig
from intersectionqa.export.jsonl import (
    build_metadata,
    source_manifest_hash,
    validate_rows,
    write_metadata,
    write_schema,
    write_split_files,
)
from intersectionqa.prompts.materialize import materialize_rows
from intersectionqa.schema import GeometryRecord, PublicTaskRow
from intersectionqa.sources.cadevolve import CadevolveTarLoader
from intersectionqa.sources.synthetic import fixture_geometry_records
from intersectionqa.splits.grouped import assign_geometry_splits, audit_group_leakage, split_manifest


def build_smoke_geometry(config: DatasetConfig) -> tuple[list[GeometryRecord], dict[str, object]]:
    cadevolve = CadevolveTarLoader(config.cadevolve_archive, config.config_hash).load(
        limit=config.smoke.geometry_limit
    )
    records: list[GeometryRecord] = []
    if config.smoke.use_synthetic_fixtures:
        records.extend(fixture_geometry_records(config.label_policy, config.config_hash))
    report = {
        "cadevolve_archive_members_scanned": cadevolve.scanned_count,
        "cadevolve_source_records_loaded": len(cadevolve.records),
        "cadevolve_source_failures": len(cadevolve.failures),
        "geometry_records": len(records),
        "synthetic_fixture_count": len(records),
        "label_policy": config.label_policy.model_dump(mode="json"),
        "seed": config.seed,
        "config_hash": config.config_hash,
    }
    return records, report


def build_smoke_rows(config: DatasetConfig) -> tuple[list[PublicTaskRow], dict[str, object]]:
    geometry_records, report = build_smoke_geometry(config)
    split_by_geometry_id = assign_geometry_splits(geometry_records, config.seed)
    rows = materialize_rows(geometry_records, split_by_geometry_id, config.smoke.task_types)
    validate_rows(rows)
    audit = audit_group_leakage(rows)
    report.update(
        {
            "task_rows": len(rows),
            "task_counts": _counts(row.task_type for row in rows),
            "relation_counts": _counts(row.labels.relation for row in rows),
            "split_counts": _counts(row.split for row in rows),
            "leakage_audit_status": audit.status,
            "leakage_violation_count": audit.violation_count,
        }
    )
    return rows, report


def write_smoke_dataset(config: DatasetConfig) -> dict[str, object]:
    rows, report = build_smoke_rows(config)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split_summary = write_split_files(rows, output_dir)
    write_schema(output_dir / "schema.json")
    manifest = split_manifest(rows)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_hash = source_manifest_hash(
        {
            "source": "synthetic_fixture_fallback",
            "geometry_ids": [row.geometry_ids[0] for row in rows],
            "config_hash": config.config_hash,
        }
    )
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
    report["source_manifest_hash"] = source_hash
    report["output_dir"] = str(output_dir)
    (output_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def validate_dataset_dir(path: Path) -> list[PublicTaskRow]:
    rows: list[PublicTaskRow] = []
    for file_path in sorted(path.glob("*.jsonl")):
        if file_path.name == "failure_manifest.jsonl":
            continue
        from intersectionqa.export.jsonl import read_jsonl

        rows.extend(read_jsonl(file_path))
    validate_rows(rows)
    audit = audit_group_leakage(rows)
    if audit.status != "pass":
        raise ValueError(f"group leakage audit failed: {audit.violations}")
    return rows


def _counts(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:  # type: ignore[assignment]
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))
