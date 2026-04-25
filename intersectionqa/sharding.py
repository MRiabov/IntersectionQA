"""Small deterministic shard runner for smoke dataset generation."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from intersectionqa.config import DatasetConfig
from intersectionqa.export.jsonl import (
    build_metadata,
    read_failure_manifest,
    read_jsonl,
    read_object_validation_manifest,
    read_source_manifest,
    source_manifest_hash,
    write_failure_manifest,
    write_jsonl_like,
    write_metadata,
    write_schema,
    write_source_manifest,
    write_split_files,
)
from intersectionqa.export.dataset_card import write_dataset_card
from intersectionqa.export.parquet import write_parquet_files
from intersectionqa.hashing import sha256_json
from intersectionqa.pipeline import validate_dataset_dir, write_smoke_dataset
from intersectionqa.prompts.common import TASK_PREFIX
from intersectionqa.schema import (
    FailureRecord,
    ObjectValidationRecord,
    PublicTaskRow,
    SourceManifest,
    SourceManifestEntry,
)
from intersectionqa.splits.grouped import ALL_SPLITS, split_manifest


class SourceShardSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shard_id: str
    index: int
    source_offset: int
    source_limit: int
    output_dir: Path


class SourceShardResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shard_id: str
    output_dir: Path
    status: str
    source_offset: int
    source_limit: int
    geometry_records: int | None = None
    task_rows: int | None = None
    validated_rows: int | None = None
    error: str | None = None


def build_source_shard_specs(
    output_dir: Path,
    *,
    shard_count: int,
    source_shard_size: int,
) -> list[SourceShardSpec]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if source_shard_size <= 0:
        raise ValueError("source_shard_size must be positive")
    return [
        SourceShardSpec(
            shard_id=f"shard_{index:04d}",
            index=index,
            source_offset=index * source_shard_size,
            source_limit=source_shard_size,
            output_dir=output_dir / "shards" / f"shard_{index:04d}",
        )
        for index in range(shard_count)
    ]


def config_for_source_shard(config: DatasetConfig, spec: SourceShardSpec) -> DatasetConfig:
    return config.model_copy(
        update={
            "output_dir": spec.output_dir,
            "smoke": config.smoke.model_copy(
                update={
                    "object_validation_offset": spec.source_offset,
                    "object_validation_limit": spec.source_limit,
                }
            ),
        }
    )


def generate_source_shards(
    config: DatasetConfig,
    output_dir: Path,
    *,
    shard_count: int,
    source_shard_size: int,
    force: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_source_shard_specs(
        output_dir,
        shard_count=shard_count,
        source_shard_size=source_shard_size,
    )
    return generate_source_shards_from_specs(config, output_dir, specs, force=force)


def generate_source_shards_from_specs(
    config: DatasetConfig,
    output_dir: Path,
    specs: list[SourceShardSpec],
    *,
    force: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[SourceShardResult] = []
    for spec in specs:
        existing = _validated_existing_shard(spec)
        if existing is not None and not force:
            results.append(existing)
            _write_shard_manifest(output_dir, specs, results)
            continue
        result = _generate_one_shard(config, spec)
        results.append(result)
        _write_shard_manifest(output_dir, specs, results)
    manifest = _manifest(output_dir, specs, results)
    _write_shard_manifest(output_dir, specs, results)
    return manifest


def seed_existing_dataset_shard(
    output_dir: Path,
    spec: SourceShardSpec,
    dataset_dir: Path,
    *,
    force: bool = False,
) -> SourceShardResult:
    rows = validate_dataset_dir(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if spec.output_dir.exists():
        existing = _validated_existing_shard(spec)
        if existing is not None and not force:
            return existing.model_copy(update={"status": "skipped_seed_existing_valid"})
        if force:
            shutil.rmtree(spec.output_dir)
        else:
            raise ValueError(f"seed shard output exists but is not valid: {spec.output_dir}")
    shutil.copytree(dataset_dir, spec.output_dir)
    return SourceShardResult(
        shard_id=spec.shard_id,
        output_dir=spec.output_dir,
        status="seeded_existing_valid",
        source_offset=spec.source_offset,
        source_limit=spec.source_limit,
        validated_rows=len(rows),
    )


def merge_validated_source_shards(
    shard_root: Path,
    output_dir: Path,
    *,
    config: DatasetConfig,
) -> dict[str, Any]:
    manifest_path = shard_root / "shard_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"missing shard manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_entries = [
        entry for entry in manifest.get("shards", []) if entry.get("status") != "failed"
    ]
    if not shard_entries:
        raise ValueError("no valid shards to merge")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[PublicTaskRow] = []
    validations: list[ObjectValidationRecord] = []
    failures: list[FailureRecord] = []
    source_manifests: list[SourceManifest] = []
    task_counters: dict[str, int] = defaultdict(int)
    for entry in sorted(shard_entries, key=lambda item: item["shard_id"]):
        shard_id = entry["shard_id"]
        shard_dir = Path(entry["output_dir"])
        validate_dataset_dir(shard_dir)
        rows.extend(_read_and_rewrite_shard_rows(shard_dir, shard_id, task_counters))
        validations.extend(_rewrite_object_validations(shard_dir, shard_id))
        failures.extend(_rewrite_failures(shard_dir, shard_id))
        source_manifest = read_source_manifest(shard_dir / "source_manifest.json")
        if source_manifest is not None:
            source_manifests.append(source_manifest)

    split_summary = write_split_files(rows, output_dir)
    write_schema(output_dir / "schema.json")
    write_jsonl_like(validations, output_dir / "object_validation_manifest.jsonl")
    write_failure_manifest(failures, output_dir / "failure_manifest.jsonl")
    merged_source_manifest = _merge_source_manifests(
        source_manifests,
        dataset_version=config.dataset_version,
        config_hash=config.config_hash,
    )
    source_hash = source_manifest_hash(merged_source_manifest)
    write_source_manifest(merged_source_manifest, output_dir / "source_manifest.json")
    (output_dir / "split_manifest.json").write_text(
        split_manifest(rows).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    write_metadata(
        build_metadata(
            dataset_version=config.dataset_version,
            config_hash=config.config_hash,
            source_manifest_hash=source_hash,
            label_policy=config.label_policy,
            splits=split_summary,
            rows=rows,
            license=config.license,
        ),
        output_dir / "metadata.json",
    )
    write_dataset_card(output_dir)
    parquet_counts = write_parquet_files(rows, output_dir / "parquet")
    (output_dir / "parquet_manifest.json").write_text(
        json.dumps({"files": parquet_counts, "compression": "zstd"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_dataset_dir(output_dir)
    result = {
        "schema": "intersectionqa_merged_source_shards_v1",
        "shard_root": str(shard_root),
        "output_dir": str(output_dir),
        "merged_shards": [entry["shard_id"] for entry in sorted(shard_entries, key=lambda item: item["shard_id"])],
        "row_count": len(rows),
        "object_validation_records": len(validations),
        "failure_records": len(failures),
        "source_manifest_hash": source_hash,
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _validated_existing_shard(spec: SourceShardSpec) -> SourceShardResult | None:
    if not (spec.output_dir / "metadata.json").exists():
        return None
    try:
        rows = validate_dataset_dir(spec.output_dir)
    except Exception:
        return None
    return SourceShardResult(
        shard_id=spec.shard_id,
        output_dir=spec.output_dir,
        status="skipped_existing_valid",
        source_offset=spec.source_offset,
        source_limit=spec.source_limit,
        validated_rows=len(rows),
    )


def _read_and_rewrite_shard_rows(
    shard_dir: Path,
    shard_id: str,
    task_counters: dict[str, int],
) -> list[PublicTaskRow]:
    rows: list[PublicTaskRow] = []
    for split in ALL_SPLITS:
        path = shard_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            task_counters[row.task_type] += 1
            rows.append(_rewrite_row(row, shard_id, task_counters[row.task_type]))
    return rows


def _rewrite_row(row: PublicTaskRow, shard_id: str, row_number: int) -> PublicTaskRow:
    geometry_ids = [_prefixed(shard_id, geometry_id) for geometry_id in row.geometry_ids]
    base_object_pair_id = _prefixed(shard_id, row.base_object_pair_id)
    assembly_group_id = _prefixed(shard_id, row.assembly_group_id)
    counterfactual_group_id = (
        _prefixed(shard_id, row.counterfactual_group_id) if row.counterfactual_group_id else None
    )
    variant_id = _prefixed(shard_id, row.variant_id) if row.variant_id else None
    metadata = {
        **row.metadata,
        "source_shard_id": shard_id,
        "original_row_id": row.id,
        "original_geometry_ids": row.geometry_ids,
        "split_group": counterfactual_group_id or assembly_group_id or base_object_pair_id,
    }
    prompt_template_version = metadata.get("prompt_template_version")
    prompt_hash = sha256_json(
        {
            "template_version": prompt_template_version,
            "task_type": row.task_type,
            "prompt": row.prompt,
            "geometry_ids": geometry_ids,
        }
    )
    return row.model_copy(
        update={
            "id": f"{TASK_PREFIX[row.task_type]}_{row_number:06d}",
            "geometry_ids": geometry_ids,
            "base_object_pair_id": base_object_pair_id,
            "assembly_group_id": assembly_group_id,
            "counterfactual_group_id": counterfactual_group_id,
            "variant_id": variant_id,
            "hashes": row.hashes.model_copy(update={"prompt_hash": prompt_hash}),
            "metadata": metadata,
        }
    )


def _rewrite_object_validations(shard_dir: Path, shard_id: str) -> list[ObjectValidationRecord]:
    return [
        record.model_copy(update={"object_id": _prefixed(shard_id, record.object_id)})
        for record in read_object_validation_manifest(shard_dir / "object_validation_manifest.jsonl")
    ]


def _rewrite_failures(shard_dir: Path, shard_id: str) -> list[FailureRecord]:
    failures: list[FailureRecord] = []
    for record in read_failure_manifest(shard_dir / "failure_manifest.jsonl"):
        failures.append(
            record.model_copy(
                update={
                    "failure_id": _prefixed(shard_id, record.failure_id),
                    "object_id": _prefix_compound_id(shard_id, record.object_id),
                    "geometry_id": _prefixed(shard_id, record.geometry_id) if record.geometry_id else None,
                }
            )
        )
    return failures


def _merge_source_manifests(
    manifests: list[SourceManifest],
    *,
    dataset_version: str,
    config_hash: str,
) -> SourceManifest:
    by_source: dict[str, SourceManifestEntry] = {}
    for manifest in manifests:
        for entry in manifest.sources:
            current = by_source.get(entry.source)
            if current is None:
                by_source[entry.source] = entry.model_copy()
                continue
            by_source[entry.source] = current.model_copy(
                update={
                    "archive_members_scanned": _sum_optional(
                        current.archive_members_scanned,
                        entry.archive_members_scanned,
                    ),
                    "source_records_loaded": _sum_optional(
                        current.source_records_loaded,
                        entry.source_records_loaded,
                    ),
                    "fixture_count": _sum_optional(current.fixture_count, entry.fixture_count),
                    "object_validation_records": _sum_optional(
                        current.object_validation_records,
                        entry.object_validation_records,
                    ),
                }
            )
    return SourceManifest(
        dataset_version=dataset_version,
        config_hash=config_hash,
        sources=[by_source[source] for source in sorted(by_source)],
    )


def _sum_optional(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


def _prefixed(shard_id: str, value: str) -> str:
    return f"{shard_id}__{value}"


def _prefix_compound_id(shard_id: str, value: str | None) -> str | None:
    if value is None:
        return None
    return "|".join(_prefixed(shard_id, part) for part in value.split("|"))


def _generate_one_shard(config: DatasetConfig, spec: SourceShardSpec) -> SourceShardResult:
    try:
        report = write_smoke_dataset(config_for_source_shard(config, spec))
        rows = validate_dataset_dir(spec.output_dir)
    except Exception as exc:
        return SourceShardResult(
            shard_id=spec.shard_id,
            output_dir=spec.output_dir,
            status="failed",
            source_offset=spec.source_offset,
            source_limit=spec.source_limit,
            error=str(exc),
        )
    return SourceShardResult(
        shard_id=spec.shard_id,
        output_dir=spec.output_dir,
        status="generated_valid",
        source_offset=spec.source_offset,
        source_limit=spec.source_limit,
        geometry_records=report.geometry_records,
        task_rows=report.task_rows,
        validated_rows=len(rows),
    )


def _write_shard_manifest(
    output_dir: Path,
    specs: list[SourceShardSpec],
    results: list[SourceShardResult],
) -> None:
    path = output_dir / "shard_manifest.json"
    path.write_text(json.dumps(_manifest(output_dir, specs, results), indent=2, sort_keys=True) + "\n")


def _manifest(
    output_dir: Path,
    specs: list[SourceShardSpec],
    results: list[SourceShardResult],
) -> dict[str, Any]:
    return {
        "schema": "intersectionqa_source_shards_v1",
        "output_dir": str(output_dir),
        "shard_count": len(specs),
        "completed_count": len(results),
        "all_valid": len(results) == len(specs) and all(result.status != "failed" for result in results),
        "shards": [result.model_dump(mode="json") for result in results],
        "pending_shards": [spec.shard_id for spec in specs[len(results) :]],
    }
