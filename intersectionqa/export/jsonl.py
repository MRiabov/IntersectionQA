"""JSONL export, validation, and metadata helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from intersectionqa.hashing import sha256_json
from intersectionqa.geometry.labels import validate_label_consistency
from intersectionqa.schema import (
    DatasetCounts,
    DatasetMetadata,
    FailureRecord,
    ObjectValidationRecord,
    LabelPolicy,
    PublicTaskRow,
    SourceManifest,
    SplitFileSummary,
)
from intersectionqa.splits.grouped import DEFAULT_SPLITS


def write_jsonl(rows: Iterable[PublicTaskRow], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json(exclude_none=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[PublicTaskRow]:
    rows: list[PublicTaskRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(PublicTaskRow.model_validate_json(line))
            except Exception as exc:  # pragma: no cover - exercised by CLI
                raise ValueError(f"{path}:{line_number}: invalid public row: {exc}") from exc
    return rows


def read_object_validation_manifest(path: Path) -> list[ObjectValidationRecord]:
    if not path.exists():
        return []
    records: list[ObjectValidationRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(ObjectValidationRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid object validation record: {exc}") from exc
    return records


def read_failure_manifest(path: Path) -> list[FailureRecord]:
    if not path.exists():
        return []
    records: list[FailureRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(FailureRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid failure record: {exc}") from exc
    return records


def write_split_files(rows: list[PublicTaskRow], output_dir: Path) -> dict[str, SplitFileSummary]:
    by_split: dict[str, list[PublicTaskRow]] = defaultdict(list)
    for row in rows:
        by_split[row.split].append(row)

    summary: dict[str, SplitFileSummary] = {}
    for split in DEFAULT_SPLITS:
        split_rows = by_split.get(split, [])
        split_rows = sorted(split_rows, key=lambda row: row.id)
        path = output_dir / f"{split}.jsonl"
        write_jsonl(split_rows, path)
        summary[split] = SplitFileSummary(
            path=path.name,
            row_count=len(split_rows),
            task_counts=dict(Counter(row.task_type for row in split_rows)),
            holdout_rule=_holdout_rule(split),
        )
    return summary


def validate_rows(rows: Iterable[PublicTaskRow]) -> None:
    seen: set[str] = set()
    for row in rows:
        if row.id in seen:
            raise ValueError(f"duplicate public row id: {row.id}")
        seen.add(row.id)
        PublicTaskRow.model_validate(row)
        validate_label_consistency(row.labels, row.diagnostics, row.label_policy)


def write_failure_manifest(failures: Iterable[FailureRecord], path: Path) -> int:
    return write_jsonl_like(failures, path)


def write_jsonl_like(records: Iterable[BaseModel], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json(exclude_none=False) + "\n")
            count += 1
    return count


def write_source_manifest(manifest: SourceManifest, path: Path) -> None:
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_source_manifest(path: Path) -> SourceManifest | None:
    if not path.exists():
        return None
    try:
        return SourceManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{path}: invalid source manifest: {exc}") from exc


def write_schema(path: Path) -> None:
    path.write_text(
        json.dumps(PublicTaskRow.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_metadata(
    *,
    dataset_version: str,
    config_hash: str,
    source_manifest_hash: str,
    label_policy: LabelPolicy,
    splits: dict[str, SplitFileSummary],
    rows: list[PublicTaskRow],
    license: str,
) -> DatasetMetadata:
    return DatasetMetadata(
        dataset_version=dataset_version,
        created_from_commit=_git_commit(),
        config_hash=config_hash,
        source_manifest_hash=source_manifest_hash,
        label_policy=label_policy,
        splits=splits,
        task_types=sorted(set(row.task_type for row in rows)),
        counts=DatasetCounts(
            total_rows=len(rows),
            by_task=dict(Counter(row.task_type for row in rows)),
            by_split=dict(Counter(row.split for row in rows)),
            by_relation=dict(Counter(row.labels.relation for row in rows)),
            by_source=dict(Counter(row.source for row in rows)),
            source_manifest_hash=source_manifest_hash,
        ),
        cadquery_version=None,
        ocp_version=None,
        license=license,
        known_limitations=[
            "Smoke generation uses synthetic primitive fixtures when CADEvolve is unavailable.",
            "This MVP path does not execute CADEvolve or CadQuery in-process.",
            "AABB baseline is diagnostic and not an official label source.",
        ],
    )


def write_metadata(metadata: DatasetMetadata, path: Path) -> None:
    path.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_metadata(path: Path) -> DatasetMetadata | None:
    if not path.exists():
        return None
    try:
        return DatasetMetadata.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{path}: invalid dataset metadata: {exc}") from exc


def source_manifest_hash(records: object) -> str:
    if isinstance(records, BaseModel):
        records = records.model_dump(mode="json")
    return sha256_json(records)


def _holdout_rule(split: str) -> str:
    if split == "train":
        return "training_split"
    if split == "validation":
        return "group_safe_validation"
    return split


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return f"python:{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
