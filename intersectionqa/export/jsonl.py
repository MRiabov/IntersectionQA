"""JSONL export, validation, and metadata helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from intersectionqa.hashing import sha256_json
from intersectionqa.schema import DatasetMetadata, PublicTaskRow
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


def write_split_files(rows: list[PublicTaskRow], output_dir: Path) -> dict[str, dict[str, object]]:
    by_split: dict[str, list[PublicTaskRow]] = defaultdict(list)
    for row in rows:
        by_split[row.split].append(row)

    summary: dict[str, dict[str, object]] = {}
    for split in DEFAULT_SPLITS:
        split_rows = by_split.get(split, [])
        split_rows = sorted(split_rows, key=lambda row: row.id)
        path = output_dir / f"{split}.jsonl"
        write_jsonl(split_rows, path)
        summary[split] = {
            "path": path.name,
            "row_count": len(split_rows),
            "task_counts": dict(Counter(row.task_type for row in split_rows)),
            "holdout_rule": _holdout_rule(split),
        }
    return summary


def validate_rows(rows: Iterable[PublicTaskRow]) -> None:
    seen: set[str] = set()
    for row in rows:
        if row.id in seen:
            raise ValueError(f"duplicate public row id: {row.id}")
        seen.add(row.id)
        PublicTaskRow.model_validate(row.model_dump(mode="json"))


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
    label_policy: object,
    splits: dict[str, object],
    rows: list[PublicTaskRow],
    license: str,
) -> DatasetMetadata:
    return DatasetMetadata(
        dataset_version=dataset_version,
        created_from_commit=_git_commit(),
        config_hash=config_hash,
        source_manifest_hash=source_manifest_hash,
        label_policy=label_policy,  # type: ignore[arg-type]
        splits=splits,
        task_types=sorted(set(row.task_type for row in rows)),
        counts={
            "total_rows": len(rows),
            "by_task": dict(Counter(row.task_type for row in rows)),
            "by_split": dict(Counter(row.split for row in rows)),
            "by_relation": dict(Counter(row.labels.relation for row in rows)),
            "by_source": dict(Counter(row.source for row in rows)),
            "source_manifest_hash": source_manifest_hash,
        },
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


def source_manifest_hash(records: object) -> str:
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
