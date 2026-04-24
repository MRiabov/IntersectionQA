"""Parquet packaging for public IntersectionQA rows."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from intersectionqa.schema import PublicTaskRow
from intersectionqa.splits.grouped import DEFAULT_SPLITS


def write_parquet_files(rows: list[PublicTaskRow], output_dir: Path) -> dict[str, int]:
    """Write one Parquet file per split and return row counts by file name."""

    output_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[PublicTaskRow]] = defaultdict(list)
    for row in rows:
        by_split[row.split].append(row)

    counts: dict[str, int] = {}
    for split in DEFAULT_SPLITS:
        split_rows = sorted(by_split.get(split, []), key=lambda row: row.id)
        path = output_dir / f"{split}.parquet"
        table = public_rows_to_table(split_rows)
        pq.write_table(table, path, compression="zstd")
        counts[path.name] = len(split_rows)
    return counts


def public_rows_to_table(rows: list[PublicTaskRow]) -> pa.Table:
    records = [_flatten_public_row(row) for row in rows]
    return pa.Table.from_pylist(records, schema=_public_row_schema())


def read_parquet_rows(path: Path) -> list[dict[str, object]]:
    return pq.read_table(path).to_pylist()


def _flatten_public_row(row: PublicTaskRow) -> dict[str, object]:
    data = row.model_dump(mode="json")
    return {
        "id": data["id"],
        "dataset_version": data["dataset_version"],
        "split": data["split"],
        "task_type": data["task_type"],
        "prompt": data["prompt"],
        "answer": data["answer"],
        "script": data["script"],
        "geometry_ids_json": json.dumps(data["geometry_ids"], sort_keys=True),
        "source": data["source"],
        "generator_id": data["generator_id"],
        "base_object_pair_id": data["base_object_pair_id"],
        "assembly_group_id": data["assembly_group_id"],
        "counterfactual_group_id": data["counterfactual_group_id"],
        "variant_id": data["variant_id"],
        "changed_parameter": data["changed_parameter"],
        "changed_value_json": json.dumps(data["changed_value"], sort_keys=True),
        "labels_json": json.dumps(data["labels"], sort_keys=True),
        "diagnostics_json": json.dumps(data["diagnostics"], sort_keys=True),
        "difficulty_tags_json": json.dumps(data["difficulty_tags"], sort_keys=True),
        "label_policy_json": json.dumps(data["label_policy"], sort_keys=True),
        "hashes_json": json.dumps(data["hashes"], sort_keys=True),
        "metadata_json": json.dumps(data["metadata"], sort_keys=True),
    }


def _public_row_schema() -> pa.Schema:
    return pa.schema(
        [
            ("id", pa.string()),
            ("dataset_version", pa.string()),
            ("split", pa.string()),
            ("task_type", pa.string()),
            ("prompt", pa.string()),
            ("answer", pa.string()),
            ("script", pa.string()),
            ("geometry_ids_json", pa.string()),
            ("source", pa.string()),
            ("generator_id", pa.string()),
            ("base_object_pair_id", pa.string()),
            ("assembly_group_id", pa.string()),
            ("counterfactual_group_id", pa.string()),
            ("variant_id", pa.string()),
            ("changed_parameter", pa.string()),
            ("changed_value_json", pa.string()),
            ("labels_json", pa.string()),
            ("diagnostics_json", pa.string()),
            ("difficulty_tags_json", pa.string()),
            ("label_policy_json", pa.string()),
            ("hashes_json", pa.string()),
            ("metadata_json", pa.string()),
        ]
    )
