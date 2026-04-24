"""Materialize MVP public task rows from stored geometry records."""

from __future__ import annotations

from collections import defaultdict

from intersectionqa.enums import LabelStatus, Relation, TaskType
from intersectionqa.prompts.binary import materialize_binary_row
from intersectionqa.prompts.buckets import materialize_volume_bucket_row
from intersectionqa.prompts.relation import materialize_relation_row
from intersectionqa.schema import GeometryRecord, PublicTaskRow

MATERIALIZERS = {
    TaskType.BINARY_INTERFERENCE: materialize_binary_row,
    TaskType.RELATION_CLASSIFICATION: materialize_relation_row,
    TaskType.VOLUME_BUCKET: materialize_volume_bucket_row,
}


def materialize_rows(
    records: list[GeometryRecord],
    split_by_geometry_id: dict[str, str],
    task_types: list[TaskType],
) -> list[PublicTaskRow]:
    counters: dict[TaskType, int] = defaultdict(int)
    rows: list[PublicTaskRow] = []
    for record in sorted(records, key=lambda item: item.geometry_id):
        if record.diagnostics.label_status != LabelStatus.OK or record.labels.relation == Relation.INVALID:
            continue
        split = split_by_geometry_id[record.geometry_id]
        for task_type in task_types:
            counters[task_type] += 1
            rows.append(MATERIALIZERS[task_type](record, counters[task_type], split))
    return rows
