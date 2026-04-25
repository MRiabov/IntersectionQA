"""Materialize MVP public task rows from stored geometry records."""

from __future__ import annotations

from collections import defaultdict

from collections.abc import Callable
from intersectionqa.enums import LabelStatus, Relation, TaskType
from intersectionqa.prompts.binary import materialize_binary_row
from intersectionqa.prompts.buckets import materialize_clearance_bucket_row, materialize_volume_bucket_row
from intersectionqa.prompts.counterfactual import materialize_pairwise_rows
from intersectionqa.prompts.fit import materialize_tolerance_fit_row
from intersectionqa.prompts.ranking import materialize_ranking_rows
from intersectionqa.prompts.repair import materialize_repair_direction_row
from intersectionqa.prompts.relation import materialize_relation_row
from intersectionqa.schema import GeometryRecord, PublicTaskRow

SINGLE_RECORD_MATERIALIZERS = {
    TaskType.BINARY_INTERFERENCE: materialize_binary_row,
    TaskType.RELATION_CLASSIFICATION: materialize_relation_row,
    TaskType.VOLUME_BUCKET: materialize_volume_bucket_row,
    TaskType.CLEARANCE_BUCKET: materialize_clearance_bucket_row,
    TaskType.REPAIR_DIRECTION: materialize_repair_direction_row,
    TaskType.TOLERANCE_FIT: materialize_tolerance_fit_row,
}

GROUP_MATERIALIZERS = {
    TaskType.PAIRWISE_INTERFERENCE: materialize_pairwise_rows,
    TaskType.RANKING_NORMALIZED_INTERSECTION: materialize_ranking_rows,
}

GroupMaterializer = Callable[[list[GeometryRecord], int, str], list[PublicTaskRow]]


def materialize_rows(
    records: list[GeometryRecord],
    split_by_geometry_id: dict[str, str],
    task_types: list[TaskType],
) -> list[PublicTaskRow]:
    counters: dict[TaskType, int] = defaultdict(int)
    rows: list[PublicTaskRow] = []
    valid_records = [
        record
        for record in sorted(records, key=lambda item: item.geometry_id)
        if record.diagnostics.label_status == LabelStatus.OK and record.labels.relation != Relation.INVALID
    ]
    for record in valid_records:
        split = split_by_geometry_id[record.geometry_id]
        for task_type in task_types:
            materializer = SINGLE_RECORD_MATERIALIZERS.get(task_type)
            if materializer is None:
                continue
            row = materializer(record, counters[task_type] + 1, split)
            if row is None:
                continue
            counters[task_type] += 1
            rows.append(row)

    grouped = _counterfactual_groups(valid_records)
    for group_records in grouped:
        split = split_by_geometry_id[group_records[0].geometry_id]
        for task_type in task_types:
            materializer = GROUP_MATERIALIZERS.get(task_type)
            if materializer is None:
                continue
            group_rows = materializer(group_records, counters[task_type] + 1, split)
            counters[task_type] += len(group_rows)
            rows.extend(group_rows)
    return rows


def _counterfactual_groups(records: list[GeometryRecord]) -> list[list[GeometryRecord]]:
    by_group: dict[str, list[GeometryRecord]] = defaultdict(list)
    for record in records:
        if record.diagnostics.label_status != LabelStatus.OK or record.labels.relation == Relation.INVALID:
            continue
        if record.counterfactual_group_id:
            by_group[record.counterfactual_group_id].append(record)
    return [
        sorted(group_records, key=lambda item: item.geometry_id)
        for _, group_records in sorted(by_group.items())
        if len(group_records) >= 2
    ]
