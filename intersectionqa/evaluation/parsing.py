"""Strict task-specific answer parsers."""

from __future__ import annotations

from intersectionqa.geometry.labels import VOLUME_BUCKETS
from intersectionqa.enums import TaskType
from intersectionqa.prompts.buckets import CLEARANCE_BUCKETS

BINARY_ANSWERS = {"yes", "no"}
RELATION_ANSWERS = {
    "disjoint",
    "touching",
    "near_miss",
    "intersecting",
    "contained",
    "invalid",
}
VOLUME_BUCKET_ANSWERS = set(VOLUME_BUCKETS)
CLEARANCE_BUCKET_ANSWERS = set(CLEARANCE_BUCKETS)
PAIRWISE_ANSWERS = {"A", "B", "both", "neither"}
RANKING_ANSWERS = {
    "".join(items)
    for items in (
        (a, b, c)
        for a in "ABCDE"
        for b in "ABCDE"
        for c in "ABCDE"
        if len({a, b, c}) == 3
    )
} | {
    "".join(items)
    for items in (
        (a, b, c, d)
        for a in "ABCDE"
        for b in "ABCDE"
        for c in "ABCDE"
        for d in "ABCDE"
        if len({a, b, c, d}) == 4
    )
} | {
    "".join(items)
    for items in (
        (a, b, c, d, e)
        for a in "ABCDE"
        for b in "ABCDE"
        for c in "ABCDE"
        for d in "ABCDE"
        for e in "ABCDE"
        if len({a, b, c, d, e}) == 5
    )
}

ALLOWED_BY_TASK = {
    TaskType.BINARY_INTERFERENCE: BINARY_ANSWERS,
    TaskType.RELATION_CLASSIFICATION: RELATION_ANSWERS,
    TaskType.VOLUME_BUCKET: VOLUME_BUCKET_ANSWERS,
    TaskType.CLEARANCE_BUCKET: CLEARANCE_BUCKET_ANSWERS,
    TaskType.PAIRWISE_INTERFERENCE: PAIRWISE_ANSWERS,
    TaskType.RANKING_NORMALIZED_INTERSECTION: RANKING_ANSWERS,
    TaskType.TOLERANCE_FIT: BINARY_ANSWERS,
}


def parse_answer(task_type: TaskType, output: str) -> str | None:
    allowed = ALLOWED_BY_TASK.get(task_type)
    if allowed is None:
        raise ValueError(f"unsupported task type for strict parsing: {task_type}")
    stripped = output.strip(" \t\r\n")
    return stripped if stripped in allowed else None
