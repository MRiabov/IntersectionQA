"""Strict task-specific answer parsers."""

from __future__ import annotations

from intersectionqa.geometry.labels import VOLUME_BUCKETS

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

ALLOWED_BY_TASK = {
    "binary_interference": BINARY_ANSWERS,
    "relation_classification": RELATION_ANSWERS,
    "volume_bucket": VOLUME_BUCKET_ANSWERS,
}


def parse_answer(task_type: str, output: str) -> str | None:
    allowed = ALLOWED_BY_TASK.get(task_type)
    if allowed is None:
        raise ValueError(f"unsupported task type for strict parsing: {task_type}")
    stripped = output.strip(" \t\r\n")
    return stripped if stripped in allowed else None
