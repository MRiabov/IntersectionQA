"""Strict task-specific answer parsers."""

from __future__ import annotations

import math
import re

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
REPAIR_DIRECTION_ANSWERS = {"+x", "-x", "+y", "-y", "+z", "-z"}
REPAIR_TRANSLATION_RE = re.compile(r"^(\+x|-x|\+y|-y|\+z|-z) ([0-9]+)\.([0-9]{6})$")
AXIS_ALIGNED_REPAIR_RE = re.compile(
    r"^direction=(\+x|-x|\+y|-y|\+z|-z), distance_mm=([0-9]+)\.([0-9])$"
)
EDIT_CANDIDATE_SELECTION_ANSWERS = {"A", "B", "C", "D"}
EDIT_CANDIDATE_RANKING_ANSWERS = {
    "".join(items)
    for items in (
        (a, b, c, d)
        for a in "ABCD"
        for b in "ABCD"
        for c in "ABCD"
        for d in "ABCD"
        if len({a, b, c, d}) == 4
    )
}

ALLOWED_BY_TASK = {
    TaskType.BINARY_INTERFERENCE: BINARY_ANSWERS,
    TaskType.RELATION_CLASSIFICATION: RELATION_ANSWERS,
    TaskType.VOLUME_BUCKET: VOLUME_BUCKET_ANSWERS,
    TaskType.CLEARANCE_BUCKET: CLEARANCE_BUCKET_ANSWERS,
    TaskType.PAIRWISE_INTERFERENCE: PAIRWISE_ANSWERS,
    TaskType.RANKING_NORMALIZED_INTERSECTION: RANKING_ANSWERS,
    TaskType.REPAIR_DIRECTION: REPAIR_DIRECTION_ANSWERS,
    TaskType.EDIT_CANDIDATE_SELECTION: EDIT_CANDIDATE_SELECTION_ANSWERS,
    TaskType.EDIT_CANDIDATE_RANKING: EDIT_CANDIDATE_RANKING_ANSWERS,
    TaskType.TOLERANCE_FIT: BINARY_ANSWERS,
}


def parse_answer(task_type: TaskType, output: str) -> str | None:
    stripped = output.strip(" \t\r\n")
    if task_type == TaskType.REPAIR_TRANSLATION:
        match = REPAIR_TRANSLATION_RE.match(stripped)
        if match is None:
            return None
        magnitude = float(f"{match.group(2)}.{match.group(3)}")
        return stripped if math.isfinite(magnitude) else None
    if task_type in {TaskType.AXIS_ALIGNED_REPAIR, TaskType.TARGET_CLEARANCE_REPAIR}:
        match = AXIS_ALIGNED_REPAIR_RE.match(stripped)
        if match is None:
            return None
        magnitude = float(f"{match.group(2)}.{match.group(3)}")
        return stripped if math.isfinite(magnitude) else None
    allowed = ALLOWED_BY_TASK.get(task_type)
    if allowed is None:
        raise ValueError(f"unsupported task type for strict parsing: {task_type}")
    return stripped if stripped in allowed else None
