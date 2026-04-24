"""AABB diagnostic baseline for binary interference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from intersectionqa.enums import TaskType
from intersectionqa.schema import PublicTaskRow


@dataclass(frozen=True)
class BaselineResult:
    total: int
    correct: int
    accuracy: float
    invalid_output_rate: float = 0.0
    per_relation_accuracy: dict[str, float] | None = None
    per_split_accuracy: dict[str, float] | None = None
    per_difficulty_accuracy: dict[str, float] | None = None


def predict_binary_from_aabb(row: PublicTaskRow) -> str:
    if row.diagnostics.aabb_overlap is None:
        return "no"
    return "yes" if row.diagnostics.aabb_overlap else "no"


def evaluate_aabb_binary(rows: Iterable[PublicTaskRow]) -> BaselineResult:
    binary_rows = [row for row in rows if row.task_type == TaskType.BINARY_INTERFERENCE]
    correct = sum(1 for row in binary_rows if _is_correct(row))
    total = len(binary_rows)
    return BaselineResult(
        total=total,
        correct=correct,
        accuracy=correct / total if total else 0.0,
        per_relation_accuracy=_group_accuracy(binary_rows, lambda row: row.labels.relation),
        per_split_accuracy=_group_accuracy(binary_rows, lambda row: row.split),
        per_difficulty_accuracy=_difficulty_accuracy(binary_rows),
    )


def _is_correct(row: PublicTaskRow) -> bool:
    return predict_binary_from_aabb(row) == row.answer


def _group_accuracy(rows: list[PublicTaskRow], key_fn: Callable[[PublicTaskRow], object]) -> dict[str, float]:
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for row in rows:
        key = key_fn(row)
        totals[key] = totals.get(key, 0) + 1
        correct[key] = correct.get(key, 0) + int(_is_correct(row))
    return {key: correct[key] / totals[key] for key in sorted(totals)}


def _difficulty_accuracy(rows: list[PublicTaskRow]) -> dict[str, float]:
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for row in rows:
        for tag in row.difficulty_tags:
            totals[tag] = totals.get(tag, 0) + 1
            correct[tag] = correct.get(tag, 0) + int(_is_correct(row))
    return {key: correct[key] / totals[key] for key in sorted(totals)}
