"""AABB diagnostic baseline for binary interference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from intersectionqa.schema import PublicTaskRow


@dataclass(frozen=True)
class BaselineResult:
    total: int
    correct: int
    accuracy: float
    invalid_output_rate: float = 0.0


def predict_binary_from_aabb(row: PublicTaskRow) -> str:
    if row.diagnostics.aabb_overlap is None:
        return "no"
    return "yes" if row.diagnostics.aabb_overlap else "no"


def evaluate_aabb_binary(rows: Iterable[PublicTaskRow]) -> BaselineResult:
    binary_rows = [row for row in rows if row.task_type == "binary_interference"]
    correct = sum(1 for row in binary_rows if predict_binary_from_aabb(row) == row.answer)
    total = len(binary_rows)
    return BaselineResult(total=total, correct=correct, accuracy=correct / total if total else 0.0)
