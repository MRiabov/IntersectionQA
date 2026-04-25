"""Comparison-table helpers for baselines and saved model predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from intersectionqa.enums import TaskType
from intersectionqa.evaluation.aabb import BaselineResult
from intersectionqa.evaluation.metrics import TaskMetrics
from intersectionqa.evaluation.repair import RepairVerificationRunResult


@dataclass(frozen=True)
class ComparisonRow:
    system: str
    task_type: str
    total: int
    correct: int
    accuracy: float
    invalid_output_rate: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def comparison_rows_from_aabb(
    result: BaselineResult,
    *,
    system: str = "aabb_overlap",
) -> list[ComparisonRow]:
    return [
        ComparisonRow(
            system=system,
            task_type=TaskType.BINARY_INTERFERENCE.value,
            total=result.total,
            correct=result.correct,
            accuracy=result.accuracy,
            invalid_output_rate=result.invalid_output_rate,
        )
    ]


def comparison_rows_from_metrics(
    metrics: Iterable[TaskMetrics],
    *,
    system: str,
) -> list[ComparisonRow]:
    return [
        ComparisonRow(
            system=system,
            task_type=metric.task_type.value,
            total=metric.total,
            correct=metric.correct,
            accuracy=metric.accuracy,
            invalid_output_rate=metric.invalid_output_rate,
        )
        for metric in metrics
    ]


def comparison_rows_from_repair_verifier(
    result: RepairVerificationRunResult,
    *,
    system: str = "repair_verifier",
) -> list[ComparisonRow]:
    row_count = int(result.report["row_count"])
    if row_count == 0:
        return []
    return [
        ComparisonRow(
            system=system,
            task_type=TaskType.REPAIR_DIRECTION.value,
            total=row_count,
            correct=int(result.report["repaired_count"]),
            accuracy=float(result.report["repair_success_rate"]),
            invalid_output_rate=1.0 - float(result.report["valid_output_rate"]),
        )
    ]


def sort_comparison_rows(rows: Iterable[ComparisonRow]) -> list[ComparisonRow]:
    return sorted(rows, key=lambda row: (row.task_type, row.system))


def comparison_rows_to_markdown(rows: Iterable[ComparisonRow]) -> str:
    ordered = sort_comparison_rows(rows)
    lines = [
        "| system | task_type | total | correct | accuracy | invalid_output_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.system,
                    row.task_type,
                    str(row.total),
                    str(row.correct),
                    f"{row.accuracy:.4f}",
                    f"{row.invalid_output_rate:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"
