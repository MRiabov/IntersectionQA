"""Metrics for exact-answer task evaluation and dataset summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from intersectionqa.evaluation.parsing import parse_answer
from intersectionqa.schema import PublicTaskRow


@dataclass(frozen=True)
class Prediction:
    row_id: str
    output: str


@dataclass(frozen=True)
class TaskMetrics:
    task_type: str
    total: int
    correct: int
    invalid_outputs: int
    accuracy: float
    invalid_output_rate: float
    per_label_accuracy: dict[str, float]


def evaluate_predictions(rows: Iterable[PublicTaskRow], predictions: Iterable[Prediction]) -> list[TaskMetrics]:
    rows_by_id = {row.id: row for row in rows}
    predictions_by_id = {prediction.row_id: prediction.output for prediction in predictions}
    by_task: dict[str, list[tuple[PublicTaskRow, str | None]]] = defaultdict(list)

    for row_id, row in rows_by_id.items():
        output = predictions_by_id.get(row_id, "")
        parsed = parse_answer(row.task_type, output)
        by_task[row.task_type].append((row, parsed))

    return [_metrics_for_task(task_type, items) for task_type, items in sorted(by_task.items())]


def dataset_stats(rows: Iterable[PublicTaskRow]) -> dict[str, object]:
    rows = list(rows)
    return {
        "total_rows": len(rows),
        "by_task": _counts(row.task_type for row in rows),
        "by_split": _counts(row.split for row in rows),
        "by_relation": _counts(row.labels.relation for row in rows),
        "by_answer": _counts(row.answer for row in rows),
        "by_source": _counts(row.source for row in rows),
        "by_difficulty_tag": _counts(tag for row in rows for tag in row.difficulty_tags),
    }


def _metrics_for_task(task_type: str, items: list[tuple[PublicTaskRow, str | None]]) -> TaskMetrics:
    total = len(items)
    correct = sum(1 for row, parsed in items if parsed == row.answer)
    invalid = sum(1 for _, parsed in items if parsed is None)
    by_label_total: Counter[str] = Counter(row.answer for row, _ in items)
    by_label_correct: Counter[str] = Counter(row.answer for row, parsed in items if parsed == row.answer)
    return TaskMetrics(
        task_type=task_type,
        total=total,
        correct=correct,
        invalid_outputs=invalid,
        accuracy=correct / total if total else 0.0,
        invalid_output_rate=invalid / total if total else 0.0,
        per_label_accuracy={
            label: by_label_correct[label] / count for label, count in sorted(by_label_total.items())
        },
    )


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: Counter[str] = Counter(values)
    return dict(sorted(counts.items()))
