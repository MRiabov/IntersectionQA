"""Metrics for exact-answer task evaluation and dataset summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Iterable

from intersectionqa.enums import TaskType
from intersectionqa.evaluation.edit_difficulty import edit_difficulty_tags
from intersectionqa.evaluation.parsing import parse_answer
from intersectionqa.schema import FailureRecord, ObjectValidationRecord, PublicTaskRow


@dataclass(frozen=True)
class Prediction:
    row_id: str
    output: str


@dataclass(frozen=True)
class TaskMetrics:
    task_type: TaskType
    total: int
    correct: int
    invalid_outputs: int
    accuracy: float
    invalid_output_rate: float
    per_label_accuracy: dict[str, float]
    numeric_mae_mm: float | None = None
    within_tolerance_accuracy: float | None = None
    pairwise_ranking_accuracy: float | None = None


def evaluate_predictions(rows: Iterable[PublicTaskRow], predictions: Iterable[Prediction]) -> list[TaskMetrics]:
    rows_by_id = {row.id: row for row in rows}
    predictions_by_id = {prediction.row_id: prediction.output for prediction in predictions}
    by_task: dict[TaskType, list[tuple[PublicTaskRow, str | None]]] = defaultdict(list)

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
        "by_candidate_strategy": _counts(
            row.metadata.get("candidate_strategy") or "unknown" for row in rows
        ),
        "by_source_subtree": _counts(
            subtree
            for row in rows
            for subtree in sorted(set(row.metadata.get("source_subtrees") or ["unknown"]))
            if subtree
        ),
        "by_split_relation": {
            split: _counts(row.labels.relation for row in rows if row.split == split)
            for split in sorted({row.split for row in rows})
        },
        "by_task_answer": {
            task_type: _counts(row.answer for row in rows if row.task_type == task_type)
            for task_type in sorted({row.task_type for row in rows})
        },
        "by_source": _counts(row.source for row in rows),
        "by_difficulty_tag": _counts(tag for row in rows for tag in row.difficulty_tags),
        "by_edit_difficulty_tag": _counts(tag for row in rows for tag in edit_difficulty_tags(row)),
        "repair_direction": _repair_direction_stats(rows),
        "repair_translation": _repair_translation_stats(rows),
        "axis_aligned_repair": _axis_aligned_edit_stats(rows, TaskType.AXIS_ALIGNED_REPAIR),
        "axis_aligned_repair_vector": _vector_edit_stats(rows, TaskType.AXIS_ALIGNED_REPAIR_VECTOR),
        "axis_aligned_repair_program": _vector_edit_stats(rows, TaskType.AXIS_ALIGNED_REPAIR_PROGRAM),
        "target_clearance_repair": _axis_aligned_edit_stats(rows, TaskType.TARGET_CLEARANCE_REPAIR),
        "target_clearance_move": _signed_distance_edit_stats(rows, TaskType.TARGET_CLEARANCE_MOVE),
        "target_contact_move": _signed_distance_edit_stats(rows, TaskType.TARGET_CONTACT_MOVE),
        "centroid_distance_move": _signed_distance_edit_stats(rows, TaskType.CENTROID_DISTANCE_MOVE),
        "edit_candidate_selection": _candidate_task_stats(rows, TaskType.EDIT_CANDIDATE_SELECTION),
        "edit_candidate_ranking": _candidate_task_stats(rows, TaskType.EDIT_CANDIDATE_RANKING),
    }


def manifest_stats(
    object_validations: Iterable[ObjectValidationRecord],
    failures: Iterable[FailureRecord],
) -> dict[str, object]:
    object_validations = list(object_validations)
    failures = list(failures)
    return {
        "object_validation_records": len(object_validations),
        "valid_objects": sum(1 for record in object_validations if record.valid),
        "invalid_objects": sum(1 for record in object_validations if not record.valid),
        "object_failures_by_reason": _counts(
            record.failure_reason for record in object_validations if record.failure_reason
        ),
        "failure_manifest_records": len(failures),
        "failures_by_stage": _counts(record.stage for record in failures),
        "failures_by_reason": _counts(record.failure_reason for record in failures),
    }


def _metrics_for_task(task_type: TaskType, items: list[tuple[PublicTaskRow, str | None]]) -> TaskMetrics:
    total = len(items)
    correct = sum(1 for row, parsed in items if parsed == row.answer)
    invalid = sum(1 for _, parsed in items if parsed is None)
    by_label_total: Counter[str] = Counter(row.answer for row, _ in items)
    by_label_correct: Counter[str] = Counter(row.answer for row, parsed in items if parsed == row.answer)
    numeric_mae, within_tolerance = _numeric_edit_metrics(task_type, items)
    pairwise_ranking_accuracy = _ranking_metric(task_type, items)
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
        numeric_mae_mm=numeric_mae,
        within_tolerance_accuracy=within_tolerance,
        pairwise_ranking_accuracy=pairwise_ranking_accuracy,
    )


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: Counter[str] = Counter(values)
    return dict(sorted(counts.items()))


_AXIS_REPAIR_RE = re.compile(r"^direction=(\+x|-x|\+y|-y|\+z|-z), distance_mm=([0-9]+(?:\.[0-9]+)?)$")
_SIGNED_DISTANCE_RE = re.compile(r"^distance_mm=(-?[0-9]+(?:\.[0-9]+)?)$")
_TRANSLATION_VECTOR_RE = re.compile(
    r"^dx=(-?[0-9]+(?:\.[0-9]+)?), dy=(-?[0-9]+(?:\.[0-9]+)?), dz=(-?[0-9]+(?:\.[0-9]+)?)$"
)
_EDIT_PROGRAM_RE = re.compile(
    r"^object_b = object_b\.translate\(\((-?[0-9]+(?:\.[0-9]+)?), (-?[0-9]+(?:\.[0-9]+)?), (-?[0-9]+(?:\.[0-9]+)?)\)\)$"
)


def _numeric_edit_metrics(
    task_type: TaskType,
    items: list[tuple[PublicTaskRow, str | None]],
) -> tuple[float | None, float | None]:
    if task_type not in {
        TaskType.AXIS_ALIGNED_REPAIR,
        TaskType.AXIS_ALIGNED_REPAIR_VECTOR,
        TaskType.AXIS_ALIGNED_REPAIR_PROGRAM,
        TaskType.TARGET_CLEARANCE_REPAIR,
        TaskType.TARGET_CLEARANCE_MOVE,
        TaskType.TARGET_CONTACT_MOVE,
        TaskType.CENTROID_DISTANCE_MOVE,
    }:
        return None, None
    errors: list[float] = []
    within = 0
    for row, parsed in items:
        if parsed is None:
            continue
        if task_type in {
            TaskType.TARGET_CLEARANCE_MOVE,
            TaskType.TARGET_CONTACT_MOVE,
            TaskType.CENTROID_DISTANCE_MOVE,
        }:
            truth_distance = _signed_distance_answer_part(row.answer)
            prediction_distance = _signed_distance_answer_part(parsed)
            if truth_distance is None or prediction_distance is None:
                continue
            error = abs(prediction_distance - truth_distance)
        elif task_type in {
            TaskType.AXIS_ALIGNED_REPAIR_VECTOR,
            TaskType.AXIS_ALIGNED_REPAIR_PROGRAM,
        }:
            truth_vector = _vector_answer_parts(row.answer, task_type)
            prediction_vector = _vector_answer_parts(parsed, task_type)
            if truth_vector is None or prediction_vector is None:
                continue
            error = _vector_l2(
                tuple(prediction_vector[index] - truth_vector[index] for index in range(3))
            )
        else:
            truth = _axis_answer_parts(row.answer)
            prediction = _axis_answer_parts(parsed)
            if truth is None or prediction is None or truth[0] != prediction[0]:
                continue
            error = abs(prediction[1] - truth[1])
        errors.append(error)
        tolerance = row.metadata.get("numeric_output_policy", {}).get(
            "acceptance_tolerance_mm",
            0.15,
        )
        if error <= float(tolerance):
            within += 1
    return (
        sum(errors) / len(errors) if errors else None,
        within / len(items) if items else 0.0,
    )


def _axis_answer_parts(value: str) -> tuple[str, float] | None:
    match = _AXIS_REPAIR_RE.match(value)
    if match is None:
        return None
    return match.group(1), float(match.group(2))


def _signed_distance_answer_part(value: str) -> float | None:
    match = _SIGNED_DISTANCE_RE.match(value)
    if match is None:
        return None
    return float(match.group(1))


def _vector_answer_parts(value: str, task_type: TaskType) -> tuple[float, float, float] | None:
    pattern = _EDIT_PROGRAM_RE if task_type == TaskType.AXIS_ALIGNED_REPAIR_PROGRAM else _TRANSLATION_VECTOR_RE
    match = pattern.match(value)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def _vector_l2(vector: tuple[float, float, float]) -> float:
    return sum(component * component for component in vector) ** 0.5


def _ranking_metric(
    task_type: TaskType,
    items: list[tuple[PublicTaskRow, str | None]],
) -> float | None:
    if task_type != TaskType.EDIT_CANDIDATE_RANKING:
        return None
    scores = [
        _pairwise_ranking_accuracy(row.answer, parsed)
        for row, parsed in items
        if parsed is not None
    ]
    return sum(scores) / len(items) if items else 0.0


def _pairwise_ranking_accuracy(expected: str, predicted: str | None) -> float:
    if predicted is None:
        return 0.0
    total = 0
    correct = 0
    expected_position = {label: index for index, label in enumerate(expected)}
    predicted_position = {label: index for index, label in enumerate(predicted)}
    for left in expected:
        for right in expected:
            if left >= right:
                continue
            total += 1
            expected_order = expected_position[left] < expected_position[right]
            predicted_order = predicted_position[left] < predicted_position[right]
            if expected_order == predicted_order:
                correct += 1
    return correct / total if total else 0.0


def _repair_direction_stats(rows: list[PublicTaskRow]) -> dict[str, object]:
    repair_rows = [row for row in rows if row.task_type == TaskType.REPAIR_DIRECTION]
    magnitudes = [
        float(row.metadata["selected_magnitude_mm"])
        for row in repair_rows
        if isinstance(row.metadata.get("selected_magnitude_mm"), int | float)
    ]
    return {
        "row_count": len(repair_rows),
        "by_policy": _counts(row.metadata.get("repair_policy", "unknown") for row in repair_rows),
        "by_selected_direction": _counts(row.answer for row in repair_rows),
        "selected_magnitude_mm": {
            "min": min(magnitudes) if magnitudes else None,
            "mean": sum(magnitudes) / len(magnitudes) if magnitudes else None,
            "max": max(magnitudes) if magnitudes else None,
        },
    }


def _repair_translation_stats(rows: list[PublicTaskRow]) -> dict[str, object]:
    repair_rows = [row for row in rows if row.task_type == TaskType.REPAIR_TRANSLATION]
    magnitudes = [
        float(row.metadata["selected_magnitude_mm"])
        for row in repair_rows
        if isinstance(row.metadata.get("selected_magnitude_mm"), int | float)
    ]
    return {
        "row_count": len(repair_rows),
        "by_policy": _counts(row.metadata.get("repair_policy", "unknown") for row in repair_rows),
        "by_selected_direction": _counts(
            str(row.metadata.get("selected_direction", "unknown")) for row in repair_rows
        ),
        "selected_magnitude_mm": {
            "min": min(magnitudes) if magnitudes else None,
            "mean": sum(magnitudes) / len(magnitudes) if magnitudes else None,
            "max": max(magnitudes) if magnitudes else None,
        },
    }


def _axis_aligned_edit_stats(rows: list[PublicTaskRow], task_type: TaskType) -> dict[str, object]:
    edit_rows = [row for row in rows if row.task_type == task_type]
    magnitudes = [
        float(row.metadata["selected_magnitude_mm"])
        for row in edit_rows
        if isinstance(row.metadata.get("selected_magnitude_mm"), int | float)
    ]
    exact_magnitudes = [
        float(row.metadata["selected_exact_magnitude_mm"])
        for row in edit_rows
        if isinstance(row.metadata.get("selected_exact_magnitude_mm"), int | float)
    ]
    ambiguous = [
        row
        for row in edit_rows
        if isinstance(row.metadata.get("edit_diagnostics"), dict)
        and row.metadata["edit_diagnostics"].get("ambiguous") is True
    ]
    return {
        "row_count": len(edit_rows),
        "by_policy": _counts(row.metadata.get("edit_policy", "unknown") for row in edit_rows),
        "by_selected_direction": _counts(
            str(row.metadata.get("selected_direction", "unknown")) for row in edit_rows
        ),
        "ambiguous_count": len(ambiguous),
        "selected_magnitude_mm": {
            "min": min(magnitudes) if magnitudes else None,
            "mean": sum(magnitudes) / len(magnitudes) if magnitudes else None,
            "max": max(magnitudes) if magnitudes else None,
        },
        "selected_exact_magnitude_mm": {
            "min": min(exact_magnitudes) if exact_magnitudes else None,
            "mean": sum(exact_magnitudes) / len(exact_magnitudes) if exact_magnitudes else None,
            "max": max(exact_magnitudes) if exact_magnitudes else None,
        },
    }


def _candidate_task_stats(rows: list[PublicTaskRow], task_type: TaskType) -> dict[str, object]:
    candidate_rows = [row for row in rows if row.task_type == task_type]
    return {
        "row_count": len(candidate_rows),
        "by_policy": _counts(row.metadata.get("edit_policy", "unknown") for row in candidate_rows),
        "by_answer": _counts(row.answer for row in candidate_rows),
    }


def _signed_distance_edit_stats(rows: list[PublicTaskRow], task_type: TaskType) -> dict[str, object]:
    edit_rows = [row for row in rows if row.task_type == task_type]
    signed_distances = [
        float(row.metadata["selected_signed_distance_mm"])
        for row in edit_rows
        if isinstance(row.metadata.get("selected_signed_distance_mm"), int | float)
    ]
    return {
        "row_count": len(edit_rows),
        "by_policy": _counts(row.metadata.get("edit_policy", "unknown") for row in edit_rows),
        "by_selected_direction": _counts(
            str(row.metadata.get("selected_direction", "unknown")) for row in edit_rows
        ),
        "by_move_kind": _counts(
            str(row.metadata.get("edit_diagnostics", {}).get("move_kind", "unknown"))
            for row in edit_rows
        ),
        "selected_signed_distance_mm": {
            "min": min(signed_distances) if signed_distances else None,
            "mean": sum(signed_distances) / len(signed_distances) if signed_distances else None,
            "max": max(signed_distances) if signed_distances else None,
        },
    }


def _vector_edit_stats(rows: list[PublicTaskRow], task_type: TaskType) -> dict[str, object]:
    edit_rows = [row for row in rows if row.task_type == task_type]
    magnitudes = [
        _vector_l2(tuple(float(item) for item in row.metadata["selected_translation_vector_mm"]))
        for row in edit_rows
        if isinstance(row.metadata.get("selected_translation_vector_mm"), list | tuple)
    ]
    return {
        "row_count": len(edit_rows),
        "by_policy": _counts(row.metadata.get("edit_policy", "unknown") for row in edit_rows),
        "by_output_format": _counts(row.metadata.get("output_format", "unknown") for row in edit_rows),
        "translation_magnitude_mm": {
            "min": min(magnitudes) if magnitudes else None,
            "mean": sum(magnitudes) / len(magnitudes) if magnitudes else None,
            "max": max(magnitudes) if magnitudes else None,
        },
    }
