"""Failure-case summaries for dataset generation and model predictions."""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

from intersectionqa.enums import TaskType
from intersectionqa.evaluation.metrics import Prediction
from intersectionqa.evaluation.parsing import parse_answer
from intersectionqa.evaluation.repair import verify_repair_predictions
from intersectionqa.schema import FailureRecord, ObjectValidationRecord, PublicTaskRow

AXIS_EDIT_RE = re.compile(r"^direction=(\+x|-x|\+y|-y|\+z|-z), distance_mm=([0-9]+(?:\.[0-9]+)?)$")
SIGNED_DISTANCE_RE = re.compile(r"^distance_mm=(-?[0-9]+(?:\.[0-9]+)?)$")
TRANSLATION_VECTOR_RE = re.compile(
    r"^dx=(-?[0-9]+(?:\.[0-9]+)?), dy=(-?[0-9]+(?:\.[0-9]+)?), dz=(-?[0-9]+(?:\.[0-9]+)?)$"
)
EDIT_PROGRAM_RE = re.compile(
    r"^object_b = object_b\.translate\(\((-?[0-9]+(?:\.[0-9]+)?), (-?[0-9]+(?:\.[0-9]+)?), (-?[0-9]+(?:\.[0-9]+)?)\)\)$"
)
INTERSECTIONEDIT_TASKS = {
    TaskType.AXIS_ALIGNED_REPAIR,
    TaskType.AXIS_ALIGNED_REPAIR_VECTOR,
    TaskType.AXIS_ALIGNED_REPAIR_PROGRAM,
    TaskType.TARGET_CLEARANCE_REPAIR,
    TaskType.TARGET_CLEARANCE_MOVE,
    TaskType.TARGET_CONTACT_MOVE,
    TaskType.CENTROID_DISTANCE_MOVE,
    TaskType.EDIT_CANDIDATE_SELECTION,
    TaskType.EDIT_CANDIDATE_RANKING,
}


def failure_case_analysis(
    rows: Iterable[PublicTaskRow],
    object_validations: Iterable[ObjectValidationRecord],
    failures: Iterable[FailureRecord],
    *,
    predictions: Iterable[Prediction] | None = None,
    max_examples: int = 20,
) -> dict[str, object]:
    rows = list(rows)
    object_validations = list(object_validations)
    failures = list(failures)
    report: dict[str, object] = {
        "summary": {
            "rows": len(rows),
            "object_validation_records": len(object_validations),
            "invalid_object_records": sum(1 for record in object_validations if not record.valid),
            "failure_manifest_records": len(failures),
        },
        "object_validation_failures": _object_validation_failures(object_validations),
        "generation_failures": _generation_failures(failures),
    }
    if predictions is not None:
        prediction_list = list(predictions)
        report["prediction_failures"] = _prediction_failures(
            rows,
            prediction_list,
            max_examples=max_examples,
        )
        edit_failures = _intersectionedit_prediction_failures(
            rows,
            prediction_list,
            max_examples=max_examples,
        )
        if edit_failures["row_count"]:
            report["intersectionedit_prediction_failures"] = edit_failures
        repair_verifier = verify_repair_predictions(rows, prediction_list)
        if repair_verifier.report["row_count"]:
            report["repair_prediction_verifier"] = {
                **repair_verifier.report,
                "failed_examples": [
                    {
                        "id": result.row_id,
                        "output": result.output,
                        "parsed_output": result.parsed_output,
                        "valid_output": result.valid_output,
                        "repaired": result.repaired,
                        "relation_after_move": result.relation_after_move,
                        "failure_reason": result.failure_reason,
                    }
                    for result in repair_verifier.results
                    if not result.repaired
                ][: max(0, max_examples)],
            }
    return report


def _object_validation_failures(records: list[ObjectValidationRecord]) -> dict[str, object]:
    invalid = [record for record in records if not record.valid]
    return {
        "total": len(invalid),
        "by_reason": _counts(record.failure_reason for record in invalid if record.failure_reason),
    }


def _generation_failures(records: list[FailureRecord]) -> dict[str, object]:
    return {
        "total": len(records),
        "by_stage": _counts(record.stage for record in records),
        "by_reason": _counts(record.failure_reason for record in records),
        "by_source": _counts(record.source or "unknown" for record in records),
        "by_source_subset": _counts(
            subset for record in records for subset in _source_subsets(record.source_id)
        ),
        "top_error_summaries": dict(Counter(record.error_summary for record in records).most_common(20)),
    }


def _prediction_failures(
    rows: list[PublicTaskRow],
    predictions: list[Prediction],
    *,
    max_examples: int,
) -> dict[str, object]:
    rows_by_id = {row.id: row for row in rows}
    predictions_by_id = {prediction.row_id: prediction.output for prediction in predictions}
    missing = [row for row in rows if row.id not in predictions_by_id]
    invalid_rows: list[PublicTaskRow] = []
    incorrect_rows: list[tuple[PublicTaskRow, str, str | None]] = []

    for row in rows:
        output = predictions_by_id.get(row.id, "")
        parsed = parse_answer(row.task_type, output)
        if parsed is None:
            invalid_rows.append(row)
        if parsed != row.answer:
            incorrect_rows.append((row, output, parsed))

    unknown_prediction_ids = sorted(set(predictions_by_id) - set(rows_by_id))
    return {
        "prediction_records": len(predictions),
        "missing_prediction_count": len(missing),
        "unknown_prediction_count": len(unknown_prediction_ids),
        "invalid_output_count": len(invalid_rows),
        "incorrect_count": len(incorrect_rows),
        "incorrect_by_task": _counts(row.task_type for row, _, _ in incorrect_rows),
        "incorrect_by_relation": _counts(row.labels.relation for row, _, _ in incorrect_rows),
        "incorrect_by_split": _counts(row.split for row, _, _ in incorrect_rows),
        "incorrect_by_answer": _counts(row.answer for row, _, _ in incorrect_rows),
        "incorrect_by_difficulty_tag": _counts(
            tag for row, _, _ in incorrect_rows for tag in row.difficulty_tags
        ),
        "invalid_by_task": _counts(row.task_type for row in invalid_rows),
        "invalid_by_split": _counts(row.split for row in invalid_rows),
        "examples": [
            {
                "id": row.id,
                "task_type": row.task_type,
                "split": row.split,
                "relation": row.labels.relation,
                "answer": row.answer,
                "output": output,
                "parsed": parsed,
                "difficulty_tags": row.difficulty_tags,
            }
            for row, output, parsed in incorrect_rows[:max(0, max_examples)]
        ],
        "unknown_prediction_ids": unknown_prediction_ids[:max(0, max_examples)],
    }


def _intersectionedit_prediction_failures(
    rows: list[PublicTaskRow],
    predictions: list[Prediction],
    *,
    max_examples: int,
) -> dict[str, object]:
    edit_rows = [row for row in rows if row.task_type in INTERSECTIONEDIT_TASKS]
    predictions_by_id = {prediction.row_id: prediction.output for prediction in predictions}
    examples: list[dict[str, object]] = []
    categories: Counter[str] = Counter()
    task_categories: Counter[str] = Counter()

    for row in edit_rows:
        output = predictions_by_id.get(row.id, "")
        parsed = parse_answer(row.task_type, output)
        row_categories = _edit_failure_categories(row, parsed)
        for category in row_categories:
            categories[category] += 1
            task_categories[f"{row.task_type}:{category}"] += 1
        if parsed != row.answer and len(examples) < max(0, max_examples):
            examples.append(
                {
                    "id": row.id,
                    "task_type": row.task_type,
                    "answer": row.answer,
                    "output": output,
                    "parsed": parsed,
                    "categories": row_categories,
                    "edit_family": row.metadata.get("edit_family"),
                }
            )

    return {
        "row_count": len(edit_rows),
        "by_category": dict(sorted(categories.items())),
        "by_task_category": dict(sorted(task_categories.items())),
        "examples": examples,
    }


def _edit_failure_categories(row: PublicTaskRow, parsed: str | None) -> list[str]:
    if parsed is None:
        return ["invalid_output"]
    if parsed == row.answer:
        return ["correct"]
    if row.task_type in {TaskType.AXIS_ALIGNED_REPAIR, TaskType.TARGET_CLEARANCE_REPAIR}:
        return _axis_edit_failure_categories(row, parsed)
    if row.task_type in {TaskType.AXIS_ALIGNED_REPAIR_VECTOR, TaskType.AXIS_ALIGNED_REPAIR_PROGRAM}:
        return _vector_edit_failure_categories(row, parsed)
    if row.task_type in {
        TaskType.TARGET_CLEARANCE_MOVE,
        TaskType.TARGET_CONTACT_MOVE,
        TaskType.CENTROID_DISTANCE_MOVE,
    }:
        return _signed_distance_failure_categories(row, parsed)
    if row.task_type == TaskType.EDIT_CANDIDATE_SELECTION:
        return ["non_minimal_valid_edit" if _candidate_satisfies_target(row, parsed) else "wrong_candidate"]
    if row.task_type == TaskType.EDIT_CANDIDATE_RANKING:
        return ["candidate_ranking_error"]
    return ["incorrect"]


def _axis_edit_failure_categories(row: PublicTaskRow, parsed: str) -> list[str]:
    expected = _axis_answer_parts(row.answer)
    predicted = _axis_answer_parts(parsed)
    if expected is None or predicted is None:
        return ["invalid_output"]
    categories: list[str] = []
    expected_direction, expected_distance = expected
    predicted_direction, predicted_distance = predicted
    tolerance = _edit_tolerance(row)
    if predicted_direction != expected_direction:
        categories.append("wrong_direction")
    else:
        categories.append("correct_direction_wrong_distance")
    _add_distance_categories(
        categories,
        row,
        expected_distance=expected_distance,
        predicted_distance=predicted_distance,
        tolerance=tolerance,
    )
    return categories


def _signed_distance_failure_categories(row: PublicTaskRow, parsed: str) -> list[str]:
    expected = _signed_distance_answer_part(row.answer)
    predicted = _signed_distance_answer_part(parsed)
    if expected is None or predicted is None:
        return ["invalid_output"]
    categories: list[str] = []
    if (expected == 0.0 and predicted != 0.0) or (expected * predicted < 0.0):
        categories.append("wrong_direction")
    _add_distance_categories(
        categories,
        row,
        expected_distance=abs(expected),
        predicted_distance=abs(predicted),
        tolerance=_edit_tolerance(row),
    )
    if row.task_type == TaskType.CENTROID_DISTANCE_MOVE:
        categories.append("centroid_distance_error")
    return categories or ["signed_distance_error"]


def _vector_edit_failure_categories(row: PublicTaskRow, parsed: str) -> list[str]:
    expected = _vector_answer_parts(row.answer, row.task_type)
    predicted = _vector_answer_parts(parsed, row.task_type)
    if expected is None or predicted is None:
        return ["invalid_output"]
    categories: list[str] = ["translation_vector_error"]
    expected_magnitude = _vector_l2(expected)
    predicted_magnitude = _vector_l2(predicted)
    dot = sum(expected[index] * predicted[index] for index in range(3))
    if expected_magnitude > 1e-9 and predicted_magnitude > 1e-9 and dot <= 0.0:
        categories.append("wrong_direction")
    _add_distance_categories(
        categories,
        row,
        expected_distance=expected_magnitude,
        predicted_distance=predicted_magnitude,
        tolerance=_edit_tolerance(row),
    )
    return categories


def _add_distance_categories(
    categories: list[str],
    row: PublicTaskRow,
    *,
    expected_distance: float,
    predicted_distance: float,
    tolerance: float,
) -> None:
    if predicted_distance > expected_distance + tolerance:
        categories.append("excessive_movement")
    elif predicted_distance + tolerance < expected_distance:
        if row.task_type in {
            TaskType.TARGET_CLEARANCE_REPAIR,
            TaskType.TARGET_CLEARANCE_MOVE,
            TaskType.TARGET_CONTACT_MOVE,
        }:
            categories.append("unresolved_target_clearance")
        else:
            categories.append("insufficient_movement")
    else:
        categories.append("wrong_distance")


def _candidate_satisfies_target(row: PublicTaskRow, parsed: str) -> bool:
    candidates = row.metadata.get("candidate_edits")
    if not isinstance(candidates, dict):
        return False
    candidate = candidates.get(parsed)
    return isinstance(candidate, dict) and candidate.get("satisfies_target") is True


def _axis_answer_parts(value: str) -> tuple[str, float] | None:
    match = AXIS_EDIT_RE.match(value)
    if match is None:
        return None
    return match.group(1), float(match.group(2))


def _signed_distance_answer_part(value: str) -> float | None:
    match = SIGNED_DISTANCE_RE.match(value)
    if match is None:
        return None
    return float(match.group(1))


def _vector_answer_parts(value: str, task_type: TaskType) -> tuple[float, float, float] | None:
    match = (
        EDIT_PROGRAM_RE.match(value)
        if task_type == TaskType.AXIS_ALIGNED_REPAIR_PROGRAM
        else TRANSLATION_VECTOR_RE.match(value)
    )
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def _vector_l2(vector: tuple[float, float, float]) -> float:
    return sum(component * component for component in vector) ** 0.5


def _edit_tolerance(row: PublicTaskRow) -> float:
    policy = row.metadata.get("numeric_output_policy")
    if not isinstance(policy, dict):
        return 0.15
    value = policy.get("acceptance_tolerance_mm", 0.15)
    return float(value) if isinstance(value, int | float) else 0.15


def _source_subsets(source_id: str | None) -> list[str]:
    if not source_id:
        return ["unknown"]
    subsets: list[str] = []
    for item in source_id.split("|"):
        parts = item.split("/")
        if len(parts) >= 2:
            subsets.append("/".join(parts[:2]))
        elif parts and parts[0]:
            subsets.append(parts[0])
    return subsets or ["unknown"]


def _counts(values: Iterable[object]) -> dict[str, int]:
    counts: Counter[str] = Counter(str(value) for value in values)
    return dict(sorted(counts.items()))
