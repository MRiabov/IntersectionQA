"""Failure-case summaries for dataset generation and model predictions."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from intersectionqa.evaluation.metrics import Prediction
from intersectionqa.evaluation.parsing import parse_answer
from intersectionqa.evaluation.repair import verify_repair_predictions
from intersectionqa.schema import FailureRecord, ObjectValidationRecord, PublicTaskRow


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
