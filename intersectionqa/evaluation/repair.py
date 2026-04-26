"""Verifier helpers for IntersectionEdit repair-direction rows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from intersectionqa.enums import Relation, TaskType
from intersectionqa.evaluation.metrics import Prediction
from intersectionqa.evaluation.parsing import parse_answer
from intersectionqa.geometry.cadquery_exec import apply_transform, measure_shape_pair, object_to_shape
from intersectionqa.geometry.labels import derive_labels
from intersectionqa.geometry.transforms import IDENTITY_TRANSFORM
from intersectionqa.schema import BoundingBox, PublicTaskRow, Transform

REPAIR_TIE_BREAK_ORDER = ["+x", "-x", "+y", "-y", "+z", "-z"]
REPAIR_POLICY_NAME = "conservative_aabb_separating_translation_v01"
REPAIR_TASK_TYPES = {TaskType.REPAIR_DIRECTION, TaskType.REPAIR_TRANSLATION}
TRUSTED_SCRIPT_EXECUTION_MODEL = (
    "trusted_dataset_rows_only: repair verification executes row.script with Python exec "
    "because CadQuery reconstruction requires code execution. Do not run it on untrusted rows."
)
_PLACED_SHAPES_CACHE: dict[str, tuple[Any, Any]] = {}
_RELATION_AFTER_MOVE_CACHE: dict[tuple[str, str], Relation] = {}


@dataclass(frozen=True)
class RepairVerificationResult:
    row_id: str
    output: str
    parsed_output: str | None
    valid_output: bool
    repaired: bool
    relation_after_move: str | None
    failure_reason: str | None = None


@dataclass(frozen=True)
class RepairVerificationRunResult:
    report: dict[str, Any]
    results: list[RepairVerificationResult]


def validate_repair_row_metadata(row: PublicTaskRow) -> None:
    """Recompute conservative AABB repair metadata from stored row bboxes."""
    if row.task_type not in REPAIR_TASK_TYPES:
        return
    if row.labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}:
        raise ValueError(f"{row.task_type} rows require positive-overlap relation")
    if row.metadata.get("repair_policy") != REPAIR_POLICY_NAME:
        raise ValueError("unsupported repair policy")

    expected = _expected_candidates_from_bboxes(row)
    actual = {str(candidate["direction"]): candidate for candidate in _candidate_moves(row)}
    if set(actual) != set(expected):
        raise ValueError("repair metadata candidate directions do not match recomputed policy")
    for direction, expected_candidate in expected.items():
        actual_candidate = actual[direction]
        if not math.isclose(
            float(actual_candidate["magnitude_mm"]),
            float(expected_candidate["magnitude_mm"]),
            abs_tol=1e-9,
        ):
            raise ValueError("repair metadata candidate magnitude does not match stored bboxes")
        if _candidate_vector(actual_candidate) != tuple(expected_candidate["translation_vector_mm"]):
            raise ValueError("repair metadata candidate vector does not match stored bboxes")

    selected_direction = _selected_direction_from_metadata(row)
    expected_selected = min(
        expected.values(),
        key=lambda candidate: (
            float(candidate["magnitude_mm"]),
            REPAIR_TIE_BREAK_ORDER.index(str(candidate["direction"])),
        ),
    )
    if selected_direction != expected_selected["direction"]:
        raise ValueError("repair selected direction does not match recomputed AABB policy")
    if row.task_type == TaskType.REPAIR_TRANSLATION:
        expected_answer = f"{selected_direction} {float(expected_selected['magnitude_mm']):.6f}"
        if row.answer != expected_answer:
            raise ValueError("repair_translation answer does not match recomputed AABB policy")


def verify_repair_rows_exact(rows: Iterable[PublicTaskRow]) -> RepairVerificationRunResult:
    """Verify stored edit answers with exact CadQuery geometry execution."""
    repair_rows = [row for row in rows if row.task_type in REPAIR_TASK_TYPES]
    results = [_verify_stored_repair_answer(row) for row in repair_rows]
    invalid = [result for result in results if not result.valid_output]
    repaired = [result for result in results if result.repaired]
    failures = [result for result in results if result.failure_reason]
    report = {
        "system": "intersectionedit_exact_repair_row_validator",
        "execution_model": TRUSTED_SCRIPT_EXECUTION_MODEL,
        "row_count": len(repair_rows),
        "valid_output_count": len(results) - len(invalid),
        "invalid_output_count": len(invalid),
        "repaired_count": len(repaired),
        "valid_output_rate": (len(results) - len(invalid)) / len(results) if results else 0.0,
        "repair_success_rate": len(repaired) / len(results) if results else 0.0,
        "failure_reasons": _counts(result.failure_reason for result in failures),
        "by_output": _output_breakdown(repair_rows, results),
    }
    return RepairVerificationRunResult(report=report, results=results)


def verified_repair_direction(row: PublicTaskRow) -> str:
    """Return the first candidate direction that exactly removes positive overlap."""
    _require_repair_row(row)
    for candidate in sorted(_candidate_moves(row), key=_candidate_sort_key):
        if _relation_after_direction(row, str(candidate["direction"])) not in {
            Relation.INTERSECTING,
            Relation.CONTAINED,
        }:
            return str(candidate["direction"])
    raise ValueError("no repair_direction candidate removed positive overlap")


def verify_repair_direction_output(row: PublicTaskRow, output: str) -> RepairVerificationResult:
    _require_repair_row(row)
    parsed = parse_answer(TaskType.REPAIR_DIRECTION, output)
    if parsed is None:
        return RepairVerificationResult(
            row_id=row.id,
            output=output,
            parsed_output=None,
            valid_output=False,
            repaired=False,
            relation_after_move=None,
            failure_reason="invalid_output",
        )
    try:
        relation_after_move = _relation_after_direction(row, parsed)
    except Exception as exc:
        return RepairVerificationResult(
            row_id=row.id,
            output=output,
            parsed_output=parsed,
            valid_output=True,
            repaired=False,
            relation_after_move=None,
            failure_reason=type(exc).__name__,
        )
    repaired = relation_after_move not in {Relation.INTERSECTING, Relation.CONTAINED}
    return RepairVerificationResult(
        row_id=row.id,
        output=output,
        parsed_output=parsed,
        valid_output=True,
        repaired=repaired,
        relation_after_move=str(relation_after_move),
    )


def _verify_stored_repair_answer(row: PublicTaskRow) -> RepairVerificationResult:
    _require_repair_row(row)
    parsed = parse_answer(row.task_type, row.answer)
    if parsed is None:
        return RepairVerificationResult(
            row_id=row.id,
            output=row.answer,
            parsed_output=None,
            valid_output=False,
            repaired=False,
            relation_after_move=None,
            failure_reason="invalid_stored_answer",
        )
    direction = parsed.split(" ", 1)[0] if row.task_type == TaskType.REPAIR_TRANSLATION else parsed
    try:
        validate_repair_row_metadata(row)
        relation_after_move = _relation_after_direction(row, direction)
    except Exception as exc:
        return RepairVerificationResult(
            row_id=row.id,
            output=row.answer,
            parsed_output=parsed,
            valid_output=True,
            repaired=False,
            relation_after_move=None,
            failure_reason=type(exc).__name__,
        )
    repaired = relation_after_move not in {Relation.INTERSECTING, Relation.CONTAINED}
    return RepairVerificationResult(
        row_id=row.id,
        output=row.answer,
        parsed_output=parsed,
        valid_output=True,
        repaired=repaired,
        relation_after_move=str(relation_after_move),
    )


def verify_repair_predictions(
    rows: list[PublicTaskRow],
    predictions: list[Prediction],
) -> RepairVerificationRunResult:
    repair_rows = [row for row in rows if row.task_type == TaskType.REPAIR_DIRECTION]
    predictions_by_id = {prediction.row_id: prediction.output for prediction in predictions}
    results = [
        verify_repair_direction_output(row, predictions_by_id.get(row.id, ""))
        for row in repair_rows
    ]
    invalid = [result for result in results if not result.valid_output]
    repaired = [result for result in results if result.repaired]
    failures = [result for result in results if result.failure_reason]
    exact_answer_correct = sum(
        1
        for row in repair_rows
        if parse_answer(
            TaskType.REPAIR_DIRECTION,
            predictions_by_id.get(row.id, ""),
        )
        == row.answer
    )
    report = {
        "system": "intersectionedit_repair_direction_exact_verifier",
        "row_count": len(repair_rows),
        "valid_output_count": len(results) - len(invalid),
        "invalid_output_count": len(invalid),
        "repaired_count": len(repaired),
        "exact_answer_correct_count": exact_answer_correct,
        "valid_output_rate": (len(results) - len(invalid)) / len(results) if results else 0.0,
        "repair_success_rate": len(repaired) / len(results) if results else 0.0,
        "exact_answer_accuracy": exact_answer_correct / len(results) if results else 0.0,
        "failure_reasons": _counts(result.failure_reason for result in failures),
        "by_output": _output_breakdown(repair_rows, results),
    }
    return RepairVerificationRunResult(report=report, results=results)


def _relation_after_direction(row: PublicTaskRow, direction: str) -> Relation:
    cache_key = _geometry_cache_key(row)
    relation_cache_key = (cache_key, direction)
    cached = _RELATION_AFTER_MOVE_CACHE.get(relation_cache_key)
    if cached is not None:
        return cached
    candidate = _candidate_by_direction(row, direction)
    bbox_a = _metadata_bbox(row, "bbox_a")
    bbox_b = _metadata_bbox(row, "bbox_b")
    if bbox_a is not None and bbox_b is not None:
        moved_bbox = _translated_bbox(bbox_b, _candidate_vector(candidate))
        if not _boxes_overlap(bbox_a, moved_bbox):
            _RELATION_AFTER_MOVE_CACHE[relation_cache_key] = Relation.DISJOINT
            return Relation.DISJOINT
    shape_a, shape_b = _placed_shapes(row)
    moved_b = apply_transform(
        shape_b,
        Transform(
            translation=_candidate_vector(candidate),
            rotation_xyz_deg=(0.0, 0.0, 0.0),
        ),
    )
    raw = measure_shape_pair(
        shape_a,
        moved_b,
        IDENTITY_TRANSFORM,
        IDENTITY_TRANSFORM,
        row.label_policy,
    )
    labels, _ = derive_labels(raw, row.label_policy)
    _RELATION_AFTER_MOVE_CACHE[relation_cache_key] = labels.relation
    return labels.relation


def _placed_shapes(row: PublicTaskRow) -> tuple[Any, Any]:
    """Execute trusted dataset row code to reconstruct placed CadQuery shapes."""
    cache_key = _geometry_cache_key(row)
    cached = _PLACED_SHAPES_CACHE.get(cache_key)
    if cached is not None:
        return cached
    import cadquery as cq

    namespace: dict[str, Any] = {"cq": cq, "cadquery": cq, "__builtins__": __builtins__}
    exec(compile(row.script, "<intersectionedit-repair-verifier>", "exec"), namespace)
    assembly = namespace.get("assembly")
    if not callable(assembly):
        raise ValueError("assembly script does not define assembly()")
    placed_a, placed_b = assembly()
    shapes = object_to_shape(placed_a), object_to_shape(placed_b)
    _PLACED_SHAPES_CACHE[cache_key] = shapes
    return shapes


def _geometry_cache_key(row: PublicTaskRow) -> str:
    return row.hashes.geometry_hash or row.geometry_ids[0] or row.id


def _metadata_bbox(row: PublicTaskRow, key: str) -> BoundingBox | None:
    value = row.metadata.get(key)
    if value is None:
        return None
    return BoundingBox.model_validate(value)


def _translated_bbox(bbox: BoundingBox, vector: tuple[float, float, float]) -> BoundingBox:
    return BoundingBox(
        min=tuple(bbox.min[index] + vector[index] for index in range(3)),
        max=tuple(bbox.max[index] + vector[index] for index in range(3)),
    )


def _boxes_overlap(a: BoundingBox, b: BoundingBox) -> bool:
    return all(a.min[index] <= b.max[index] and b.min[index] <= a.max[index] for index in range(3))


def _candidate_by_direction(row: PublicTaskRow, direction: str) -> dict[str, Any]:
    for candidate in _candidate_moves(row):
        if candidate["direction"] == direction:
            return candidate
    raise ValueError(f"repair_direction metadata missing candidate {direction}")


def _candidate_moves(row: PublicTaskRow) -> list[dict[str, Any]]:
    if row.metadata.get("repair_policy") != REPAIR_POLICY_NAME:
        raise ValueError("unsupported repair policy")
    candidate_moves = row.metadata.get("candidate_moves")
    if not isinstance(candidate_moves, list):
        raise ValueError("repair_direction metadata requires candidate_moves")
    candidates: list[dict[str, Any]] = []
    for item in candidate_moves:
        if not isinstance(item, dict):
            raise ValueError("repair_direction candidate moves must be objects")
        direction = item.get("direction")
        if direction not in REPAIR_TIE_BREAK_ORDER:
            raise ValueError("repair_direction candidate has invalid direction")
        candidates.append(item)
    return candidates


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, int]:
    magnitude = candidate.get("magnitude_mm")
    if not isinstance(magnitude, int | float) or not math.isfinite(float(magnitude)):
        raise ValueError("repair_direction candidate requires finite magnitude_mm")
    return float(magnitude), REPAIR_TIE_BREAK_ORDER.index(str(candidate["direction"]))


def _candidate_vector(candidate: dict[str, Any]) -> tuple[float, float, float]:
    value = candidate.get("translation_vector_mm")
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError("repair_direction candidate requires translation_vector_mm")
    vector = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in vector):
        raise ValueError("repair_direction candidate vector must be finite")
    _validate_direction_vector(str(candidate["direction"]), vector)
    return vector


def _require_repair_row(row: PublicTaskRow) -> None:
    if row.task_type not in REPAIR_TASK_TYPES:
        raise ValueError("repair verifier requires an IntersectionEdit repair row")


def _expected_candidates_from_bboxes(row: PublicTaskRow) -> dict[str, dict[str, Any]]:
    bbox_a = _bbox_from_metadata(row, "bbox_a")
    bbox_b = _bbox_from_metadata(row, "bbox_b")
    clearance = float(row.label_policy.epsilon_distance_mm)
    expected: dict[str, dict[str, Any]] = {}
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        positive_delta = bbox_a.max[axis_index] - bbox_b.min[axis_index] + clearance
        negative_delta = bbox_a.min[axis_index] - bbox_b.max[axis_index] - clearance
        expected[f"+{axis_name}"] = _candidate_metadata(
            f"+{axis_name}",
            axis_index,
            max(0.0, positive_delta),
        )
        expected[f"-{axis_name}"] = _candidate_metadata(
            f"-{axis_name}",
            axis_index,
            min(0.0, negative_delta),
        )
    return expected


def _candidate_metadata(direction: str, axis_index: int, delta: float) -> dict[str, Any]:
    vector = [0.0, 0.0, 0.0]
    vector[axis_index] = float(delta)
    return {
        "direction": direction,
        "magnitude_mm": abs(float(delta)),
        "translation_vector_mm": tuple(vector),
    }


def _bbox_from_metadata(row: PublicTaskRow, key: str) -> BoundingBox:
    value = row.metadata.get(key)
    if value is None:
        raise ValueError(f"repair row requires {key} metadata")
    return BoundingBox.model_validate(value)


def _selected_direction_from_metadata(row: PublicTaskRow) -> str:
    selected = row.metadata.get("selected_direction")
    if selected not in REPAIR_TIE_BREAK_ORDER:
        raise ValueError("repair row has invalid selected_direction")
    return str(selected)


def _validate_direction_vector(direction: str, vector: tuple[float, float, float]) -> None:
    axis_by_name = {"x": 0, "y": 1, "z": 2}
    axis = axis_by_name[direction[1]]
    for index, value in enumerate(vector):
        if index != axis and not math.isclose(value, 0.0, abs_tol=1e-9):
            raise ValueError("repair_direction vector must move only along its direction axis")
    axis_value = vector[axis]
    if direction[0] == "+" and axis_value < -1e-9:
        raise ValueError("repair_direction vector sign does not match direction")
    if direction[0] == "-" and axis_value > 1e-9:
        raise ValueError("repair_direction vector sign does not match direction")


def _counts(values: Iterable[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _output_breakdown(
    rows: list[PublicTaskRow],
    results: list[RepairVerificationResult],
) -> dict[str, dict[str, int]]:
    answer_by_id = {row.id: row.answer for row in rows}
    breakdown: dict[str, dict[str, int]] = {}
    for result in results:
        key = result.parsed_output or "invalid"
        item = breakdown.setdefault(
            key,
            {
                "total": 0,
                "repaired": 0,
                "exact_answer_correct": 0,
                "invalid": 0,
            },
        )
        item["total"] += 1
        if result.repaired:
            item["repaired"] += 1
        if result.parsed_output is None:
            item["invalid"] += 1
        if result.parsed_output == answer_by_id[result.row_id]:
            item["exact_answer_correct"] += 1
    return dict(sorted(breakdown.items()))
