"""Reusable verifier-style rewards for IntersectionEdit and IntersectionQA rows."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from intersectionqa.enums import TaskType
from intersectionqa.evaluation.parsing import canonical_answer_candidate, parse_answer
from intersectionqa.schema import PublicTaskRow

AXIS_REPAIR_RE = re.compile(r"^direction=(\+x|-x|\+y|-y|\+z|-z), distance_mm=([0-9]+)\.([0-9])$")
SIGNED_DISTANCE_RE = re.compile(r"^distance_mm=(-?[0-9]+)\.([0-9])$")
TRANSLATION_VECTOR_RE = re.compile(
    r"^dx=(-?[0-9]+)\.([0-9]), dy=(-?[0-9]+)\.([0-9]), dz=(-?[0-9]+)\.([0-9])$"
)
EDIT_PROGRAM_RE = re.compile(
    r"^object_b = object_b\.translate\(\((-?[0-9]+)\.([0-9]), (-?[0-9]+)\.([0-9]), (-?[0-9]+)\.([0-9])\)\)$"
)


@dataclass(frozen=True)
class RewardResult:
    row_id: str
    task_type: TaskType
    output: str
    parsed_output: str | None
    reward: float
    components: dict[str, float]
    failure_reason: str | None = None


def reward_prediction(row: PublicTaskRow, output: str) -> RewardResult:
    return reward_from_fields(
        row_id=row.id,
        task_type=row.task_type,
        answer=row.answer,
        metadata=row.metadata,
        output=output,
    )


def reward_from_fields(
    *,
    row_id: str,
    task_type: TaskType | str,
    answer: str,
    metadata: dict[str, Any],
    output: str,
) -> RewardResult:
    task_type = TaskType(task_type)
    candidate_output, format_components = canonical_answer_candidate(output)
    parsed = parse_answer(task_type, candidate_output)
    if parsed is None:
        scaffold_reward = _format_scaffold_reward(format_components)
        return RewardResult(
            row_id=row_id,
            task_type=task_type,
            output=output,
            parsed_output=None,
            reward=scaffold_reward,
            components={
                **format_components,
                "answer_format": 0.0,
                "format_scaffold": scaffold_reward,
            },
            failure_reason="invalid_output",
        )

    if task_type in {TaskType.AXIS_ALIGNED_REPAIR, TaskType.TARGET_CLEARANCE_REPAIR}:
        reward, components = _axis_repair_reward(answer, parsed, metadata)
    elif task_type in {TaskType.AXIS_ALIGNED_REPAIR_VECTOR, TaskType.AXIS_ALIGNED_REPAIR_PROGRAM}:
        reward, components = _vector_repair_reward(answer, parsed, metadata, task_type)
    elif task_type in {
        TaskType.TARGET_CLEARANCE_MOVE,
        TaskType.TARGET_CONTACT_MOVE,
        TaskType.CENTROID_DISTANCE_MOVE,
    }:
        reward, components = _signed_distance_reward(answer, parsed, metadata)
    elif task_type == TaskType.EDIT_CANDIDATE_SELECTION:
        reward, components = _candidate_selection_reward(parsed, metadata)
    elif task_type == TaskType.EDIT_CANDIDATE_RANKING:
        reward, components = _candidate_ranking_reward(answer, parsed, metadata)
    else:
        reward = 1.0 if parsed == answer else 0.0
        components = {"format": 1.0, "exact": reward}

    answer_format_reward = 0.05 * format_components["format"]
    final_reward = max(_clamp01(reward * format_components["format"]), answer_format_reward)
    return RewardResult(
        row_id=row_id,
        task_type=task_type,
        output=output,
        parsed_output=parsed,
        reward=final_reward,
        components={
            key: _clamp01(value)
            for key, value in {
                **format_components,
                **components,
                "answer_format": 1.0,
                "answer_format_reward": answer_format_reward,
            }.items()
        },
    )


def _format_scaffold_reward(format_components: dict[str, float]) -> float:
    if format_components.get("answer_tag", 0.0) <= 0.0:
        return 0.0
    return 0.03 + 0.02 * format_components.get("reasoning_format", 0.0)


def _axis_repair_reward(
    answer: str,
    parsed: str,
    metadata: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    expected = _axis_answer_parts(answer)
    predicted = _axis_answer_parts(parsed)
    if expected is None or predicted is None:
        return 0.0, {"format": 1.0, "parse_error": 1.0}

    expected_direction, expected_distance = expected
    predicted_direction, predicted_distance = predicted
    tolerance = _acceptance_tolerance(metadata)
    direction_score = 1.0 if predicted_direction == expected_direction else 0.0
    distance_error = abs(predicted_distance - expected_distance)
    within_tolerance = 1.0 if direction_score and distance_error <= tolerance else 0.0
    distance_score = max(0.0, 1.0 - distance_error / max(tolerance * 4.0, 1e-9))
    movement_score = _movement_score(
        predicted_distance,
        expected_distance,
        metadata.get("target", {}),
    )
    target_score = within_tolerance
    reward = (
        0.10
        + 0.30 * direction_score
        + 0.35 * distance_score * direction_score
        + 0.15 * movement_score * direction_score
        + 0.10 * target_score
    )
    return reward, {
        "format": 1.0,
        "direction": direction_score,
        "distance": distance_score * direction_score,
        "within_tolerance": within_tolerance,
        "movement": movement_score * direction_score,
        "target": target_score,
    }


def _candidate_selection_reward(
    parsed: str,
    metadata: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    candidates = _candidate_edits(metadata)
    selected = candidates.get(parsed)
    if selected is None:
        return 0.0, {"format": 1.0, "candidate_exists": 0.0}
    best = str(metadata.get("candidate_selection_answer", ""))
    candidate_score = _candidate_score(selected, candidates)
    exact = 1.0 if parsed == best else 0.0
    reward = max(exact, candidate_score)
    return reward, {
        "format": 1.0,
        "exact": exact,
        "candidate_score": candidate_score,
        "target": 1.0 if selected.get("satisfies_target") is True else 0.0,
        "non_intersection": 1.0 if selected.get("no_interference") is True else 0.0,
    }


def _vector_repair_reward(
    answer: str,
    parsed: str,
    metadata: dict[str, Any],
    task_type: TaskType,
) -> tuple[float, dict[str, float]]:
    expected = _vector_answer_parts(answer, task_type)
    predicted = _vector_answer_parts(parsed, task_type)
    if expected is None or predicted is None:
        return 0.0, {"format": 1.0, "parse_error": 1.0}
    tolerance = _acceptance_tolerance(metadata)
    error = _vector_l2(tuple(predicted[index] - expected[index] for index in range(3)))
    within_tolerance = 1.0 if error <= tolerance else 0.0
    distance_score = max(0.0, 1.0 - error / max(tolerance * 4.0, 1e-9))
    expected_magnitude = _vector_l2(expected)
    predicted_magnitude = _vector_l2(predicted)
    movement_score = _movement_score(predicted_magnitude, expected_magnitude, metadata.get("target", {}))
    direction_score = 1.0 if _same_vector_direction(expected, predicted) else 0.0
    reward = (
        0.10
        + 0.25 * direction_score
        + 0.40 * distance_score
        + 0.15 * movement_score
        + 0.10 * within_tolerance
    )
    return reward, {
        "format": 1.0,
        "direction": direction_score,
        "vector_l2": distance_score,
        "movement": movement_score,
        "within_tolerance": within_tolerance,
    }


def _signed_distance_reward(
    answer: str,
    parsed: str,
    metadata: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    expected = _signed_distance_answer_part(answer)
    predicted = _signed_distance_answer_part(parsed)
    if expected is None or predicted is None:
        return 0.0, {"format": 1.0, "parse_error": 1.0}
    tolerance = _acceptance_tolerance(metadata)
    distance_error = abs(predicted - expected)
    within_tolerance = 1.0 if distance_error <= tolerance else 0.0
    distance_score = max(0.0, 1.0 - distance_error / max(tolerance * 4.0, 1e-9))
    movement_score = _movement_score(abs(predicted), abs(expected), metadata.get("target", {}))
    sign_score = 1.0 if (predicted == 0.0 and expected == 0.0) or (predicted * expected > 0.0) else 0.0
    reward = (
        0.10
        + 0.20 * sign_score
        + 0.40 * distance_score
        + 0.20 * movement_score
        + 0.10 * within_tolerance
    )
    return reward, {
        "format": 1.0,
        "sign": sign_score,
        "distance": distance_score,
        "movement": movement_score,
        "within_tolerance": within_tolerance,
    }


def _candidate_ranking_reward(
    answer: str,
    parsed: str,
    metadata: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    candidates = _candidate_edits(metadata)
    if set(parsed) != set(candidates):
        return 0.0, {"format": 1.0, "candidate_set": 0.0}
    exact = 1.0 if parsed == answer else 0.0
    pairwise = _pairwise_ranking_accuracy(answer, parsed)
    top = 1.0 if parsed[0] == answer[0] else 0.0
    reward = 0.15 + 0.70 * pairwise + 0.15 * top
    return reward, {
        "format": 1.0,
        "exact": exact,
        "pairwise_ranking": pairwise,
        "top_candidate": top,
    }


def _candidate_edits(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = metadata.get("candidate_edits")
    if not isinstance(value, dict):
        return {}
    return {
        str(label): candidate
        for label, candidate in value.items()
        if isinstance(candidate, dict)
    }


def _candidate_score(candidate: dict[str, Any], candidates: dict[str, dict[str, Any]]) -> float:
    movement = _finite_float(candidate.get("movement_magnitude"))
    if movement is None:
        return 0.0
    satisfying_movements = [
        value
        for item in candidates.values()
        if item.get("satisfies_target") is True
        for value in [_finite_float(item.get("movement_magnitude"))]
        if value is not None
    ]
    reference = min(satisfying_movements) if satisfying_movements else movement
    movement_ratio = reference / max(movement, reference, 1e-9)
    if candidate.get("satisfies_target") is True:
        return 0.70 + 0.30 * movement_ratio
    if candidate.get("no_interference") is True:
        clearance_error = _finite_float(candidate.get("clearance_error_mm"))
        clearance_score = 0.0 if clearance_error is None else max(0.0, 1.0 - clearance_error)
        return 0.35 + 0.25 * clearance_score
    return 0.10 * movement_ratio


def _movement_score(
    predicted_distance: float,
    expected_distance: float,
    target: object,
) -> float:
    if predicted_distance < 0.0:
        return 0.0
    if not isinstance(target, dict):
        target = {}
    target_clearance = _finite_float(target.get("target_clearance_mm"))
    reference = max(expected_distance, target_clearance or 0.0, 1e-9)
    overshoot = max(0.0, predicted_distance - expected_distance)
    return max(0.0, 1.0 - overshoot / max(reference, 1e-9))


def _axis_answer_parts(value: str) -> tuple[str, float] | None:
    match = AXIS_REPAIR_RE.match(value)
    if match is None:
        return None
    return match.group(1), float(f"{match.group(2)}.{match.group(3)}")


def _signed_distance_answer_part(value: str) -> float | None:
    match = SIGNED_DISTANCE_RE.match(value)
    if match is None:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def _vector_answer_parts(value: str, task_type: TaskType) -> tuple[float, float, float] | None:
    match = (
        EDIT_PROGRAM_RE.match(value)
        if task_type == TaskType.AXIS_ALIGNED_REPAIR_PROGRAM
        else TRANSLATION_VECTOR_RE.match(value)
    )
    if match is None:
        return None
    return tuple(float(f"{match.group(index)}.{match.group(index + 1)}") for index in (1, 3, 5))  # type: ignore[return-value]


def _vector_l2(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _same_vector_direction(expected: tuple[float, float, float], predicted: tuple[float, float, float]) -> bool:
    expected_mag = _vector_l2(expected)
    predicted_mag = _vector_l2(predicted)
    if expected_mag <= 1e-9 or predicted_mag <= 1e-9:
        return expected_mag <= 1e-9 and predicted_mag <= 1e-9
    dot = sum(expected[index] * predicted[index] for index in range(3))
    return dot > 0.0


def _acceptance_tolerance(metadata: dict[str, Any]) -> float:
    policy = metadata.get("numeric_output_policy")
    if not isinstance(policy, dict):
        return 0.15
    value = _finite_float(policy.get("acceptance_tolerance_mm"))
    return value if value is not None and value > 0.0 else 0.15


def _pairwise_ranking_accuracy(expected: str, predicted: str) -> float:
    total = 0
    correct = 0
    expected_position = {label: index for index, label in enumerate(expected)}
    predicted_position = {label: index for index, label in enumerate(predicted)}
    for left_index, left in enumerate(expected):
        for right in expected[left_index + 1 :]:
            total += 1
            if (expected_position[left] < expected_position[right]) == (
                predicted_position[left] < predicted_position[right]
            ):
                correct += 1
    return correct / total if total else 0.0


def _finite_float(value: object) -> float | None:
    if not isinstance(value, int | float):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))
