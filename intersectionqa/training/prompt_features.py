"""Opt-in prompt feature augmentation for training experiments."""

from __future__ import annotations

from typing import Any, Mapping

from intersectionqa.enums import TaskType
from intersectionqa.schema import PublicTaskRow

PROMPT_FEATURE_MODES = ("none", "edit_geometry")
_EDIT_TASKS = {
    TaskType.REPAIR_DIRECTION.value,
    TaskType.REPAIR_TRANSLATION.value,
    TaskType.AXIS_ALIGNED_REPAIR.value,
    TaskType.AXIS_ALIGNED_REPAIR_VECTOR.value,
    TaskType.AXIS_ALIGNED_REPAIR_PROGRAM.value,
    TaskType.TARGET_CLEARANCE_REPAIR.value,
    TaskType.TARGET_CLEARANCE_MOVE.value,
    TaskType.TARGET_CONTACT_MOVE.value,
    TaskType.CENTROID_DISTANCE_MOVE.value,
    TaskType.EDIT_CANDIDATE_SELECTION.value,
    TaskType.EDIT_CANDIDATE_RANKING.value,
}
_REPAIR_TASKS = {
    TaskType.REPAIR_DIRECTION.value,
    TaskType.REPAIR_TRANSLATION.value,
    TaskType.AXIS_ALIGNED_REPAIR.value,
    TaskType.AXIS_ALIGNED_REPAIR_VECTOR.value,
    TaskType.AXIS_ALIGNED_REPAIR_PROGRAM.value,
    TaskType.TARGET_CLEARANCE_REPAIR.value,
}
_SIGNED_DISTANCE_TASKS = {
    TaskType.TARGET_CLEARANCE_MOVE.value,
    TaskType.TARGET_CONTACT_MOVE.value,
    TaskType.CENTROID_DISTANCE_MOVE.value,
}
def prompt_for_row(row: PublicTaskRow, *, mode: str = "none") -> str:
    metadata = {
        **row.metadata,
        "label_policy": row.label_policy.model_dump(mode="json"),
    }
    return augment_prompt(
        prompt=row.prompt,
        task_type=str(row.task_type),
        metadata=metadata,
        mode=mode,
    )


def prompt_for_mapping(row: Mapping[str, Any], *, mode: str = "none") -> str:
    return augment_prompt(
        prompt=str(row["prompt"]),
        task_type=str(row["task_type"]),
        metadata=_metadata_from_mapping(row),
        mode=mode,
    )


def augment_prompt(
    *,
    prompt: str,
    task_type: str,
    metadata: Mapping[str, Any],
    mode: str = "none",
) -> str:
    if mode == "none":
        return prompt
    if mode != "edit_geometry":
        raise ValueError(f"Unsupported prompt feature mode: {mode}")
    if task_type not in _EDIT_TASKS:
        return prompt
    lines = edit_geometry_feature_lines(task_type, metadata)
    if not lines:
        return prompt
    return (
        f"{prompt.rstrip()}\n\n"
        "Trusted geometry features for this training/evaluation run:\n"
        + "\n".join(lines)
    )


def edit_geometry_feature_lines(task_type: str, metadata: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    bbox_a = metadata.get("bbox_a")
    bbox_b = metadata.get("bbox_b")
    if isinstance(bbox_a, Mapping):
        lines.append(f"- object_a_world_aabb: {_format_bbox(bbox_a)}")
    if isinstance(bbox_b, Mapping):
        lines.append(f"- object_b_world_aabb: {_format_bbox(bbox_b)}")
    if task_type in _REPAIR_TASKS:
        lines.extend(_repair_lines(metadata))
    if task_type in _SIGNED_DISTANCE_TASKS:
        lines.extend(_signed_distance_lines(metadata))
    return lines


def _repair_lines(metadata: Mapping[str, Any]) -> list[str]:
    lines = [
        "- conservative_repair_rule: compute signed-axis AABB separating moves "
        "and choose the smallest magnitude; tie order is +x, -x, +y, -y, +z, -z."
    ]
    epsilon = _finite_float(_nested_get(metadata, "label_policy", "epsilon_distance_mm"))
    if epsilon is None:
        epsilon = _finite_float(metadata.get("epsilon_distance_mm"))
    if epsilon is not None:
        lines.append(f"- contact_tolerance_mm: {_fmt(epsilon)}")
    return lines


def _signed_distance_lines(metadata: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    allowed_edit = metadata.get("allowed_edit")
    if isinstance(allowed_edit, Mapping):
        direction = allowed_edit.get("direction")
        direction_vector = allowed_edit.get("direction_vector")
        if direction is not None:
            lines.append(f"- allowed_signed_direction: {direction}")
        elif isinstance(direction_vector, list | tuple) and len(direction_vector) == 3:
            lines.append(f"- allowed_direction_vector: {_format_tuple(direction_vector)}")
        positive = allowed_edit.get("positive_distance_effect")
        negative = allowed_edit.get("negative_distance_effect")
        if positive is not None:
            lines.append(f"- positive_distance_effect: {positive}")
        if negative is not None:
            lines.append(f"- negative_distance_effect: {negative}")
    initial = metadata.get("initial_state")
    if isinstance(initial, Mapping):
        relation = initial.get("relation")
        if relation is not None:
            lines.append(f"- initial_relation: {relation}")
        clearance = _finite_float(initial.get("minimum_distance"))
        if clearance is not None:
            lines.append(f"- initial_clearance_mm: {_fmt(clearance)}")
        centroid_distance = _finite_float(initial.get("centroid_distance_mm"))
        if centroid_distance is not None:
            lines.append(f"- initial_centroid_distance_mm: {_fmt(centroid_distance)}")
    target = metadata.get("target")
    if isinstance(target, Mapping):
        target_clearance = _finite_float(target.get("target_clearance_mm"))
        if target_clearance is not None:
            lines.append(f"- target_clearance_mm: {_fmt(target_clearance)}")
        target_centroid = _finite_float(target.get("target_centroid_distance_mm"))
        if target_centroid is not None:
            lines.append(f"- target_centroid_distance_mm: {_fmt(target_centroid)}")
    return lines


def _format_bbox(value: Mapping[str, Any]) -> str:
    return f"min={_format_tuple(value.get('min'))}, max={_format_tuple(value.get('max'))}"


def _format_tuple(value: Any) -> str:
    if not isinstance(value, list | tuple):
        return str(value)
    return "(" + ", ".join(_fmt(float(item)) for item in value) + ")"


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def _metadata_from_mapping(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        import json

        return json.loads(metadata)
    if isinstance(metadata, Mapping):
        return metadata
    return {}
