"""Oriented bounding-box baseline for binary interference."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from math import cos, radians, sin
import re
from typing import Iterable

from intersectionqa.enums import TaskType
from intersectionqa.evaluation.aabb import BaselineResult
from intersectionqa.geometry.cadquery_exec import bounding_box_from_shape, execute_object_code
from intersectionqa.prompts.common import object_code_from_script
from intersectionqa.schema import BoundingBox, PublicTaskRow, Transform

_PLACE_RE = re.compile(
    r"^\s*(?P<name>[ab])\s*=\s*_place\("
    r"object_(?P<object_name>[ab])\(\),\s*"
    r"(?P<translation>\([^)]*\)),\s*"
    r"(?P<rotation>\([^)]*\))\)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class OrientedBox:
    center: tuple[float, float, float]
    axes: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    half_extents: tuple[float, float, float]


def predict_binary_from_obb(row: PublicTaskRow) -> str:
    return "yes" if obb_overlap_for_row(row) else "no"


def evaluate_obb_binary(rows: Iterable[PublicTaskRow]) -> BaselineResult:
    binary_rows = [row for row in rows if row.task_type == TaskType.BINARY_INTERFERENCE]
    predictions: dict[str, str] = {}
    invalid = 0
    for row in binary_rows:
        try:
            predictions[row.id] = predict_binary_from_obb(row)
        except Exception:
            invalid += 1
    correct = sum(1 for row in binary_rows if predictions.get(row.id) == row.answer)
    total = len(binary_rows)
    return BaselineResult(
        total=total,
        correct=correct,
        accuracy=correct / total if total else 0.0,
        invalid_output_rate=invalid / total if total else 0.0,
        per_relation_accuracy=_group_accuracy(binary_rows, predictions, lambda row: row.labels.relation),
        per_split_accuracy=_group_accuracy(binary_rows, predictions, lambda row: row.split),
        per_difficulty_accuracy=_difficulty_accuracy(binary_rows, predictions),
    )


def obb_overlap_for_row(row: PublicTaskRow) -> bool:
    transforms = _transforms_from_script(row.script)
    object_code = object_code_from_script(row.script)
    bbox_a = bounding_box_from_shape(execute_object_code(object_code, "object_a"))
    bbox_b = bounding_box_from_shape(execute_object_code(object_code, "object_b"))
    return obb_overlap(
        obb_from_bbox(bbox_a, transforms["a"]),
        obb_from_bbox(bbox_b, transforms["b"]),
    )


def obb_from_bbox(bbox: BoundingBox, transform: Transform) -> OrientedBox:
    local_center = tuple((bbox.min[index] + bbox.max[index]) / 2.0 for index in range(3))
    half_extents = tuple((bbox.max[index] - bbox.min[index]) / 2.0 for index in range(3))
    axes = tuple(_rotate_vector(axis, transform.rotation_xyz_deg) for axis in _BASIS)
    center = _add(_rotate_vector(local_center, transform.rotation_xyz_deg), transform.translation)
    return OrientedBox(center=center, axes=axes, half_extents=half_extents)  # type: ignore[arg-type]


def obb_overlap(left: OrientedBox, right: OrientedBox, tolerance: float = 1e-9) -> bool:
    rotation = [[_dot(left.axes[i], right.axes[j]) for j in range(3)] for i in range(3)]
    abs_rotation = [[abs(rotation[i][j]) + tolerance for j in range(3)] for i in range(3)]
    center_delta = _sub(right.center, left.center)
    projected_delta = [_dot(center_delta, left.axes[i]) for i in range(3)]

    for i in range(3):
        radius_left = left.half_extents[i]
        radius_right = sum(right.half_extents[j] * abs_rotation[i][j] for j in range(3))
        if abs(projected_delta[i]) > radius_left + radius_right:
            return False

    for j in range(3):
        radius_left = sum(left.half_extents[i] * abs_rotation[i][j] for i in range(3))
        radius_right = right.half_extents[j]
        distance = abs(sum(projected_delta[i] * rotation[i][j] for i in range(3)))
        if distance > radius_left + radius_right:
            return False

    for i in range(3):
        for j in range(3):
            radius_left = (
                left.half_extents[(i + 1) % 3] * abs_rotation[(i + 2) % 3][j]
                + left.half_extents[(i + 2) % 3] * abs_rotation[(i + 1) % 3][j]
            )
            radius_right = (
                right.half_extents[(j + 1) % 3] * abs_rotation[i][(j + 2) % 3]
                + right.half_extents[(j + 2) % 3] * abs_rotation[i][(j + 1) % 3]
            )
            distance = abs(
                projected_delta[(i + 2) % 3] * rotation[(i + 1) % 3][j]
                - projected_delta[(i + 1) % 3] * rotation[(i + 2) % 3][j]
            )
            if distance > radius_left + radius_right:
                return False
    return True


def _transforms_from_script(script: str) -> dict[str, Transform]:
    transforms: dict[str, Transform] = {}
    for match in _PLACE_RE.finditer(script):
        name = match.group("name")
        translation = _triplet(match.group("translation"))
        rotation = _triplet(match.group("rotation"))
        transforms[name] = Transform(translation=translation, rotation_xyz_deg=rotation)
    if set(transforms) != {"a", "b"}:
        raise ValueError("could not parse object_a/object_b transforms from assembly script")
    return transforms


def _triplet(value: str) -> tuple[float, float, float]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, tuple) or len(parsed) != 3:
        raise ValueError(f"expected transform triplet, got {value}")
    return tuple(float(item) for item in parsed)


_BASIS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _rotate_vector(
    vector: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z = vector
    rx, ry, rz = (radians(value) for value in rotation_xyz_deg)
    cx, sx = cos(rx), sin(rx)
    cy, sy = cos(ry), sin(ry)
    cz, sz = cos(rz), sin(rz)
    y, z = y * cx - z * sx, y * sx + z * cx
    x, z = x * cy + z * sy, -x * sy + z * cy
    x, y = x * cz - y * sz, x * sz + y * cz
    return (x, y, z)


def _add(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def _sub(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _group_accuracy(rows, predictions, key_fn) -> dict[str, float]:
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for row in rows:
        key = key_fn(row)
        totals[key] = totals.get(key, 0) + 1
        correct[key] = correct.get(key, 0) + int(predictions.get(row.id) == row.answer)
    return {key: correct[key] / totals[key] for key in sorted(totals)}


def _difficulty_accuracy(rows, predictions) -> dict[str, float]:
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for row in rows:
        for tag in row.difficulty_tags:
            totals[tag] = totals.get(tag, 0) + 1
            correct[tag] = correct.get(tag, 0) + int(predictions.get(row.id) == row.answer)
    return {key: correct[key] / totals[key] for key in sorted(totals)}
