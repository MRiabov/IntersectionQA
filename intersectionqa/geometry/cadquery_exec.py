"""CadQuery execution and exact geometry measurement helpers."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from typing import Any

from intersectionqa.enums import BooleanStatus, DistanceStatus, FailureReason
from intersectionqa.geometry.bbox import AABB, aabb_overlap
from intersectionqa.geometry.labels import RawGeometry
from intersectionqa.schema import BoundingBox, LabelPolicy, SourceObjectRecord, Transform


class CadQueryExecutionError(RuntimeError):
    def __init__(self, failure_reason: FailureReason, message: str) -> None:
        super().__init__(message)
        self.failure_reason = failure_reason


@dataclass(frozen=True)
class MeasuredObject:
    shape: Any
    volume: float
    bbox: BoundingBox


def cadquery_available() -> bool:
    return importlib.util.find_spec("cadquery") is not None


def cadquery_version() -> str | None:
    if not cadquery_available():
        return None
    import cadquery as cq

    return getattr(cq, "__version__", None)


def ocp_version() -> str | None:
    try:
        import OCP
    except Exception:
        return None
    return getattr(OCP, "__version__", None)


def execute_source_object(record: SourceObjectRecord) -> MeasuredObject:
    shape = execute_object_code(record.normalized_code, record.object_function_name)
    return measure_shape(shape)


def execute_object_code(code: str, function_name: str) -> Any:
    if not cadquery_available():
        raise CadQueryExecutionError(
            FailureReason.SOURCE_EXEC_ERROR,
            "cadquery is not installed",
        )
    import cadquery as cq

    namespace: dict[str, Any] = {"cq": cq, "cadquery": cq, "__builtins__": __builtins__}
    try:
        compiled = compile(code, "<intersectionqa-source>", "exec")
    except SyntaxError as exc:
        raise CadQueryExecutionError(FailureReason.SOURCE_PARSE_ERROR, str(exc)) from exc
    try:
        exec(compiled, namespace)
        function = namespace.get(function_name)
        if not callable(function):
            raise CadQueryExecutionError(
                FailureReason.MISSING_RESULT_OBJECT,
                f"missing callable object function: {function_name}",
            )
        result = function()
    except CadQueryExecutionError:
        raise
    except Exception as exc:
        raise CadQueryExecutionError(FailureReason.SOURCE_EXEC_ERROR, str(exc)) from exc
    return object_to_shape(result)


def object_to_shape(value: Any) -> Any:
    import cadquery as cq

    if isinstance(value, cq.Workplane):
        values = value.vals()
        if not values:
            raise CadQueryExecutionError(
                FailureReason.MISSING_RESULT_OBJECT,
                "CadQuery Workplane produced no values",
            )
        if len(values) == 1:
            return values[0]
        return cq.Compound.makeCompound(values)
    if all(hasattr(value, name) for name in ("Volume", "BoundingBox", "ShapeType")):
        return value
    raise CadQueryExecutionError(
        FailureReason.INVALID_CADQUERY_TYPE,
        f"unsupported result type: {type(value).__name__}",
    )


def measure_shape(shape: Any) -> MeasuredObject:
    try:
        if hasattr(shape, "isValid") and not shape.isValid():
            raise CadQueryExecutionError(FailureReason.NON_SOLID_RESULT, "shape is not valid")
        shape_type = shape.ShapeType()
        if shape_type not in {"Solid", "Compound"}:
            raise CadQueryExecutionError(
                FailureReason.NON_SOLID_RESULT,
                f"expected Solid or Compound, got {shape_type}",
            )
        volume = float(shape.Volume())
        if not math.isfinite(volume) or volume <= 0.0:
            raise CadQueryExecutionError(
                FailureReason.ZERO_OR_NEGATIVE_VOLUME,
                f"shape volume is not positive finite: {volume}",
            )
        bbox = bounding_box_from_shape(shape)
    except CadQueryExecutionError:
        raise
    except Exception as exc:
        raise CadQueryExecutionError(FailureReason.UNKNOWN_ERROR, str(exc)) from exc
    return MeasuredObject(shape=shape, volume=volume, bbox=bbox)


def bounding_box_from_shape(shape: Any) -> BoundingBox:
    try:
        bbox = shape.BoundingBox()
        result = BoundingBox(
            min=(float(bbox.xmin), float(bbox.ymin), float(bbox.zmin)),
            max=(float(bbox.xmax), float(bbox.ymax), float(bbox.zmax)),
        )
    except Exception as exc:
        raise CadQueryExecutionError(FailureReason.NON_FINITE_BBOX, str(exc)) from exc
    return result


def apply_transform(shape: Any, transform: Transform) -> Any:
    transformed = shape
    transformed = transformed.rotate((0, 0, 0), (1, 0, 0), transform.rotation_xyz_deg[0])
    transformed = transformed.rotate((0, 0, 0), (0, 1, 0), transform.rotation_xyz_deg[1])
    transformed = transformed.rotate((0, 0, 0), (0, 0, 1), transform.rotation_xyz_deg[2])
    transformed = transformed.translate(tuple(transform.translation))
    return transformed


def measure_source_pair(
    object_a: SourceObjectRecord,
    object_b: SourceObjectRecord,
    transform_a: Transform,
    transform_b: Transform,
    policy: LabelPolicy,
) -> RawGeometry:
    measured_a = execute_source_object(object_a)
    measured_b = execute_source_object(object_b)
    placed_a = apply_transform(measured_a.shape, transform_a)
    placed_b = apply_transform(measured_b.shape, transform_b)
    placed_a_measure = measure_shape(placed_a)
    placed_b_measure = measure_shape(placed_b)
    bbox_a = _aabb_from_schema(placed_a_measure.bbox)
    bbox_b = _aabb_from_schema(placed_b_measure.bbox)
    overlap = aabb_overlap(bbox_a, bbox_b)

    intersection_volume, boolean_status = _intersection_volume(placed_a, placed_b)
    minimum_distance, distance_status = _minimum_distance(
        placed_a,
        placed_b,
        intersection_volume,
        policy,
        placed_a_measure.volume,
        placed_b_measure.volume,
    )
    contains_a_in_b, contains_b_in_a = _containment_flags(
        intersection_volume,
        placed_a_measure.volume,
        placed_b_measure.volume,
        policy,
    )
    return RawGeometry(
        volume_a=placed_a_measure.volume,
        volume_b=placed_b_measure.volume,
        intersection_volume=intersection_volume,
        minimum_distance=minimum_distance,
        contains_a_in_b=contains_a_in_b,
        contains_b_in_a=contains_b_in_a,
        aabb_overlap=overlap,
        boolean_status=boolean_status,
        distance_status=distance_status,
    )


def _intersection_volume(shape_a: Any, shape_b: Any) -> tuple[float | None, BooleanStatus]:
    try:
        intersection = shape_a.intersect(shape_b)
        return max(0.0, float(intersection.Volume())), BooleanStatus.OK
    except Exception:
        return None, BooleanStatus.FAILED


def _minimum_distance(
    shape_a: Any,
    shape_b: Any,
    intersection_volume: float | None,
    policy: LabelPolicy,
    volume_a: float,
    volume_b: float,
) -> tuple[float | None, DistanceStatus]:
    if intersection_volume is not None and intersection_volume > policy.epsilon_volume(volume_a, volume_b):
        return 0.0, DistanceStatus.SKIPPED_POSITIVE_OVERLAP
    try:
        return max(0.0, float(shape_a.distance(shape_b))), DistanceStatus.OK
    except Exception:
        return None, DistanceStatus.FAILED


def _containment_flags(
    intersection_volume: float | None,
    volume_a: float,
    volume_b: float,
    policy: LabelPolicy,
) -> tuple[bool, bool]:
    if intersection_volume is None:
        return False, False
    tolerance = policy.epsilon_volume(volume_a, volume_b)
    contains_a_in_b = abs(intersection_volume - volume_a) <= tolerance
    contains_b_in_a = abs(intersection_volume - volume_b) <= tolerance
    return contains_a_in_b, contains_b_in_a


def _aabb_from_schema(bbox: BoundingBox) -> AABB:
    return AABB(min=bbox.min, max=bbox.max)
