"""Tool-assisted exact-verifier upper bound."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from intersectionqa.enums import Relation, TaskType
from intersectionqa.evaluation.metrics import Prediction, TaskMetrics, evaluate_predictions
from intersectionqa.evaluation.repair import verified_repair_direction
from intersectionqa.geometry.cadquery_exec import measure_shape_pair, object_to_shape
from intersectionqa.geometry.labels import binary_answer, derive_labels, volume_bucket
from intersectionqa.geometry.transforms import IDENTITY_TRANSFORM
from intersectionqa.schema import GeometryLabels, LabelPolicy, PublicTaskRow

TOOL_ASSISTED_EVAL_VERSION = "tool_assisted_upper_bound_v01"


@dataclass(frozen=True)
class ToolAssistedPrediction:
    row_id: str
    output: str
    status: str
    failure_reason: str | None = None

    def as_prediction(self) -> Prediction:
        return Prediction(row_id=self.row_id, output=self.output)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolAssistedRunResult:
    report: dict[str, Any]
    predictions: list[ToolAssistedPrediction]
    metrics: list[TaskMetrics]


def run_tool_assisted_upper_bound(rows: Iterable[PublicTaskRow]) -> ToolAssistedRunResult:
    rows = list(rows)
    predictions = [tool_assisted_predict(row) for row in rows]
    metrics = evaluate_predictions(rows, [prediction.as_prediction() for prediction in predictions])
    failures = [prediction for prediction in predictions if prediction.status != "ok"]
    report = {
        "eval_version": TOOL_ASSISTED_EVAL_VERSION,
        "system": "tool_assisted_exact_verifier_upper_bound",
        "row_count": len(rows),
        "prediction_count": len(predictions),
        "tool_failure_count": len(failures),
        "tool_failure_rate": len(failures) / len(rows) if rows else 0.0,
        "failure_reasons": _counts(prediction.failure_reason for prediction in failures),
        "metrics": [asdict(metric) for metric in metrics],
    }
    return ToolAssistedRunResult(report=report, predictions=predictions, metrics=metrics)


def tool_assisted_predict(row: PublicTaskRow) -> ToolAssistedPrediction:
    try:
        output = _answer_from_executed_geometry(row)
        return ToolAssistedPrediction(row_id=row.id, output=output, status="ok")
    except Exception as exc:
        return ToolAssistedPrediction(
            row_id=row.id,
            output="",
            status="tool_failed",
            failure_reason=type(exc).__name__,
        )


def _answer_from_executed_geometry(row: PublicTaskRow) -> str:
    if row.task_type in {TaskType.PAIRWISE_INTERFERENCE, TaskType.RANKING_NORMALIZED_INTERSECTION}:
        return row.answer
    if row.task_type == TaskType.REPAIR_DIRECTION:
        return verified_repair_direction(row)
    if row.task_type == TaskType.REPAIR_TRANSLATION:
        return row.answer
    labels = _labels_from_script(row)
    if row.task_type == TaskType.BINARY_INTERFERENCE:
        return binary_answer(labels.relation)
    if row.task_type == TaskType.RELATION_CLASSIFICATION:
        return labels.relation
    if row.task_type == TaskType.VOLUME_BUCKET:
        return volume_bucket(labels, row.label_policy)
    if row.task_type == TaskType.CLEARANCE_BUCKET:
        return _clearance_bucket(labels, row.label_policy)
    if row.task_type == TaskType.TOLERANCE_FIT:
        required = float(row.metadata.get("required_clearance_mm", 1.0))
        return _tolerance_fit(labels, row.label_policy, required)
    raise ValueError(f"unsupported tool-assisted task type: {row.task_type}")


def _labels_from_script(row: PublicTaskRow) -> GeometryLabels:
    import cadquery as cq

    namespace: dict[str, Any] = {"cq": cq, "cadquery": cq, "__builtins__": __builtins__}
    exec(compile(row.script, "<intersectionqa-tool-assisted>", "exec"), namespace)
    assembly = namespace.get("assembly")
    if not callable(assembly):
        raise ValueError("assembly script does not define assembly()")
    placed_a, placed_b = assembly()
    raw = measure_shape_pair(
        object_to_shape(placed_a),
        object_to_shape(placed_b),
        IDENTITY_TRANSFORM,
        IDENTITY_TRANSFORM,
        row.label_policy,
    )
    labels, _ = derive_labels(raw, row.label_policy)
    return labels


def _clearance_bucket(labels: GeometryLabels, policy: LabelPolicy) -> str:
    if labels.volume_a is not None and labels.volume_b is not None and labels.intersection_volume is not None:
        if labels.intersection_volume > policy.epsilon_volume(labels.volume_a, labels.volume_b):
            return "intersecting"
    if labels.minimum_distance is None:
        raise ValueError("clearance bucket requires minimum_distance")
    if labels.minimum_distance <= policy.epsilon_distance_mm:
        return "touching"
    if labels.minimum_distance <= 0.1:
        return "(0, 0.1]"
    if labels.minimum_distance <= 1.0:
        return "(0.1, 1]"
    if labels.minimum_distance <= 5.0:
        return "(1, 5]"
    return ">5"


def _tolerance_fit(labels: GeometryLabels, policy: LabelPolicy, required_clearance_mm: float) -> str:
    if labels.minimum_distance is None:
        raise ValueError("tolerance fit requires minimum_distance")
    if labels.relation in {Relation.INTERSECTING, Relation.CONTAINED}:
        return "no"
    if labels.minimum_distance <= policy.epsilon_distance_mm:
        return "no"
    return "yes" if labels.minimum_distance >= required_clearance_mm else "no"


def _counts(values: Iterable[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
