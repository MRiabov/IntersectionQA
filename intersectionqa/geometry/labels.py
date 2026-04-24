"""Official label derivation from stored geometry fields."""

from __future__ import annotations

import math
from dataclasses import dataclass

from intersectionqa.schema import Diagnostics, GeometryLabels, LabelPolicy, Relation

VOLUME_BUCKETS = [
    "0",
    "(0, 0.01]",
    "(0.01, 0.05]",
    "(0.05, 0.20]",
    "(0.20, 0.50]",
    ">0.50",
]


@dataclass(frozen=True)
class RawGeometry:
    volume_a: float | None
    volume_b: float | None
    intersection_volume: float | None
    minimum_distance: float | None
    contains_a_in_b: bool | None = False
    contains_b_in_a: bool | None = False
    aabb_overlap: bool | None = None
    boolean_status: str = "ok"
    distance_status: str = "ok"


def _positive_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0.0


def _nonnegative_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value >= 0.0


def normalized_intersection(
    volume_a: float | None, volume_b: float | None, intersection_volume: float | None
) -> float | None:
    if not (_positive_finite(volume_a) and _positive_finite(volume_b)):
        return None
    if intersection_volume is None or not math.isfinite(intersection_volume):
        return None
    return intersection_volume / min(float(volume_a), float(volume_b))


def derive_relation(raw: RawGeometry, policy: LabelPolicy) -> Relation:
    if not (_positive_finite(raw.volume_a) and _positive_finite(raw.volume_b)):
        return "invalid"
    if not _nonnegative_finite(raw.intersection_volume):
        return "invalid"
    if raw.boolean_status not in {"ok", "skipped_aabb_disjoint"}:
        return "invalid"

    epsilon_volume = policy.epsilon_volume(float(raw.volume_a), float(raw.volume_b))
    exact_overlap = float(raw.intersection_volume) > epsilon_volume

    if raw.contains_a_in_b or raw.contains_b_in_a:
        return "contained" if exact_overlap else "invalid"
    if exact_overlap:
        return "intersecting"
    if raw.distance_status != "ok" or not _nonnegative_finite(raw.minimum_distance):
        return "invalid"
    if float(raw.minimum_distance) <= policy.epsilon_distance_mm:
        return "touching"
    if float(raw.minimum_distance) <= policy.near_miss_threshold_mm:
        return "near_miss"
    return "disjoint"


def derive_labels(raw: RawGeometry, policy: LabelPolicy) -> tuple[GeometryLabels, Diagnostics]:
    relation = derive_relation(raw, policy)
    norm = normalized_intersection(raw.volume_a, raw.volume_b, raw.intersection_volume)
    exact_overlap = None
    if _positive_finite(raw.volume_a) and _positive_finite(raw.volume_b):
        if raw.intersection_volume is not None and math.isfinite(raw.intersection_volume):
            exact_overlap = raw.intersection_volume > policy.epsilon_volume(raw.volume_a, raw.volume_b)

    failure_reason = None if relation != "invalid" else "unknown_error"
    label_status = "ok" if relation != "invalid" else "invalid"
    distance_status = raw.distance_status
    if relation in {"intersecting", "contained"} and distance_status == "ok":
        distance_status = "skipped_positive_overlap"

    labels = GeometryLabels(
        volume_a=raw.volume_a,
        volume_b=raw.volume_b,
        intersection_volume=raw.intersection_volume,
        normalized_intersection=norm,
        minimum_distance=raw.minimum_distance,
        relation=relation,
        contained=(raw.contains_a_in_b or raw.contains_b_in_a)
        if raw.contains_a_in_b is not None or raw.contains_b_in_a is not None
        else None,
        contains_a_in_b=raw.contains_a_in_b,
        contains_b_in_a=raw.contains_b_in_a,
    )
    diagnostics = Diagnostics(
        aabb_overlap=raw.aabb_overlap,
        exact_overlap=exact_overlap,
        boolean_status=raw.boolean_status,  # type: ignore[arg-type]
        distance_status=distance_status,  # type: ignore[arg-type]
        label_status=label_status,
        failure_reason=failure_reason,  # type: ignore[arg-type]
    )
    return labels, diagnostics


def binary_answer(relation: Relation) -> str:
    if relation in {"intersecting", "contained"}:
        return "yes"
    if relation in {"disjoint", "touching", "near_miss"}:
        return "no"
    raise ValueError("invalid relation is excluded from binary_interference")


def volume_bucket(labels: GeometryLabels, policy: LabelPolicy) -> str:
    if labels.volume_a is None or labels.volume_b is None:
        raise ValueError("volume bucket requires valid volumes")
    if labels.intersection_volume is None or labels.normalized_intersection is None:
        raise ValueError("volume bucket requires intersection fields")
    epsilon_volume = policy.epsilon_volume(labels.volume_a, labels.volume_b)
    if labels.intersection_volume <= epsilon_volume:
        return "0"
    ratio = labels.normalized_intersection
    if ratio <= 0.01:
        return "(0, 0.01]"
    if ratio <= 0.05:
        return "(0.01, 0.05]"
    if ratio <= 0.20:
        return "(0.05, 0.20]"
    if ratio <= 0.50:
        return "(0.20, 0.50]"
    return ">0.50"


def validate_label_consistency(labels: GeometryLabels, diagnostics: Diagnostics, policy: LabelPolicy) -> None:
    if diagnostics.label_status != "ok":
        if labels.relation != "invalid":
            raise ValueError("invalid diagnostics must use invalid relation")
        return
    if labels.volume_a is None or labels.volume_a <= 0:
        raise ValueError("ok labels require positive volume_a")
    if labels.volume_b is None or labels.volume_b <= 0:
        raise ValueError("ok labels require positive volume_b")
    if labels.intersection_volume is None or labels.intersection_volume < 0:
        raise ValueError("ok labels require non-negative intersection_volume")
    expected_overlap = labels.intersection_volume > policy.epsilon_volume(labels.volume_a, labels.volume_b)
    if diagnostics.exact_overlap != expected_overlap:
        raise ValueError("exact_overlap does not match label policy")
    if labels.relation in {"intersecting", "contained"} and not expected_overlap:
        raise ValueError("positive-overlap relation requires policy-positive intersection")
    if labels.relation == "contained":
        if not (labels.contains_a_in_b or labels.contains_b_in_a):
            raise ValueError("contained relation requires a containment flag")
        smaller = min(labels.volume_a, labels.volume_b)
        tolerance = policy.epsilon_volume(labels.volume_a, labels.volume_b)
        if abs(labels.intersection_volume - smaller) > tolerance:
            raise ValueError("contained intersection volume must match smaller solid volume")
    if labels.relation == "touching" and labels.minimum_distance is not None:
        if labels.minimum_distance > policy.epsilon_distance_mm:
            raise ValueError("touching distance exceeds epsilon")
    if labels.relation == "near_miss" and labels.minimum_distance is not None:
        if not policy.epsilon_distance_mm < labels.minimum_distance <= policy.near_miss_threshold_mm:
            raise ValueError("near_miss distance is outside policy interval")
    if labels.relation == "disjoint" and labels.minimum_distance is not None:
        if labels.minimum_distance <= policy.near_miss_threshold_mm:
            raise ValueError("disjoint distance is not above near-miss threshold")
