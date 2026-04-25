"""Pydantic models for the v0.1 public and internal records."""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intersectionqa.enums import (
    AuditStatus,
    BooleanStatus,
    DistanceStatus,
    FailureReason,
    FailureStage,
    LabelStatus,
    Relation,
    RotationOrder,
    Split,
    TaskType,
)

HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
TRANSLATION_VECTOR_ANSWER_RE = re.compile(
    r"^dx=(-?[0-9]+(?:\.[0-9]+)?), dy=(-?[0-9]+(?:\.[0-9]+)?), dz=(-?[0-9]+(?:\.[0-9]+)?)$"
)
EDIT_PROGRAM_ANSWER_RE = re.compile(
    r"^object_b = object_b\.translate\(\((-?[0-9]+(?:\.[0-9]+)?), (-?[0-9]+(?:\.[0-9]+)?), (-?[0-9]+(?:\.[0-9]+)?)\)\)$"
)

DIFFICULTY_TAGS = {
    "axis_aligned",
    "rotated",
    "cadevolve_simple",
    "cadevolve_compound",
    "compound_boolean",
    "cavity_targeted",
    "contact_vs_interference",
    "near_boundary",
    "tiny_overlap",
    "near_miss",
    "clearance_bucket_targeted",
    "tiny_clearance",
    "mid_clearance",
    "volume_bucket_targeted",
    "small_overlap",
    "medium_overlap",
    "deep_overlap",
    "aabb_exact_disagreement",
    "contained",
    "primitive_fixture",
    "broad_placement",
    "invalid",
}


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("numeric values must be finite")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Transform(StrictModel):
    translation: tuple[float, float, float]
    rotation_xyz_deg: tuple[float, float, float]
    rotation_order: RotationOrder = RotationOrder.XYZ

    @field_validator("translation", "rotation_xyz_deg")
    @classmethod
    def finite_triplet(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        for item in value:
            _finite(float(item))
        return tuple(float(item) for item in value)


class BoundingBox(StrictModel):
    min: tuple[float, float, float]
    max: tuple[float, float, float]

    @model_validator(mode="after")
    def valid_extents(self) -> "BoundingBox":
        for lo, hi in zip(self.min, self.max, strict=True):
            _finite(float(lo))
            _finite(float(hi))
            if hi < lo:
                raise ValueError("bbox max must be >= min on every axis")
        return self


class LabelPolicy(StrictModel):
    epsilon_volume_ratio: float = 1e-6
    epsilon_distance_mm: float = 1e-4
    near_miss_threshold_mm: float = 1.0

    @model_validator(mode="after")
    def valid_thresholds(self) -> "LabelPolicy":
        if self.epsilon_volume_ratio <= 0:
            raise ValueError("epsilon_volume_ratio must be positive")
        if self.epsilon_distance_mm < 0:
            raise ValueError("epsilon_distance_mm must be non-negative")
        if self.near_miss_threshold_mm <= self.epsilon_distance_mm:
            raise ValueError("near_miss_threshold_mm must exceed epsilon_distance_mm")
        return self

    def epsilon_volume(self, volume_a: float, volume_b: float) -> float:
        return self.epsilon_volume_ratio * min(volume_a, volume_b)


class ArtifactIds(StrictModel):
    debug_step_id: str | None = None
    debug_mesh_id: str | None = None
    render_iso_id: str | None = None
    render_orthographic_ids: list[str] | None = None
    overlap_render_id: str | None = None
    failure_reproducer_id: str | None = None


class Hashes(StrictModel):
    source_code_hash: str | None
    object_hash: str | None
    transform_hash: str | None
    geometry_hash: str | None
    config_hash: str | None
    prompt_hash: str | None

    @field_validator(
        "source_code_hash",
        "object_hash",
        "transform_hash",
        "geometry_hash",
        "config_hash",
        "prompt_hash",
    )
    @classmethod
    def valid_hash(cls, value: str | None) -> str | None:
        if value is not None and not HASH_RE.match(value):
            raise ValueError("hashes must use sha256:<64 lowercase hex>")
        return value


class GeometryLabels(StrictModel):
    volume_a: float | None
    volume_b: float | None
    intersection_volume: float | None
    normalized_intersection: float | None
    minimum_distance: float | None
    relation: Relation
    contained: bool | None
    contains_a_in_b: bool | None
    contains_b_in_a: bool | None

    @field_validator(
        "volume_a",
        "volume_b",
        "intersection_volume",
        "normalized_intersection",
        "minimum_distance",
    )
    @classmethod
    def finite_or_null(cls, value: float | None) -> float | None:
        if value is not None:
            _finite(float(value))
        return value


class Diagnostics(StrictModel):
    aabb_overlap: bool | None
    exact_overlap: bool | None
    boolean_status: BooleanStatus
    distance_status: DistanceStatus
    label_status: LabelStatus
    failure_reason: FailureReason | None

    @model_validator(mode="after")
    def invalid_has_reason(self) -> "Diagnostics":
        if self.label_status == LabelStatus.OK and self.failure_reason is not None:
            raise ValueError("ok labels must not have failure_reason")
        if self.label_status == LabelStatus.INVALID and self.failure_reason is None:
            raise ValueError("invalid labels require failure_reason")
        return self


class SourceObjectRecord(StrictModel):
    object_id: str
    source: str
    source_id: str
    generator_id: str | None
    source_path: str | None
    source_license: str | None
    object_name: str
    normalized_code: str
    object_function_name: str
    cadquery_ops: list[str]
    topology_tags: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
    hashes: Hashes


class ObjectValidationRecord(StrictModel):
    object_id: str
    valid: bool
    volume: float | None
    bbox: BoundingBox | None
    label_status: LabelStatus
    failure_reason: FailureReason | None
    cadquery_version: str | None
    ocp_version: str | None
    validated_at_version: str
    hashes: Hashes

    @model_validator(mode="after")
    def validity_matches_status(self) -> "ObjectValidationRecord":
        if self.valid:
            if self.label_status != LabelStatus.OK:
                raise ValueError("valid objects require ok label_status")
            if self.failure_reason is not None:
                raise ValueError("valid objects must not have failure_reason")
            if self.volume is None or self.volume <= 0:
                raise ValueError("valid objects require positive volume")
            if self.bbox is None:
                raise ValueError("valid objects require bbox")
        else:
            if self.label_status != LabelStatus.INVALID:
                raise ValueError("invalid objects require invalid label_status")
            if self.failure_reason is None:
                raise ValueError("invalid objects require failure_reason")
        return self


class GeometryRecord(StrictModel):
    geometry_id: str
    source: str
    object_a_id: str
    object_b_id: str
    base_object_pair_id: str
    assembly_group_id: str
    counterfactual_group_id: str | None
    variant_id: str | None
    changed_parameter: str | None
    changed_value: Any
    transform_a: Transform
    transform_b: Transform
    assembly_script: str
    labels: GeometryLabels
    diagnostics: Diagnostics
    difficulty_tags: list[str]
    label_policy: LabelPolicy
    hashes: Hashes
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("difficulty_tags")
    @classmethod
    def canonical_tags(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - DIFFICULTY_TAGS)
        if unknown:
            raise ValueError(f"unknown difficulty tags: {unknown}")
        return sorted(dict.fromkeys(value))


class PublicTaskRow(StrictModel):
    id: str
    dataset_version: str
    split: Split
    task_type: TaskType
    prompt: str
    answer: str
    script: str
    geometry_ids: list[str]
    source: str
    generator_id: str | None
    base_object_pair_id: str
    assembly_group_id: str
    counterfactual_group_id: str | None
    variant_id: str | None
    changed_parameter: str | None
    changed_value: Any
    labels: GeometryLabels
    diagnostics: Diagnostics
    difficulty_tags: list[str]
    label_policy: LabelPolicy
    hashes: Hashes
    metadata: dict[str, Any]

    @model_validator(mode="after")
    def public_contracts(self) -> "PublicTaskRow":
        if not self.geometry_ids:
            raise ValueError("public rows require at least one geometry_id")
        if self.diagnostics.label_status != LabelStatus.OK:
            raise ValueError("normal MVP public rows require ok label_status")
        if self.task_type == TaskType.BINARY_INTERFERENCE:
            expected = "yes" if self.labels.relation in {Relation.INTERSECTING, Relation.CONTAINED} else "no"
            if self.labels.relation == Relation.INVALID or self.answer != expected:
                raise ValueError("binary answer does not match relation")
        if self.task_type == TaskType.RELATION_CLASSIFICATION and self.answer != self.labels.relation:
            raise ValueError("relation answer must equal labels.relation")
        if self.task_type == TaskType.VOLUME_BUCKET:
            expected_bucket = _expected_volume_bucket(self.labels, self.label_policy)
            if self.answer != expected_bucket:
                raise ValueError("volume_bucket answer does not match labels and policy")
        if self.task_type == TaskType.CLEARANCE_BUCKET:
            expected_bucket = _expected_clearance_bucket(self.labels, self.label_policy)
            if self.answer != expected_bucket:
                raise ValueError("clearance_bucket answer does not match labels and policy")
        if self.task_type == TaskType.TOLERANCE_FIT:
            required_clearance = float(self.metadata.get("required_clearance_mm", 1.0))
            if self.answer != _expected_tolerance_fit(self.labels, self.label_policy, required_clearance):
                raise ValueError("tolerance_fit answer does not match labels and policy")
        if self.task_type == TaskType.PAIRWISE_INTERFERENCE:
            if self.answer != _expected_pairwise_answer(self.metadata):
                raise ValueError("pairwise_interference answer does not match variant metadata")
        if self.task_type == TaskType.RANKING_NORMALIZED_INTERSECTION:
            if self.answer != _expected_ranking_answer(self.metadata):
                raise ValueError("ranking answer does not match variant metadata")
        if self.task_type == TaskType.REPAIR_DIRECTION:
            if self.labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}:
                raise ValueError("repair_direction rows require positive-overlap relation")
            if self.answer != _expected_repair_direction(self.metadata):
                raise ValueError("repair_direction answer does not match repair metadata")
        if self.task_type == TaskType.REPAIR_TRANSLATION:
            if self.labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}:
                raise ValueError("repair_translation rows require positive-overlap relation")
            if self.answer != _expected_repair_translation(self.metadata):
                raise ValueError("repair_translation answer does not match repair metadata")
        if self.task_type in {TaskType.AXIS_ALIGNED_REPAIR, TaskType.TARGET_CLEARANCE_REPAIR}:
            if self.labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}:
                raise ValueError(f"{self.task_type} rows require positive-overlap relation")
            if self.answer != _expected_axis_aligned_repair(self.metadata):
                raise ValueError(f"{self.task_type} answer does not match edit metadata")
        if self.task_type in {TaskType.AXIS_ALIGNED_REPAIR_VECTOR, TaskType.AXIS_ALIGNED_REPAIR_PROGRAM}:
            if self.labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}:
                raise ValueError(f"{self.task_type} rows require positive-overlap relation")
            if self.answer != _expected_axis_aligned_repair_vector_or_program(self.metadata, self.task_type):
                raise ValueError(f"{self.task_type} answer does not match edit metadata")
        if self.task_type in {TaskType.TARGET_CLEARANCE_MOVE, TaskType.TARGET_CONTACT_MOVE}:
            if self.labels.relation not in {Relation.DISJOINT, Relation.NEAR_MISS}:
                raise ValueError(f"{self.task_type} rows require non-intersecting relation")
            if self.answer != _expected_target_clearance_move(self.metadata):
                raise ValueError(f"{self.task_type} answer does not match edit metadata")
        if self.task_type == TaskType.CENTROID_DISTANCE_MOVE:
            if self.labels.relation not in {Relation.DISJOINT, Relation.NEAR_MISS, Relation.TOUCHING}:
                raise ValueError("centroid_distance_move rows require non-intersecting relation")
            if self.answer != _expected_centroid_distance_move(self.metadata):
                raise ValueError("centroid_distance_move answer does not match edit metadata")
        if self.task_type == TaskType.EDIT_CANDIDATE_SELECTION:
            if self.answer != _expected_candidate_selection(self.metadata):
                raise ValueError("edit_candidate_selection answer does not match candidate metadata")
        if self.task_type == TaskType.EDIT_CANDIDATE_RANKING:
            if self.answer != _expected_candidate_ranking(self.metadata):
                raise ValueError("edit_candidate_ranking answer does not match candidate metadata")
        if self.hashes.prompt_hash is None:
            raise ValueError("public task rows require prompt_hash")
        required = [
            self.hashes.source_code_hash,
            self.hashes.object_hash,
            self.hashes.transform_hash,
            self.hashes.geometry_hash,
            self.hashes.config_hash,
        ]
        if any(item is None for item in required):
            raise ValueError("public task rows require non-null hashes")
        return self

    @field_validator("difficulty_tags")
    @classmethod
    def canonical_tags(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - DIFFICULTY_TAGS)
        if unknown:
            raise ValueError(f"unknown difficulty tags: {unknown}")
        return sorted(dict.fromkeys(value))


def _expected_volume_bucket(labels: GeometryLabels, policy: LabelPolicy) -> str:
    if labels.volume_a is None or labels.volume_b is None:
        raise ValueError("volume bucket requires valid volumes")
    if labels.intersection_volume is None or labels.normalized_intersection is None:
        raise ValueError("volume bucket requires valid intersection fields")
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


def _expected_clearance_bucket(labels: GeometryLabels, policy: LabelPolicy) -> str:
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


def _expected_tolerance_fit(
    labels: GeometryLabels,
    policy: LabelPolicy,
    required_clearance_mm: float,
) -> str:
    if labels.minimum_distance is None:
        raise ValueError("tolerance fit requires minimum_distance")
    if labels.minimum_distance <= policy.epsilon_distance_mm:
        return "no"
    return "yes" if labels.minimum_distance >= required_clearance_mm else "no"


def _expected_pairwise_answer(metadata: dict[str, Any]) -> str:
    variant_labels = metadata.get("variant_labels")
    if not isinstance(variant_labels, dict):
        raise ValueError("pairwise rows require variant_labels metadata")
    a = _variant_interferes(variant_labels, "A")
    b = _variant_interferes(variant_labels, "B")
    if a and b:
        return "both"
    if a:
        return "A"
    if b:
        return "B"
    return "neither"


def _variant_interferes(variant_labels: dict[str, Any], letter: str) -> bool:
    value = variant_labels.get(letter)
    if not isinstance(value, dict) or not isinstance(value.get("interferes"), bool):
        raise ValueError("pairwise variant metadata requires boolean interferes fields")
    return value["interferes"]


def _expected_ranking_answer(metadata: dict[str, Any]) -> str:
    values = metadata.get("variant_values")
    if not isinstance(values, dict) or not 3 <= len(values) <= 5:
        raise ValueError("ranking rows require 3-5 variant_values metadata entries")
    ordered = sorted(
        values.items(),
        key=lambda item: (-float(item[1]["normalized_intersection"]), item[0]),
    )
    return "".join(letter for letter, _ in ordered)


def _expected_repair_direction(metadata: dict[str, Any]) -> str:
    if metadata.get("repair_policy") != "conservative_aabb_separating_translation_v01":
        raise ValueError("repair_direction rows require the conservative AABB repair policy")
    selected = metadata.get("selected_direction")
    allowed = {"+x", "-x", "+y", "-y", "+z", "-z"}
    if selected not in allowed:
        raise ValueError("repair_direction selected_direction is invalid")
    if metadata.get("movable_object") != "object_b" or metadata.get("fixed_object") != "object_a":
        raise ValueError("repair_direction rows require object_b movable and object_a fixed")
    candidate_labels = metadata.get("candidate_direction_labels")
    tie_break_order = ["+x", "-x", "+y", "-y", "+z", "-z"]
    if candidate_labels != tie_break_order:
        raise ValueError("repair_direction candidate labels must document tie-break order")
    selected_magnitude = _finite_nonnegative_metadata_float(metadata, "selected_magnitude_mm")
    selected_vector = _metadata_vector(metadata, "selected_translation_vector_mm")
    if not math.isclose(selected_magnitude, _vector_magnitude_l1(selected_vector), abs_tol=1e-9):
        raise ValueError("repair_direction selected magnitude must match translation vector")
    _validate_repair_vector_direction(selected, selected_vector)
    candidate_moves = metadata.get("candidate_moves")
    if not isinstance(candidate_moves, list) or len(candidate_moves) != 6:
        raise ValueError("repair_direction rows require six candidate moves")
    candidate_by_direction: dict[str, dict[str, Any]] = {}
    for item in candidate_moves:
        if not isinstance(item, dict):
            raise ValueError("repair_direction candidate moves must be objects")
        direction = item.get("direction")
        if direction not in allowed or direction in candidate_by_direction:
            raise ValueError("repair_direction candidate move direction is invalid")
        magnitude = _finite_nonnegative_metadata_float(item, "magnitude_mm")
        vector = _metadata_vector(item, "translation_vector_mm")
        if not math.isclose(magnitude, _vector_magnitude_l1(vector), abs_tol=1e-9):
            raise ValueError("repair_direction candidate magnitude must match translation vector")
        _validate_repair_vector_direction(direction, vector)
        candidate_by_direction[direction] = item
    if set(candidate_by_direction) != allowed:
        raise ValueError("repair_direction candidate moves must cover all six directions")
    expected = min(
        candidate_by_direction.items(),
        key=lambda item: (float(item[1]["magnitude_mm"]), tie_break_order.index(item[0])),
    )[0]
    if selected != expected:
        raise ValueError("repair_direction selected direction does not match tie-break policy")
    expected_candidate = candidate_by_direction[selected]
    expected_vector = _metadata_vector(expected_candidate, "translation_vector_mm")
    if selected_vector != expected_vector:
        raise ValueError("repair_direction selected vector must match selected candidate")
    return selected


def _expected_repair_translation(metadata: dict[str, Any]) -> str:
    direction = _expected_repair_direction(metadata)
    magnitude = _finite_nonnegative_metadata_float(metadata, "selected_magnitude_mm")
    return f"{direction} {magnitude:.6f}"


def _expected_axis_aligned_repair(metadata: dict[str, Any]) -> str:
    if metadata.get("edit_policy") != "exact_axis_aligned_cardinal_search_v01":
        raise ValueError("axis-aligned repair rows require exact cardinal search policy")
    if metadata.get("movable_object") != "object_b" or metadata.get("fixed_object") != "object_a":
        raise ValueError("axis-aligned repair rows require object_b movable and object_a fixed")
    selected = metadata.get("selected_direction")
    allowed = {"+x", "-x", "+y", "-y", "+z", "-z"}
    if selected not in allowed:
        raise ValueError("axis-aligned repair selected_direction is invalid")
    exact_magnitude = _finite_nonnegative_metadata_float(metadata, "selected_exact_magnitude_mm")
    label_magnitude = _finite_nonnegative_metadata_float(metadata, "selected_magnitude_mm")
    if label_magnitude + 1e-9 < exact_magnitude:
        raise ValueError("axis-aligned repair label magnitude must not undershoot exact magnitude")
    selected_vector = _metadata_vector(metadata, "selected_translation_vector_mm")
    if not math.isclose(label_magnitude, _vector_magnitude_l1(selected_vector), abs_tol=1e-9):
        raise ValueError("axis-aligned repair selected magnitude must match translation vector")
    _validate_repair_vector_direction(str(selected), selected_vector)
    structured = metadata.get("structured_answer")
    if not isinstance(structured, dict):
        raise ValueError("axis-aligned repair rows require structured_answer")
    if structured.get("direction") != selected:
        raise ValueError("structured answer direction must match selected direction")
    if not math.isclose(float(structured.get("distance_mm")), label_magnitude, abs_tol=1e-9):
        raise ValueError("structured answer distance must match selected magnitude")
    verification = metadata.get("verification")
    if not isinstance(verification, dict) or verification.get("satisfies_target") is not True:
        raise ValueError("axis-aligned repair verification must satisfy target")
    target = metadata.get("target")
    if not isinstance(target, dict) or target.get("type") not in {"non_intersection", "clearance"}:
        raise ValueError("axis-aligned repair rows require target metadata")
    allowed_edit = metadata.get("allowed_edit")
    if not isinstance(allowed_edit, dict) or allowed_edit.get("directions") != [
        "+x",
        "-x",
        "+y",
        "-y",
        "+z",
        "-z",
    ]:
        raise ValueError("axis-aligned repair rows require cardinal allowed directions")
    candidates = metadata.get("candidate_moves")
    if not isinstance(candidates, list) or len(candidates) != 6:
        raise ValueError("axis-aligned repair rows require six candidate moves")
    selected_candidate = None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("axis-aligned repair candidates must be objects")
        direction = candidate.get("direction")
        if direction not in allowed:
            raise ValueError("axis-aligned repair candidate direction is invalid")
        candidate_exact = _finite_nonnegative_metadata_float(candidate, "exact_magnitude_mm")
        candidate_label = _finite_nonnegative_metadata_float(candidate, "label_magnitude_mm")
        if candidate_label + 1e-9 < candidate_exact:
            raise ValueError("axis-aligned repair candidate label must not undershoot exact magnitude")
        vector = _metadata_vector(candidate, "translation_vector_mm")
        if not math.isclose(candidate_label, _vector_magnitude_l1(vector), abs_tol=1e-9):
            raise ValueError("axis-aligned repair candidate magnitude must match translation vector")
        _validate_repair_vector_direction(str(direction), vector)
        if direction == selected:
            selected_candidate = candidate
    if selected_candidate is None:
        raise ValueError("axis-aligned repair selected candidate missing")
    expected_selected = min(
        candidates,
        key=lambda item: (
            float(item["exact_magnitude_mm"]),
            ["+x", "-x", "+y", "-y", "+z", "-z"].index(str(item["direction"])),
        ),
    )
    if expected_selected.get("direction") != selected:
        raise ValueError("axis-aligned repair selected direction does not match exact policy")
    return f"direction={selected}, distance_mm={label_magnitude:.1f}"


def _expected_axis_aligned_repair_vector_or_program(
    metadata: dict[str, Any],
    task_type: TaskType,
) -> str:
    _expected_axis_aligned_repair(metadata)
    vector = _metadata_vector(metadata, "selected_translation_vector_mm")
    structured = metadata.get("structured_answer")
    if not isinstance(structured, dict):
        raise ValueError("axis-aligned repair vector/program rows require structured_answer")
    structured_vector = _metadata_vector(structured, "translation")
    if any(not math.isclose(vector[index], structured_vector[index], abs_tol=1e-9) for index in range(3)):
        raise ValueError("structured answer translation must match selected vector")
    dx, dy, dz = vector
    if task_type == TaskType.AXIS_ALIGNED_REPAIR_VECTOR:
        if metadata.get("output_format") != "translation_vector":
            raise ValueError("axis_aligned_repair_vector rows require translation_vector output_format")
        return f"dx={dx:.1f}, dy={dy:.1f}, dz={dz:.1f}"
    if metadata.get("output_format") != "edit_program":
        raise ValueError("axis_aligned_repair_program rows require edit_program output_format")
    if metadata.get("edit_program_language") != "cadquery_python":
        raise ValueError("axis_aligned_repair_program rows require cadquery_python language")
    return f"object_b = object_b.translate(({dx:.1f}, {dy:.1f}, {dz:.1f}))"


def _expected_target_clearance_move(metadata: dict[str, Any]) -> str:
    if metadata.get("edit_policy") != "exact_axis_aligned_cardinal_search_v01":
        raise ValueError("target_clearance_move rows require exact cardinal search policy")
    if metadata.get("movable_object") != "object_b" or metadata.get("fixed_object") != "object_a":
        raise ValueError("target_clearance_move rows require object_b movable and object_a fixed")
    allowed_edit = metadata.get("allowed_edit")
    if not isinstance(allowed_edit, dict):
        raise ValueError("target_clearance_move rows require allowed_edit")
    direction = allowed_edit.get("direction")
    if direction not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise ValueError("target_clearance_move direction is invalid")
    if allowed_edit.get("edit_type") != "signed_translation_distance":
        raise ValueError("target_clearance_move rows require signed distance edit type")
    signed_distance = _finite_metadata_float(metadata, "selected_signed_distance_mm")
    signed_exact = _finite_metadata_float(metadata, "selected_signed_exact_distance_mm")
    if abs(signed_distance - signed_exact) > 0.2:
        raise ValueError("target_clearance_move rounded distance is too far from exact distance")
    vector = _metadata_vector(metadata, "selected_translation_vector_mm")
    if not math.isclose(abs(signed_distance), _vector_magnitude_l1(vector), abs_tol=1e-9):
        raise ValueError("target_clearance_move signed distance must match vector magnitude")
    _validate_signed_direction_vector(str(direction), signed_distance, vector)
    structured = metadata.get("structured_answer")
    if not isinstance(structured, dict):
        raise ValueError("target_clearance_move rows require structured_answer")
    if not math.isclose(float(structured.get("distance_mm")), signed_distance, abs_tol=1e-9):
        raise ValueError("target_clearance_move structured answer must match selected distance")
    verification = metadata.get("verification")
    if not isinstance(verification, dict) or verification.get("satisfies_target") is not True:
        raise ValueError("target_clearance_move verification must satisfy target")
    target = metadata.get("target")
    if not isinstance(target, dict) or target.get("type") != "clearance":
        raise ValueError("target_clearance_move rows require clearance target")
    return f"distance_mm={signed_distance:.1f}"


def _expected_centroid_distance_move(metadata: dict[str, Any]) -> str:
    if metadata.get("edit_policy") != "exact_centroid_direction_move_v01":
        raise ValueError("centroid_distance_move rows require exact centroid-direction policy")
    if metadata.get("movable_object") != "object_b" or metadata.get("fixed_object") != "object_a":
        raise ValueError("centroid_distance_move rows require object_b movable and object_a fixed")
    allowed_edit = metadata.get("allowed_edit")
    if not isinstance(allowed_edit, dict):
        raise ValueError("centroid_distance_move rows require allowed_edit")
    if allowed_edit.get("edit_type") != "signed_centroid_direction_distance":
        raise ValueError("centroid_distance_move rows require signed centroid-direction edit type")
    direction = _metadata_vector(allowed_edit, "direction_vector")
    if not math.isclose(_vector_magnitude_l2(direction), 1.0, abs_tol=1e-9):
        raise ValueError("centroid_distance_move direction_vector must be unit length")
    signed_distance = _finite_metadata_float(metadata, "selected_signed_distance_mm")
    signed_exact = _finite_metadata_float(metadata, "selected_signed_exact_distance_mm")
    if abs(signed_distance - signed_exact) > 0.2:
        raise ValueError("centroid_distance_move rounded distance is too far from exact distance")
    vector = _metadata_vector(metadata, "selected_translation_vector_mm")
    if not math.isclose(abs(signed_distance), _vector_magnitude_l2(vector), abs_tol=1e-9):
        raise ValueError("centroid_distance_move signed distance must match vector magnitude")
    expected_vector = tuple(component * signed_distance for component in direction)
    if any(not math.isclose(vector[index], expected_vector[index], abs_tol=1e-9) for index in range(3)):
        raise ValueError("centroid_distance_move vector must follow direction_vector and signed distance")
    structured = metadata.get("structured_answer")
    if not isinstance(structured, dict):
        raise ValueError("centroid_distance_move rows require structured_answer")
    if not math.isclose(float(structured.get("distance_mm")), signed_distance, abs_tol=1e-9):
        raise ValueError("centroid_distance_move structured answer must match selected distance")
    structured_direction = _metadata_vector(structured, "direction_vector")
    if any(not math.isclose(structured_direction[index], direction[index], abs_tol=1e-9) for index in range(3)):
        raise ValueError("centroid_distance_move structured direction must match allowed direction")
    target = metadata.get("target")
    if not isinstance(target, dict) or target.get("type") != "centroid_distance":
        raise ValueError("centroid_distance_move rows require centroid_distance target")
    _finite_metadata_float(target, "target_centroid_distance_mm")
    verification = metadata.get("verification")
    if not isinstance(verification, dict) or verification.get("satisfies_target") is not True:
        raise ValueError("centroid_distance_move verification must satisfy target")
    return f"distance_mm={signed_distance:.1f}"


def _expected_candidate_selection(metadata: dict[str, Any]) -> str:
    candidates = _candidate_edits(metadata)
    expected = _rank_candidate_edits(candidates)[0]
    stored = metadata.get("candidate_selection_answer")
    if stored != expected:
        raise ValueError("candidate selection metadata answer does not match ranked candidates")
    return expected


def _expected_candidate_ranking(metadata: dict[str, Any]) -> str:
    candidates = _candidate_edits(metadata)
    expected = "".join(_rank_candidate_edits(candidates))
    stored = metadata.get("candidate_ranking_answer")
    if stored != expected:
        raise ValueError("candidate ranking metadata answer does not match ranked candidates")
    return expected


def _candidate_edits(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if metadata.get("edit_policy") != "exact_axis_aligned_cardinal_search_v01":
        raise ValueError("candidate edit rows require exact cardinal search policy")
    target = metadata.get("target")
    if not isinstance(target, dict) or target.get("type") != "clearance":
        raise ValueError("candidate edit rows require clearance target metadata")
    candidates = metadata.get("candidate_edits")
    if not isinstance(candidates, dict) or set(candidates) != {"A", "B", "C", "D"}:
        raise ValueError("candidate edit rows require A-D candidates")
    result: dict[str, dict[str, Any]] = {}
    for label, candidate in candidates.items():
        if not isinstance(candidate, dict):
            raise ValueError("candidate edit entries must be objects")
        direction = candidate.get("direction")
        if direction not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
            raise ValueError("candidate edit direction is invalid")
        magnitude = _finite_nonnegative_metadata_float(candidate, "magnitude_mm")
        movement = _finite_nonnegative_metadata_float(candidate, "movement_magnitude")
        if not math.isclose(magnitude, movement, abs_tol=1e-9):
            raise ValueError("candidate edit movement must match magnitude")
        vector = _metadata_vector(candidate, "translation_vector_mm")
        if not math.isclose(magnitude, _vector_magnitude_l1(vector), abs_tol=1e-9):
            raise ValueError("candidate edit magnitude must match vector")
        _validate_repair_vector_direction(str(direction), vector)
        if not isinstance(candidate.get("satisfies_target"), bool):
            raise ValueError("candidate edit requires satisfies_target boolean")
        if not isinstance(candidate.get("no_interference"), bool):
            raise ValueError("candidate edit requires no_interference boolean")
        result[str(label)] = candidate
    return result


def _rank_candidate_edits(candidates: dict[str, dict[str, Any]]) -> list[str]:
    return [
        label
        for label, _ in sorted(
            candidates.items(),
            key=lambda item: (
                0 if item[1].get("satisfies_target") else 1,
                0 if item[1].get("no_interference") else 1,
                float(item[1]["movement_magnitude"]),
                (
                    float(item[1]["clearance_error_mm"])
                    if isinstance(item[1].get("clearance_error_mm"), int | float)
                    else math.inf
                ),
                item[0],
            ),
        )
    ]


def _finite_nonnegative_metadata_float(metadata: dict[str, Any], key: str) -> float:
    value = metadata.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be finite and non-negative")
    return value


def _finite_metadata_float(metadata: dict[str, Any], key: str) -> float:
    value = metadata.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _metadata_vector(metadata: dict[str, Any], key: str) -> tuple[float, float, float]:
    value = metadata.get(key)
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{key} must be a 3-vector")
    result = tuple(float(item) for item in value)
    for item in result:
        if not math.isfinite(item):
            raise ValueError(f"{key} must contain finite values")
    return result


def _vector_magnitude_l1(vector: tuple[float, float, float]) -> float:
    return sum(abs(item) for item in vector)


def _vector_magnitude_l2(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(item * item for item in vector))


def _validate_repair_vector_direction(direction: str, vector: tuple[float, float, float]) -> None:
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


def _validate_signed_direction_vector(
    direction: str,
    signed_distance: float,
    vector: tuple[float, float, float],
) -> None:
    axis_by_name = {"x": 0, "y": 1, "z": 2}
    axis = axis_by_name[direction[1]]
    for index, value in enumerate(vector):
        if index != axis and not math.isclose(value, 0.0, abs_tol=1e-9):
            raise ValueError("target_clearance_move vector must move only along its direction axis")
    direction_sign = 1.0 if direction[0] == "+" else -1.0
    expected_axis_value = direction_sign * signed_distance
    if not math.isclose(vector[axis], expected_axis_value, abs_tol=1e-9):
        raise ValueError("target_clearance_move vector sign does not match signed distance")


class FailureRecord(StrictModel):
    failure_id: str
    stage: FailureStage
    source: str | None
    source_id: str | None
    object_id: str | None
    geometry_id: str | None
    failure_reason: FailureReason
    error_summary: str
    retry_count: int = 0
    hashes: Hashes


class DatasetMetadata(StrictModel):
    dataset_name: Literal["IntersectionQA"] = "IntersectionQA"
    dataset_version: str
    created_from_commit: str
    config_hash: str
    source_manifest_hash: str
    label_policy: LabelPolicy
    splits: dict[str, "SplitFileSummary"]
    task_types: list[str]
    counts: "DatasetCounts"
    cadquery_version: str | None = None
    ocp_version: str | None = None
    license: str
    known_limitations: list[str]


class SplitFileSummary(StrictModel):
    path: str
    row_count: int
    task_counts: dict[str, int]
    holdout_rule: str


class SplitLabelDistributions(StrictModel):
    relation: dict[str, int]
    binary_answer: dict[str, int]
    volume_bucket: dict[str, int]


class SplitManifestSummary(StrictModel):
    row_count: int
    task_counts: dict[str, int]
    label_distributions: SplitLabelDistributions
    generator_ids: list[str]
    base_object_pair_ids: list[str]
    assembly_group_ids: list[str]
    counterfactual_group_ids: list[str]
    group_holdout_rule_ids: list[str]


class GroupHoldoutRule(StrictModel):
    rule_id: str
    description: str
    group_fields: list[str]
    forbidden_cross_split_pairs: list[list[str]]
    status: AuditStatus


class LeakageViolation(StrictModel):
    split_pair: list[str]
    field: str
    values: list[str]


class LeakageAudit(StrictModel):
    status: AuditStatus
    checked_group_fields: list[str]
    violation_count: int
    violations: list[LeakageViolation]


class SplitManifest(StrictModel):
    dataset_version: str
    split_names: list[str]
    splits: dict[str, SplitManifestSummary]
    group_holdout_rules: list[GroupHoldoutRule]
    leakage_audit: LeakageAudit


class SourceManifestEntry(StrictModel):
    source: str
    archive_path: str | None = None
    archive_available: bool | None = None
    source_dir: str | None = None
    archive_members_scanned: int | None = None
    source_records_loaded: int | None = None
    execution_policy: str | None = None
    purpose: str | None = None
    fixture_count: int | None = None
    generator_id: str | None = None
    object_validation_records: int | None = None


class SourceManifest(StrictModel):
    dataset_version: str
    config_hash: str
    sources: list[SourceManifestEntry]


class DatasetCounts(StrictModel):
    total_rows: int
    by_task: dict[str, int]
    by_split: dict[str, int]
    by_relation: dict[str, int]
    by_source: dict[str, int]
    source_manifest_hash: str


class SmokeGeometryReport(StrictModel):
    cadevolve_archive_members_scanned: int
    cadevolve_source_records_loaded: int
    cadevolve_source_failures: int
    geometry_records: int
    synthetic_fixture_count: int
    label_policy: LabelPolicy
    seed: int
    config_hash: str
    source_manifest: SourceManifest


class SmokeRowsReport(SmokeGeometryReport):
    task_rows: int
    task_counts: dict[str, int]
    relation_counts: dict[str, int]
    split_counts: dict[str, int]
    leakage_audit_status: AuditStatus
    leakage_violation_count: int


class SmokeDatasetReport(SmokeRowsReport):
    source_manifest_hash: str
    object_validation_records: int
    output_dir: str
