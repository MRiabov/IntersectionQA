"""Pydantic models for the v0.1 public and internal records."""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

Relation = Literal[
    "disjoint", "touching", "near_miss", "intersecting", "contained", "invalid"
]
BooleanStatus = Literal["ok", "skipped_aabb_disjoint", "failed", "not_run"]
DistanceStatus = Literal["ok", "skipped_positive_overlap", "failed", "not_run"]
LabelStatus = Literal["ok", "invalid"]
TaskType = Literal[
    "binary_interference",
    "relation_classification",
    "volume_bucket",
    "clearance_bucket",
    "pairwise_interference",
    "ranking_normalized_intersection",
    "repair_direction",
    "tolerance_fit",
]
Split = Literal[
    "train",
    "validation",
    "test_random",
    "test_generator_heldout",
    "test_object_pair_heldout",
    "test_near_boundary",
    "test_topology_heldout",
    "test_operation_heldout",
]
FailureReason = Literal[
    "source_parse_error",
    "source_exec_error",
    "missing_result_object",
    "invalid_cadquery_type",
    "non_solid_result",
    "zero_or_negative_volume",
    "non_finite_bbox",
    "boolean_intersection_failed",
    "distance_query_failed",
    "timeout",
    "worker_crash",
    "unknown_error",
]

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
    "aabb_exact_disagreement",
    "contained",
    "primitive_fixture",
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
    rotation_order: Literal["XYZ"] = "XYZ"

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
        if self.label_status == "ok" and self.failure_reason is not None:
            raise ValueError("ok labels must not have failure_reason")
        if self.label_status == "invalid" and self.failure_reason is None:
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
        if self.diagnostics.label_status != "ok":
            raise ValueError("normal MVP public rows require ok label_status")
        if self.task_type == "binary_interference":
            expected = "yes" if self.labels.relation in {"intersecting", "contained"} else "no"
            if self.labels.relation == "invalid" or self.answer != expected:
                raise ValueError("binary answer does not match relation")
        if self.task_type == "relation_classification" and self.answer != self.labels.relation:
            raise ValueError("relation answer must equal labels.relation")
        if self.task_type == "volume_bucket":
            expected_bucket = _expected_volume_bucket(self.labels, self.label_policy)
            if self.answer != expected_bucket:
                raise ValueError("volume_bucket answer does not match labels and policy")
        if self.task_type in {
            "binary_interference",
            "relation_classification",
            "volume_bucket",
        }:
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


class FailureRecord(StrictModel):
    failure_id: str
    stage: Literal[
        "source_loading",
        "source_normalization",
        "object_validation",
        "assembly_generation",
        "geometry_labeling",
        "task_materialization",
        "export_validation",
    ]
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
    splits: dict[str, Any]
    task_types: list[str]
    counts: dict[str, Any]
    cadquery_version: str | None = None
    ocp_version: str | None = None
    license: str
    known_limitations: list[str]
