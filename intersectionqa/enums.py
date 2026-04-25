"""Shared categorical enums for IntersectionQA records."""

from __future__ import annotations

from enum import StrEnum


class Relation(StrEnum):
    DISJOINT = "disjoint"
    TOUCHING = "touching"
    NEAR_MISS = "near_miss"
    INTERSECTING = "intersecting"
    CONTAINED = "contained"
    INVALID = "invalid"


class BooleanStatus(StrEnum):
    OK = "ok"
    SKIPPED_AABB_DISJOINT = "skipped_aabb_disjoint"
    FAILED = "failed"
    NOT_RUN = "not_run"


class DistanceStatus(StrEnum):
    OK = "ok"
    SKIPPED_POSITIVE_OVERLAP = "skipped_positive_overlap"
    FAILED = "failed"
    NOT_RUN = "not_run"


class LabelStatus(StrEnum):
    OK = "ok"
    INVALID = "invalid"


class TaskType(StrEnum):
    BINARY_INTERFERENCE = "binary_interference"
    RELATION_CLASSIFICATION = "relation_classification"
    VOLUME_BUCKET = "volume_bucket"
    CLEARANCE_BUCKET = "clearance_bucket"
    PAIRWISE_INTERFERENCE = "pairwise_interference"
    RANKING_NORMALIZED_INTERSECTION = "ranking_normalized_intersection"
    REPAIR_DIRECTION = "repair_direction"
    REPAIR_TRANSLATION = "repair_translation"
    TOLERANCE_FIT = "tolerance_fit"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST_RANDOM = "test_random"
    TEST_GENERATOR_HELDOUT = "test_generator_heldout"
    TEST_OBJECT_PAIR_HELDOUT = "test_object_pair_heldout"
    TEST_NEAR_BOUNDARY = "test_near_boundary"
    TEST_TOPOLOGY_HELDOUT = "test_topology_heldout"
    TEST_OPERATION_HELDOUT = "test_operation_heldout"


class FailureReason(StrEnum):
    SOURCE_PARSE_ERROR = "source_parse_error"
    SOURCE_EXEC_ERROR = "source_exec_error"
    MISSING_RESULT_OBJECT = "missing_result_object"
    INVALID_CADQUERY_TYPE = "invalid_cadquery_type"
    NON_SOLID_RESULT = "non_solid_result"
    ZERO_OR_NEGATIVE_VOLUME = "zero_or_negative_volume"
    NON_FINITE_BBOX = "non_finite_bbox"
    BOOLEAN_INTERSECTION_FAILED = "boolean_intersection_failed"
    DISTANCE_QUERY_FAILED = "distance_query_failed"
    TIMEOUT = "timeout"
    WORKER_CRASH = "worker_crash"
    UNKNOWN_ERROR = "unknown_error"


class RotationOrder(StrEnum):
    XYZ = "XYZ"


class AuditStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_RUN = "not_run"


class FailureStage(StrEnum):
    SOURCE_LOADING = "source_loading"
    SOURCE_NORMALIZATION = "source_normalization"
    OBJECT_VALIDATION = "object_validation"
    ASSEMBLY_GENERATION = "assembly_generation"
    GEOMETRY_LABELING = "geometry_labeling"
    TASK_MATERIALIZATION = "task_materialization"
    EXPORT_VALIDATION = "export_validation"
