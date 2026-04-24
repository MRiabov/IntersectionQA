"""Deterministic CADEvolve two-object assembly candidates."""

from __future__ import annotations

from dataclasses import dataclass
import sys
import time
from typing import Any

from intersectionqa.enums import FailureReason, FailureStage, LabelStatus, Relation
from intersectionqa.generation.assembly import TwoObjectAssembly
from intersectionqa.generation.geometry_cache import GeometryLabelCache, geometry_label_cache_key
from intersectionqa.geometry.bbox import AABB, transform_aabb
from intersectionqa.geometry.cadquery_exec import (
    CadQueryExecutionError,
    cadquery_version,
    execute_source_object,
    measure_shape_pair,
    ocp_version,
)
from intersectionqa.geometry.labels import derive_labels, validate_label_consistency
from intersectionqa.hashing import sha256_json, sha256_text
from intersectionqa.schema import (
    ArtifactIds,
    BoundingBox,
    FailureRecord,
    GeometryRecord,
    Hashes,
    LabelPolicy,
    ObjectValidationRecord,
    SourceObjectRecord,
    Transform,
)


@dataclass(frozen=True)
class CadevolveGeometryBuild:
    records: list[GeometryRecord]
    failures: list[FailureRecord]


@dataclass(frozen=True)
class _CandidateSpec:
    strategy: str
    translation_gap: float
    rotation_b: tuple[float, float, float]
    tags: tuple[str, ...]
    changed_parameter: str | None = "transform_b.translation[0]"


def generate_cadevolve_geometry_records(
    sources: list[SourceObjectRecord],
    validations_by_object_id: dict[str, ObjectValidationRecord],
    *,
    policy: LabelPolicy,
    config_hash: str,
    max_records: int,
    geometry_cache: GeometryLabelCache | None = None,
) -> CadevolveGeometryBuild:
    """Build a bounded deterministic set of exact-labeled CADEvolve assemblies."""

    started = time.monotonic()
    valid_sources = [
        source
        for source in sorted(sources, key=lambda item: item.source_path or item.source_id)
        if _is_validated(source, validations_by_object_id)
    ]
    records: list[GeometryRecord] = []
    failures: list[FailureRecord] = []
    if max_records <= 0 or len(valid_sources) < 2:
        _progress(
            "CADEvolve candidate generation skipped: "
            f"valid_sources={len(valid_sources)}, max_records={max_records}"
        )
        return CadevolveGeometryBuild(records=records, failures=failures)

    measured_by_object_id, execution_failures = _execute_valid_sources(valid_sources, config_hash)
    failures.extend(execution_failures)
    valid_sources = [source for source in valid_sources if source.object_id in measured_by_object_id]
    if len(valid_sources) < 2:
        _progress(
            "CADEvolve candidate generation skipped after source execution: "
            f"valid_executed_sources={len(valid_sources)}, execution_failures={len(execution_failures)}"
        )
        return CadevolveGeometryBuild(records=records, failures=failures)

    _progress(
        f"generating CADEvolve geometry records: 0/{max_records}, "
        f"valid_sources={len(valid_sources)}"
    )
    pair_index = 0
    attempts = 0
    cache_hits = 0
    for object_a, object_b in _deterministic_pairs(valid_sources):
        pair_index += 1
        validation_a = validations_by_object_id[object_a.object_id]
        validation_b = validations_by_object_id[object_b.object_id]
        for spec_index, spec in enumerate(_candidate_specs(validation_a, validation_b, policy), start=1):
            if len(records) >= max_records:
                _progress(
                    f"generating CADEvolve geometry records: {len(records)}/{max_records}, "
                    f"attempts={attempts}, failures={len(failures)}, cache_hits={cache_hits}, "
                    f"elapsed={_elapsed(started)}"
                )
                return CadevolveGeometryBuild(records=records, failures=failures)
            attempts += 1
            geometry_index = len(records) + 1
            try:
                record = _measure_candidate(
                    object_a,
                    object_b,
                    validation_a,
                    validation_b,
                    measured_by_object_id,
                    spec,
                    pair_index=pair_index,
                    spec_index=spec_index,
                    geometry_index=geometry_index,
                    policy=policy,
                    config_hash=config_hash,
                    geometry_cache=geometry_cache,
                )
            except CadQueryExecutionError as exc:
                failures.append(
                    _failure_record(
                        object_a,
                        object_b,
                        spec,
                        geometry_index,
                        exc.failure_reason,
                        str(exc),
                        config_hash=config_hash,
                    )
                )
                continue
            except Exception as exc:
                failures.append(
                    _failure_record(
                        object_a,
                        object_b,
                        spec,
                        geometry_index,
                        FailureReason.UNKNOWN_ERROR,
                        str(exc),
                        config_hash=config_hash,
                    )
                )
                continue
            if record.diagnostics.label_status != LabelStatus.OK or record.labels.relation == Relation.INVALID:
                failures.append(
                    _failure_record(
                        object_a,
                        object_b,
                        spec,
                        geometry_index,
                        record.diagnostics.failure_reason or FailureReason.UNKNOWN_ERROR,
                        "candidate measured but produced an invalid label",
                        config_hash=config_hash,
                        geometry_id=record.geometry_id,
                    )
                )
                continue
            try:
                validate_label_consistency(record.labels, record.diagnostics, policy)
            except ValueError as exc:
                failures.append(
                    _failure_record(
                        object_a,
                        object_b,
                        spec,
                        geometry_index,
                        FailureReason.UNKNOWN_ERROR,
                        f"candidate failed label consistency: {exc}",
                        config_hash=config_hash,
                        geometry_id=record.geometry_id,
                    )
                )
                continue
            if record.metadata.get("geometry_label_cache_hit") is True:
                cache_hits += 1
            records.append(record)
            if len(records) == max_records or len(records) % 10 == 0:
                _progress(
                    f"generating CADEvolve geometry records: {len(records)}/{max_records}, "
                    f"attempts={attempts}, failures={len(failures)}, cache_hits={cache_hits}, "
                    f"elapsed={_elapsed(started)}"
                )
    _progress(
        f"generating CADEvolve geometry records: {len(records)}/{max_records}, "
        f"attempts={attempts}, failures={len(failures)}, cache_hits={cache_hits}, "
        f"elapsed={_elapsed(started)}"
    )
    return CadevolveGeometryBuild(records=records, failures=failures)


def _is_validated(
    source: SourceObjectRecord,
    validations_by_object_id: dict[str, ObjectValidationRecord],
) -> bool:
    validation = validations_by_object_id.get(source.object_id)
    return bool(validation and validation.valid and validation.bbox is not None)


def _execute_valid_sources(
    valid_sources: list[SourceObjectRecord],
    config_hash: str,
) -> tuple[dict[str, Any], list[FailureRecord]]:
    started = time.monotonic()
    total = len(valid_sources)
    measured_by_object_id: dict[str, Any] = {}
    failures: list[FailureRecord] = []
    _progress(f"executing valid CADEvolve source shapes for reuse: 0/{total}")
    for index, source in enumerate(valid_sources, start=1):
        try:
            measured_by_object_id[source.object_id] = execute_source_object(source)
        except CadQueryExecutionError as exc:
            failures.append(_source_execution_failure(source, index, exc.failure_reason, str(exc), config_hash))
        except Exception as exc:
            failures.append(_source_execution_failure(source, index, FailureReason.UNKNOWN_ERROR, str(exc), config_hash))
        if index == total or index % 10 == 0:
            _progress(
                f"executing valid CADEvolve source shapes for reuse: {index}/{total}, "
                f"executed={len(measured_by_object_id)}, failures={len(failures)}, elapsed={_elapsed(started)}"
            )
    return measured_by_object_id, failures


def _deterministic_pairs(
    sources: list[SourceObjectRecord],
) -> list[tuple[SourceObjectRecord, SourceObjectRecord]]:
    pairs: list[tuple[SourceObjectRecord, SourceObjectRecord]] = []
    for index, source in enumerate(sources):
        pairs.append((source, sources[(index + 1) % len(sources)]))
    return pairs


def _candidate_specs(
    validation_a: ObjectValidationRecord,
    validation_b: ObjectValidationRecord,
    policy: LabelPolicy,
) -> list[_CandidateSpec]:
    assert validation_a.bbox is not None
    assert validation_b.bbox is not None
    span_x = min(_span(validation_a.bbox, 0), _span(validation_b.bbox, 0))
    overlap_small = max(0.01, min(0.05, span_x * 0.01))
    overlap_clear = max(overlap_small * 4.0, span_x * 0.25)
    clear_gap = max(policy.near_miss_threshold_mm + 5.0, span_x * 0.5)
    far_gap = clear_gap + _diagonal(validation_a.bbox) + _diagonal(validation_b.bbox)
    return [
        _CandidateSpec(
            strategy="clear_disjoint",
            translation_gap=clear_gap,
            rotation_b=(0.0, 0.0, 0.0),
            tags=("axis_aligned",),
        ),
        _CandidateSpec(
            strategy="bbox_touching",
            translation_gap=0.0,
            rotation_b=(0.0, 0.0, 0.0),
            tags=("axis_aligned", "contact_vs_interference", "near_boundary"),
        ),
        _CandidateSpec(
            strategy="near_miss",
            translation_gap=0.5,
            rotation_b=(0.0, 0.0, 0.0),
            tags=("axis_aligned", "near_boundary", "near_miss"),
        ),
        _CandidateSpec(
            strategy="tiny_overlap",
            translation_gap=-overlap_small,
            rotation_b=(0.0, 0.0, 0.0),
            tags=("axis_aligned", "near_boundary", "tiny_overlap"),
        ),
        _CandidateSpec(
            strategy="clear_overlap",
            translation_gap=-overlap_clear,
            rotation_b=(0.0, 0.0, 0.0),
            tags=("axis_aligned",),
        ),
        _CandidateSpec(
            strategy="rotated_clear_disjoint",
            translation_gap=far_gap,
            rotation_b=(0.0, 0.0, 15.0),
            tags=("rotated",),
            changed_parameter="transform_b.rotation_xyz_deg[2]",
        ),
    ]


def _measure_candidate(
    object_a: SourceObjectRecord,
    object_b: SourceObjectRecord,
    validation_a: ObjectValidationRecord,
    validation_b: ObjectValidationRecord,
    measured_by_object_id: dict[str, Any],
    spec: _CandidateSpec,
    *,
    pair_index: int,
    spec_index: int,
    geometry_index: int,
    policy: LabelPolicy,
    config_hash: str,
    geometry_cache: GeometryLabelCache | None,
) -> GeometryRecord:
    assert validation_a.bbox is not None
    assert validation_b.bbox is not None
    local_a = _aabb_from_bbox(validation_a.bbox)
    local_b = _aabb_from_bbox(validation_b.bbox)
    transform_a = Transform(translation=(0.0, 0.0, 0.0), rotation_xyz_deg=(0.0, 0.0, 0.0))
    transform_b = _bbox_guided_transform(local_a, local_b, spec)
    pair_object_a = _with_function_name(object_a, "object_a")
    pair_object_b = _with_function_name(object_b, "object_b")
    transform_hash = sha256_json(
        {
            "candidate_strategy": spec.strategy,
            "transform_a": transform_a.model_dump(mode="json"),
            "transform_b": transform_b.model_dump(mode="json"),
        }
    )
    label_cache_key = geometry_label_cache_key(
        object_hash_a=object_a.hashes.object_hash,
        object_hash_b=object_b.hashes.object_hash,
        transform_hash=transform_hash,
        policy=policy,
    )
    cached = geometry_cache.get(label_cache_key) if geometry_cache is not None else None
    geometry_label_cache_hit = cached is not None
    if cached is not None:
        labels = cached.labels
        diagnostics = cached.diagnostics
    else:
        raw_geometry = measure_shape_pair(
            measured_by_object_id[object_a.object_id].shape,
            measured_by_object_id[object_b.object_id].shape,
            transform_a,
            transform_b,
            policy,
        )
        labels, diagnostics = derive_labels(raw_geometry, policy)
        if geometry_cache is not None:
            geometry_cache.set(label_cache_key, labels=labels, diagnostics=diagnostics)
    assembly = TwoObjectAssembly(pair_object_a, pair_object_b, transform_a, transform_b)
    object_hash = sha256_json([object_a.hashes.object_hash, object_b.hashes.object_hash])
    geometry_hash = sha256_json(
        {
            "object_hash": object_hash,
            "transform_hash": transform_hash,
            "labels": labels.model_dump(mode="json"),
            "policy": policy.model_dump(mode="json"),
        }
    )
    world_a = transform_aabb(local_a, transform_a)
    world_b = transform_aabb(local_b, transform_b)
    counterfactual_group_id = (
        f"cfg_cadevolve_{pair_index:06d}" if spec.changed_parameter == "transform_b.translation[0]" else None
    )
    variant_id = (
        f"{counterfactual_group_id}_v{spec_index:02d}" if counterfactual_group_id else None
    )
    return GeometryRecord(
        geometry_id=f"geom_cadevolve_{geometry_index:06d}",
        source="cadevolve",
        object_a_id=object_a.object_id,
        object_b_id=object_b.object_id,
        base_object_pair_id=f"pair_cadevolve_{pair_index:06d}",
        assembly_group_id=f"asmgrp_cadevolve_{pair_index:06d}",
        counterfactual_group_id=counterfactual_group_id,
        variant_id=variant_id,
        changed_parameter=spec.changed_parameter,
        changed_value=transform_b.translation[0]
        if spec.changed_parameter == "transform_b.translation[0]"
        else transform_b.rotation_xyz_deg[2],
        transform_a=transform_a,
        transform_b=transform_b,
        assembly_script=assembly.script(),
        labels=labels,
        diagnostics=diagnostics,
        difficulty_tags=_difficulty_tags(object_a, object_b, spec),
        label_policy=policy,
        hashes=Hashes(
            source_code_hash=sha256_text(assembly.object_code),
            object_hash=object_hash,
            transform_hash=transform_hash,
            geometry_hash=geometry_hash,
            config_hash=config_hash,
            prompt_hash=None,
        ),
        metadata={
            "candidate_strategy": spec.strategy,
            "source_ids": [object_a.source_id, object_b.source_id],
            "source_paths": [object_a.source_path, object_b.source_path],
            "source_subtrees": [
                object_a.metadata.get("source_subset") or object_a.metadata.get("source_tree"),
                object_b.metadata.get("source_subset") or object_b.metadata.get("source_tree"),
            ],
            "generator_ids": [object_a.generator_id, object_b.generator_id],
            "artifact_ids": ArtifactIds().model_dump(mode="json"),
            "bbox_a": world_a.to_schema().model_dump(mode="json"),
            "bbox_b": world_b.to_schema().model_dump(mode="json"),
            "object_validation": {
                object_a.object_id: _validation_summary(validation_a),
                object_b.object_id: _validation_summary(validation_b),
            },
            "cadquery_version": cadquery_version(),
            "ocp_version": ocp_version(),
            "candidate_transform_values": {
                "translation_gap": spec.translation_gap,
                "transform_a": transform_a.model_dump(mode="json"),
                "transform_b": transform_b.model_dump(mode="json"),
            },
            "geometry_label_cache_key": label_cache_key,
            "geometry_label_cache_hit": geometry_label_cache_hit,
        },
    )


def _bbox_guided_transform(local_a: AABB, local_b: AABB, spec: _CandidateSpec) -> Transform:
    translation = [
        local_a.max[0] + spec.translation_gap - local_b.min[0],
        _center(local_a, 1) - _center(local_b, 1),
        _center(local_a, 2) - _center(local_b, 2),
    ]
    return Transform(translation=tuple(translation), rotation_xyz_deg=spec.rotation_b)


def _with_function_name(record: SourceObjectRecord, function_name: str) -> SourceObjectRecord:
    if record.object_function_name == function_name:
        return record
    old_prefix = f"def {record.object_function_name}("
    new_prefix = f"def {function_name}("
    if old_prefix not in record.normalized_code:
        raise CadQueryExecutionError(
            FailureReason.MISSING_RESULT_OBJECT,
            f"cannot alias {record.object_function_name} to {function_name}",
        )
    return record.model_copy(
        update={
            "normalized_code": record.normalized_code.replace(old_prefix, new_prefix, 1),
            "object_function_name": function_name,
        }
    )


def _difficulty_tags(
    object_a: SourceObjectRecord,
    object_b: SourceObjectRecord,
    spec: _CandidateSpec,
) -> list[str]:
    tags = {"cadevolve_compound" if _is_compoundish(object_a, object_b) else "cadevolve_simple"}
    tags.update(spec.tags)
    return sorted(tags)


def _is_compoundish(object_a: SourceObjectRecord, object_b: SourceObjectRecord) -> bool:
    compound_ops = {
        "cut",
        "union",
        "intersect",
        "loft",
        "sweep",
        "fillet",
        "chamfer",
        "shell",
        "hole",
        "extrude",
    }
    ops = set(object_a.cadquery_ops) | set(object_b.cadquery_ops)
    return bool(ops & compound_ops)


def _failure_record(
    object_a: SourceObjectRecord,
    object_b: SourceObjectRecord,
    spec: _CandidateSpec,
    index: int,
    failure_reason: FailureReason,
    error_summary: str,
    *,
    config_hash: str,
    geometry_id: str | None = None,
) -> FailureRecord:
    return FailureRecord(
        failure_id=f"fail_cadevolve_geometry_{index:06d}",
        stage=FailureStage.GEOMETRY_LABELING,
        source="cadevolve",
        source_id=f"{object_a.source_id}|{object_b.source_id}",
        object_id=f"{object_a.object_id}|{object_b.object_id}",
        geometry_id=geometry_id,
        failure_reason=failure_reason,
        error_summary=f"{spec.strategy}: {error_summary}",
        retry_count=0,
        hashes=Hashes(
            source_code_hash=None,
            object_hash=sha256_json([object_a.hashes.object_hash, object_b.hashes.object_hash]),
            transform_hash=sha256_json(
                {
                    "candidate_strategy": spec.strategy,
                    "translation_gap": spec.translation_gap,
                    "rotation_b": spec.rotation_b,
                }
            ),
            geometry_hash=None,
            config_hash=config_hash,
            prompt_hash=None,
        ),
    )


def _source_execution_failure(
    source: SourceObjectRecord,
    index: int,
    failure_reason: FailureReason,
    error_summary: str,
    config_hash: str,
) -> FailureRecord:
    return FailureRecord(
        failure_id=f"fail_cadevolve_source_exec_{index:06d}",
        stage=FailureStage.GEOMETRY_LABELING,
        source="cadevolve",
        source_id=source.source_id,
        object_id=source.object_id,
        geometry_id=None,
        failure_reason=failure_reason,
        error_summary=f"source shape execution for geometry cache failed: {error_summary}",
        retry_count=0,
        hashes=Hashes(
            source_code_hash=source.hashes.source_code_hash,
            object_hash=source.hashes.object_hash,
            transform_hash=None,
            geometry_hash=None,
            config_hash=config_hash,
            prompt_hash=None,
        ),
    )


def _validation_summary(validation: ObjectValidationRecord) -> dict[str, Any]:
    return {
        "valid": validation.valid,
        "volume": validation.volume,
        "bbox": validation.bbox.model_dump(mode="json") if validation.bbox else None,
        "cadquery_version": validation.cadquery_version,
        "ocp_version": validation.ocp_version,
    }


def _aabb_from_bbox(bbox: BoundingBox) -> AABB:
    return AABB(min=bbox.min, max=bbox.max)


def _center(aabb: AABB, axis: int) -> float:
    return (aabb.min[axis] + aabb.max[axis]) / 2.0


def _span(bbox: BoundingBox, axis: int) -> float:
    return max(1e-6, bbox.max[axis] - bbox.min[axis])


def _diagonal(bbox: BoundingBox) -> float:
    return sum((bbox.max[index] - bbox.min[index]) ** 2 for index in range(3)) ** 0.5


def _progress(message: str) -> None:
    print(f"[intersectionqa] {message}", file=sys.stderr, flush=True)


def _elapsed(started: float) -> str:
    return f"{time.monotonic() - started:.1f}s"
