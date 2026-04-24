"""Object validation records with fixture-only analytic validation."""

from __future__ import annotations

from intersectionqa.geometry.cadquery_exec import cadquery_available, cadquery_version
from intersectionqa.enums import FailureReason
from intersectionqa.hashing import sha256_json
from intersectionqa.schema import (
    BoundingBox,
    Hashes,
    ObjectValidationRecord,
    SourceObjectRecord,
)


def validate_source_object(
    record: SourceObjectRecord,
    *,
    config_hash: str,
    validated_at_version: str,
) -> ObjectValidationRecord:
    if record.source == "synthetic":
        return _validate_synthetic_box(record, config_hash, validated_at_version)
    if not cadquery_available():
        return _invalid_record(
            record,
            config_hash,
            validated_at_version,
            failure_reason=FailureReason.SOURCE_EXEC_ERROR,
        )
    return _invalid_record(
        record,
        config_hash,
        validated_at_version,
        failure_reason=FailureReason.SOURCE_EXEC_ERROR,
    )


def _validate_synthetic_box(
    record: SourceObjectRecord,
    config_hash: str,
    validated_at_version: str,
) -> ObjectValidationRecord:
    params = record.metadata.get("parameters", {})
    width = float(params.get("width", 0.0))
    depth = float(params.get("depth", 0.0))
    height = float(params.get("height", 0.0))
    if width <= 0.0 or depth <= 0.0 or height <= 0.0:
        return _invalid_record(
            record,
            config_hash,
            validated_at_version,
            failure_reason=FailureReason.ZERO_OR_NEGATIVE_VOLUME,
        )
    volume = width * depth * height
    bbox = BoundingBox(
        min=(-width / 2.0, -depth / 2.0, -height / 2.0),
        max=(width / 2.0, depth / 2.0, height / 2.0),
    )
    return ObjectValidationRecord(
        object_id=record.object_id,
        valid=True,
        volume=volume,
        bbox=bbox,
        label_status="ok",
        failure_reason=None,
        cadquery_version=cadquery_version(),
        ocp_version=None,
        validated_at_version=validated_at_version,
        hashes=_validation_hashes(record, config_hash),
    )


def _invalid_record(
    record: SourceObjectRecord,
    config_hash: str,
    validated_at_version: str,
    *,
    failure_reason: FailureReason,
) -> ObjectValidationRecord:
    return ObjectValidationRecord(
        object_id=record.object_id,
        valid=False,
        volume=None,
        bbox=None,
        label_status="invalid",
        failure_reason=failure_reason,
        cadquery_version=cadquery_version(),
        ocp_version=None,
        validated_at_version=validated_at_version,
        hashes=_validation_hashes(record, config_hash),
    )


def _validation_hashes(record: SourceObjectRecord, config_hash: str) -> Hashes:
    object_hash = record.hashes.object_hash or sha256_json(record.model_dump(mode="json"))
    return Hashes(
        source_code_hash=record.hashes.source_code_hash,
        object_hash=object_hash,
        transform_hash=None,
        geometry_hash=None,
        config_hash=config_hash,
        prompt_hash=None,
    )
