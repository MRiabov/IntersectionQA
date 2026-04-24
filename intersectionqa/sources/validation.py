"""Object validation records backed by CadQuery execution."""

from __future__ import annotations

import multiprocessing as mp

from intersectionqa.enums import FailureReason
from intersectionqa.geometry.cadquery_exec import (
    CadQueryExecutionError,
    cadquery_version,
    execute_source_object,
    ocp_version,
)
from intersectionqa.hashing import sha256_json
from intersectionqa.schema import (
    Hashes,
    ObjectValidationRecord,
    SourceObjectRecord,
)


def validate_source_object(
    record: SourceObjectRecord,
    *,
    config_hash: str,
    validated_at_version: str,
    timeout_seconds: float = 10.0,
    isolated: bool | None = None,
) -> ObjectValidationRecord:
    isolated = record.source != "synthetic" if isolated is None else isolated
    if isolated:
        return _validate_source_object_isolated(
            record,
            config_hash=config_hash,
            validated_at_version=validated_at_version,
            timeout_seconds=timeout_seconds,
        )
    try:
        measured = execute_source_object(record)
    except CadQueryExecutionError as exc:
        return _invalid_record(
            record,
            config_hash,
            validated_at_version,
            failure_reason=exc.failure_reason,
        )
    except Exception:
        return _invalid_record(
            record,
            config_hash,
            validated_at_version,
            failure_reason=FailureReason.UNKNOWN_ERROR,
        )
    return ObjectValidationRecord(
        object_id=record.object_id,
        valid=True,
        volume=measured.volume,
        bbox=measured.bbox,
        label_status="ok",
        failure_reason=None,
        cadquery_version=cadquery_version(),
        ocp_version=ocp_version(),
        validated_at_version=validated_at_version,
        hashes=_validation_hashes(record, config_hash),
    )


def _validate_source_object_isolated(
    record: SourceObjectRecord,
    *,
    config_hash: str,
    validated_at_version: str,
    timeout_seconds: float,
) -> ObjectValidationRecord:
    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue(maxsize=1)
    process = context.Process(target=_validation_worker, args=(record, queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        return _invalid_record(
            record,
            config_hash,
            validated_at_version,
            failure_reason=FailureReason.TIMEOUT,
        )
    if process.exitcode != 0:
        return _invalid_record(
            record,
            config_hash,
            validated_at_version,
            failure_reason=FailureReason.WORKER_CRASH,
        )
    if queue.empty():
        return _invalid_record(
            record,
            config_hash,
            validated_at_version,
            failure_reason=FailureReason.WORKER_CRASH,
        )
    result = queue.get()
    if result["status"] == "error":
        return _invalid_record(
            record,
            config_hash,
            validated_at_version,
            failure_reason=FailureReason(result["failure_reason"]),
        )
    return ObjectValidationRecord(
        object_id=record.object_id,
        valid=True,
        volume=result["volume"],
        bbox=result["bbox"],
        label_status="ok",
        failure_reason=None,
        cadquery_version=result["cadquery_version"],
        ocp_version=result["ocp_version"],
        validated_at_version=validated_at_version,
        hashes=_validation_hashes(record, config_hash),
    )


def _validation_worker(record: SourceObjectRecord, queue: mp.Queue) -> None:
    try:
        measured = execute_source_object(record)
        queue.put(
            {
                "status": "ok",
                "volume": measured.volume,
                "bbox": measured.bbox.model_dump(mode="json"),
                "cadquery_version": cadquery_version(),
                "ocp_version": ocp_version(),
            }
        )
    except CadQueryExecutionError as exc:
        queue.put({"status": "error", "failure_reason": exc.failure_reason.value})
    except Exception:
        queue.put({"status": "error", "failure_reason": FailureReason.UNKNOWN_ERROR.value})


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
        ocp_version=ocp_version(),
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
