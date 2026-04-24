"""Object validation records backed by CadQuery execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import multiprocessing as mp
from queue import Empty
import time

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


@dataclass
class _RunningValidation:
    index: int
    record: SourceObjectRecord
    process: mp.Process
    queue: mp.Queue
    started_at: float


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


def validate_source_objects_bounded(
    records: list[SourceObjectRecord],
    *,
    config_hash: str,
    validated_at_version: str,
    timeout_seconds: float = 10.0,
    worker_count: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> list[ObjectValidationRecord]:
    """Validate source objects with bounded isolated worker processes.

    Results are returned in input order, independent of worker completion order.
    """

    if not records:
        return []
    if worker_count <= 1:
        results = [
            validate_source_object(
                record,
                config_hash=config_hash,
                validated_at_version=validated_at_version,
                timeout_seconds=timeout_seconds,
            )
            for record in records
        ]
        if progress is not None:
            progress(len(results), len(records))
        return results

    context = mp.get_context("spawn")
    pending = list(enumerate(records))
    running: list[_RunningValidation] = []
    results: list[ObjectValidationRecord | None] = [None] * len(records)
    completed = 0
    worker_count = max(1, worker_count)

    def start_next() -> None:
        if not pending:
            return
        index, record = pending.pop(0)
        queue: mp.Queue = context.Queue(maxsize=1)
        process = context.Process(target=_validation_worker, args=(record, queue))
        process.start()
        running.append(
            _RunningValidation(
                index=index,
                record=record,
                process=process,
                queue=queue,
                started_at=time.monotonic(),
            )
        )

    for _ in range(min(worker_count, len(pending))):
        start_next()

    while running:
        made_progress = False
        now = time.monotonic()
        for job in list(running):
            if job.process.is_alive() and now - job.started_at <= timeout_seconds:
                continue
            if job.process.is_alive():
                job.process.terminate()
                job.process.join(1.0)
                result = _invalid_record(
                    job.record,
                    config_hash,
                    validated_at_version,
                    failure_reason=FailureReason.TIMEOUT,
                )
            else:
                job.process.join()
                result = _record_from_completed_worker(
                    job,
                    config_hash=config_hash,
                    validated_at_version=validated_at_version,
                )
            running.remove(job)
            results[job.index] = result
            completed += 1
            made_progress = True
            if progress is not None:
                progress(completed, len(records))
            start_next()
        if not made_progress:
            time.sleep(0.05)

    return [record for record in results if record is not None]


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
    try:
        result = queue.get_nowait()
    except Empty:
        return _invalid_record(
            record,
            config_hash,
            validated_at_version,
            failure_reason=FailureReason.WORKER_CRASH,
        )
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


def _record_from_completed_worker(
    job: _RunningValidation,
    *,
    config_hash: str,
    validated_at_version: str,
) -> ObjectValidationRecord:
    if job.process.exitcode != 0:
        return _invalid_record(
            job.record,
            config_hash,
            validated_at_version,
            failure_reason=FailureReason.WORKER_CRASH,
        )
    try:
        result = job.queue.get_nowait()
    except Empty:
        return _invalid_record(
            job.record,
            config_hash,
            validated_at_version,
            failure_reason=FailureReason.WORKER_CRASH,
        )
    if result["status"] == "error":
        return _invalid_record(
            job.record,
            config_hash,
            validated_at_version,
            failure_reason=FailureReason(result["failure_reason"]),
        )
    return ObjectValidationRecord(
        object_id=job.record.object_id,
        valid=True,
        volume=result["volume"],
        bbox=result["bbox"],
        label_status="ok",
        failure_reason=None,
        cadquery_version=result["cadquery_version"],
        ocp_version=result["ocp_version"],
        validated_at_version=validated_at_version,
        hashes=_validation_hashes(job.record, config_hash),
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
