"""Persistent object-validation cache."""

from __future__ import annotations

from pathlib import Path

from intersectionqa.hashing import sha256_json
from intersectionqa.schema import ObjectValidationRecord, SourceObjectRecord


def object_validation_cache_key(
    source: SourceObjectRecord,
    *,
    timeout_seconds: float,
    validated_at_version: str,
) -> str:
    return sha256_json(
        {
            "schema": "intersectionqa_object_validation_cache_v1",
            "object_id": source.object_id,
            "source": source.source,
            "source_id": source.source_id,
            "source_code_hash": source.hashes.source_code_hash,
            "object_hash": source.hashes.object_hash,
            "timeout_seconds": timeout_seconds,
            "validated_at_version": validated_at_version,
        }
    )


class ObjectValidationCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def get(self, key: str) -> ObjectValidationRecord | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return ObjectValidationRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, key: str, record: ObjectValidationRecord) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(record.model_dump_json(exclude_none=False) + "\n", encoding="utf-8")
        tmp_path.replace(path)

    def _path(self, key: str) -> Path:
        digest = key.removeprefix("sha256:")
        return self.cache_dir / digest[:2] / f"{digest}.json"
