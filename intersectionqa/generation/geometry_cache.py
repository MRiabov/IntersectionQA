"""Persistent exact-geometry label cache."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from intersectionqa.geometry.cadquery_exec import cadquery_version, ocp_version
from intersectionqa.hashing import sha256_json
from intersectionqa.schema import Diagnostics, GeometryLabels, LabelPolicy


class GeometryLabelCacheRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_key: str
    labels: GeometryLabels
    diagnostics: Diagnostics
    cadquery_version: str | None
    ocp_version: str | None


def geometry_label_cache_key(
    *,
    object_hash_a: str | None,
    object_hash_b: str | None,
    transform_hash: str,
    policy: LabelPolicy,
) -> str:
    return sha256_json(
        {
            "schema": "intersectionqa_geometry_label_cache_v1",
            "object_hash_a": object_hash_a,
            "object_hash_b": object_hash_b,
            "transform_hash": transform_hash,
            "label_policy": policy.model_dump(mode="json"),
            "cadquery_version": cadquery_version(),
            "ocp_version": ocp_version(),
        }
    )


class GeometryLabelCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def get(self, key: str) -> GeometryLabelCacheRecord | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            record = GeometryLabelCacheRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if record.cache_key != key:
            return None
        return record

    def set(
        self,
        key: str,
        *,
        labels: GeometryLabels,
        diagnostics: Diagnostics,
    ) -> None:
        record = GeometryLabelCacheRecord(
            cache_key=key,
            labels=labels,
            diagnostics=diagnostics,
            cadquery_version=cadquery_version(),
            ocp_version=ocp_version(),
        )
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(record.model_dump_json(exclude_none=False) + "\n", encoding="utf-8")
        tmp_path.replace(path)

    def _path(self, key: str) -> Path:
        digest = key.removeprefix("sha256:")
        return self.cache_dir / digest[:2] / f"{digest}.json"
