"""CADEvolve tar archive source loader."""

from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path
from typing import Any

from intersectionqa.hashing import sha256_json, sha256_text
from intersectionqa.schema import Hashes, SourceObjectRecord
from intersectionqa.sources.base import SourceLoadResult

EXECUTABLE_PREFIXES = ("CADEvolve-P/", "CADEvolve-C/")
RESULT_NAMES = ("result", "shape", "solid", "part")
OP_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)\(")


class CadevolveTarLoader:
    def __init__(
        self,
        archive_path: Path | None,
        config_hash: str,
        *,
        member_index_cache_dir: Path | None = None,
    ) -> None:
        self.archive_path = archive_path
        self.config_hash = config_hash
        self.member_index_cache_dir = member_index_cache_dir

    def load(self, limit: int | None = None, offset: int = 0) -> SourceLoadResult:
        if self.archive_path is None or not self.archive_path.exists():
            return SourceLoadResult(records=[], failures=[], scanned_count=0)

        offset = max(0, offset)
        records: list[SourceObjectRecord] = []
        executable_members = self._load_executable_member_index()
        selected_members = (
            executable_members[offset : offset + limit]
            if limit is not None
            else executable_members[offset:]
        )
        with self.archive_path.open("rb") as raw_archive:
            for member in selected_members:
                if limit is not None and len(records) >= limit:
                    break
                source_path = str(member["path"])
                code = _read_indexed_member(raw_archive, member).decode("utf-8", errors="replace")
                object_id = f"obj_cadevolve_{len(records) + 1:06d}"
                source_hash = sha256_text(code)
                normalized_code = _wrap_as_object_function(code)
                object_hash = sha256_json(
                    {
                        "source": "cadevolve",
                        "source_path": source_path,
                        "normalized_code": normalized_code,
                    }
                )
                records.append(
                    SourceObjectRecord(
                        object_id=object_id,
                        source="cadevolve",
                        source_id=source_path,
                        generator_id=_generator_id(source_path),
                        source_path=source_path,
                        source_license="apache-2.0",
                        object_name=Path(source_path).stem,
                        normalized_code=normalized_code,
                        object_function_name="object_source",
                        cadquery_ops=sorted(set(OP_RE.findall(code))),
                        topology_tags=[],
                        metadata={
                            "source_tree": source_path.split("/", 1)[0],
                            "source_subset": _source_subset(source_path),
                            "raw_source_code_hash": source_hash,
                            "validation_status": "not_run",
                        },
                        hashes=Hashes(
                            source_code_hash=source_hash,
                            object_hash=object_hash,
                            transform_hash=None,
                            geometry_hash=None,
                            config_hash=self.config_hash,
                            prompt_hash=None,
                        ),
                    )
                )
        return SourceLoadResult(records=records, failures=[], scanned_count=len(selected_members))

    def _load_executable_member_index(self) -> list[dict[str, Any]]:
        cache_path = self._member_index_cache_path()
        if cache_path is not None and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("archive") == _archive_fingerprint(self.archive_path):
                    return list(cached["members"])
            except Exception:
                pass

        members = _build_executable_member_index(self.archive_path)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(
                    {
                        "schema": "intersectionqa_cadevolve_member_index_v1",
                        "archive": _archive_fingerprint(self.archive_path),
                        "members": members,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            tmp_path.replace(cache_path)
        return members

    def _member_index_cache_path(self) -> Path | None:
        if self.member_index_cache_dir is None or self.archive_path is None:
            return None
        key = sha256_json(_archive_fingerprint(self.archive_path)).removeprefix("sha256:")
        return self.member_index_cache_dir / key[:2] / f"{key}.json"


def _normalized_path(path: str) -> str:
    return path.removeprefix("./")


def _is_executable_source_member(member: tarfile.TarInfo, source_path: str) -> bool:
    return (
        member.isfile()
        and source_path.endswith(".py")
        and source_path.startswith(EXECUTABLE_PREFIXES)
    )


def _build_executable_member_index(archive_path: Path) -> list[dict[str, Any]]:
    with tarfile.open(archive_path, "r:*") as archive:
        return sorted(
            (
                {
                    "path": source_path,
                    "size": int(member.size),
                    "offset_data": int(member.offset_data),
                }
                for member in archive
                for source_path in [_normalized_path(member.name)]
                if _is_executable_source_member(member, source_path)
            ),
            key=lambda item: item["path"],
        )


def _read_indexed_member(raw_archive: Any, member: dict[str, Any]) -> bytes:
    raw_archive.seek(int(member["offset_data"]))
    return raw_archive.read(int(member["size"]))


def _archive_fingerprint(archive_path: Path) -> dict[str, Any]:
    stat = archive_path.stat()
    return {
        "path": str(archive_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _source_subset(path: str) -> str:
    parts = path.split("/")
    if len(parts) > 2:
        return "/".join(parts[:2])
    return parts[0]


def _generator_id(path: str) -> str:
    parts = path.split("/")
    prefix = "/".join(parts[:2]) if len(parts) > 2 else parts[0]
    return "cadevolve_" + re.sub(r"[^A-Za-z0-9]+", "_", prefix).strip("_").lower()


def _wrap_as_object_function(code: str) -> str:
    indented = "\n".join("    " + line if line.strip() else "" for line in code.strip().splitlines())
    probes = ", ".join(repr(name) for name in RESULT_NAMES)
    return "\n".join(
        [
            "def object_source():",
            indented,
            f"    for _name in ({probes}):",
            "        if _name in locals():",
            "            return locals()[_name]",
            "    raise RuntimeError('CADEvolve source did not expose a known result object')",
            "",
        ]
    )
