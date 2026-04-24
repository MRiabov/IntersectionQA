"""CADEvolve tar archive source loader."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class CadevolveExtractionReport:
    cache_root: Path
    selected_count: int
    newly_extracted_count: int
    already_present_count: int
    selected_size_bytes: int
    complete: bool


class CadevolveTarLoader:
    def __init__(
        self,
        archive_path: Path | None,
        config_hash: str,
        *,
        member_index_cache_dir: Path | None = None,
        extracted_source_cache_dir: Path | None = None,
        extracted_source_cache_root: Path | None = None,
    ) -> None:
        self.archive_path = archive_path
        self.config_hash = config_hash
        self.member_index_cache_dir = member_index_cache_dir
        self.extracted_source_cache_dir = extracted_source_cache_dir
        self._explicit_extracted_source_cache_root = extracted_source_cache_root

    def load(self, limit: int | None = None, offset: int = 0) -> SourceLoadResult:
        cache_root = self.extracted_source_cache_root()
        explicit_cache_root = self._explicit_extracted_source_cache_root is not None
        archive_available = self.archive_path is not None and self.archive_path.exists()
        if not archive_available and cache_root is None:
            return SourceLoadResult(records=[], failures=[], scanned_count=0)

        offset = max(0, offset)
        records: list[SourceObjectRecord] = []
        cached_selection = self._load_extracted_selection(limit=limit, offset=offset)
        selected_members = cached_selection
        executable_member_count: int | None = None
        if selected_members is None:
            if not archive_available:
                if explicit_cache_root:
                    raise FileNotFoundError(
                        "explicit CADEvolve source cache root is missing, invalid, or too small: "
                        f"{cache_root}"
                    )
                return SourceLoadResult(records=[], failures=[], scanned_count=0)
            executable_members = self._load_executable_member_index()
            executable_member_count = len(executable_members)
            selected_members = (
                executable_members[offset : offset + limit]
                if limit is not None
                else executable_members[offset:]
            )
        if cache_root is not None:
            source_files = self._ensure_extracted_sources(
                selected_members,
                offset=offset,
                complete=limit is None and executable_member_count == len(selected_members),
            )
            for member in selected_members:
                source_path = str(member["path"])
                code = source_files[source_path].read_text(encoding="utf-8", errors="replace")
                records.append(_source_object_record(source_path, code, len(records) + 1, self.config_hash))
        else:
            with self.archive_path.open("rb") as raw_archive:
                for member in selected_members:
                    source_path = str(member["path"])
                    code = _read_indexed_member(raw_archive, member).decode("utf-8", errors="replace")
                    records.append(_source_object_record(source_path, code, len(records) + 1, self.config_hash))
        return SourceLoadResult(records=records, failures=[], scanned_count=len(selected_members))

    def extracted_source_cache_root(self) -> Path | None:
        if self._explicit_extracted_source_cache_root is not None:
            return self._explicit_extracted_source_cache_root
        if self.extracted_source_cache_dir is None:
            return None
        if self.archive_path is None:
            return None
        key = sha256_json(_archive_fingerprint(self.archive_path)).removeprefix("sha256:")
        return self.extracted_source_cache_dir / key[:2] / key

    def prepare_extracted_sources(
        self,
        *,
        limit: int | None,
        offset: int = 0,
    ) -> CadevolveExtractionReport:
        if self.archive_path is None or not self.archive_path.exists():
            raise FileNotFoundError("CADEvolve archive is required to prepare extracted sources")
        cache_root = self.extracted_source_cache_root()
        if cache_root is None:
            raise ValueError("extracted_source_cache_dir is required to prepare extracted sources")

        offset = max(0, offset)
        cached_selection = self._load_extracted_selection(limit=limit, offset=offset)
        if cached_selection is not None:
            return CadevolveExtractionReport(
                cache_root=cache_root,
                selected_count=len(cached_selection),
                newly_extracted_count=0,
                already_present_count=len(cached_selection),
                selected_size_bytes=sum(int(member["size"]) for member in cached_selection),
                complete=limit is None,
            )

        executable_members = self._load_executable_member_index()
        selected_members = (
            executable_members[offset : offset + limit]
            if limit is not None
            else executable_members[offset:]
        )
        source_files = {
            str(member["path"]): _extracted_member_path(cache_root, str(member["path"]))
            for member in selected_members
        }
        already_present_count = sum(1 for path in source_files.values() if path.exists())
        complete = limit is None and len(selected_members) == len(executable_members)
        self._ensure_extracted_sources(selected_members, offset=offset, complete=complete)
        return CadevolveExtractionReport(
            cache_root=cache_root,
            selected_count=len(selected_members),
            newly_extracted_count=len(selected_members) - already_present_count,
            already_present_count=already_present_count,
            selected_size_bytes=sum(int(member["size"]) for member in selected_members),
            complete=complete,
        )

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

    def _load_extracted_selection(
        self,
        *,
        limit: int | None,
        offset: int,
    ) -> list[dict[str, Any]] | None:
        cache_root = self.extracted_source_cache_root()
        if cache_root is None:
            return None

        manifest = _read_extracted_manifest(cache_root, self.archive_path)
        if manifest is None:
            return None

        if limit is None:
            if manifest.get("complete") is not True:
                return None
            required_end = None
        else:
            required_end = offset + limit
            if int(manifest.get("prefix_count", 0)) < required_end:
                return None

        members = sorted(list(manifest.get("members", [])), key=lambda item: item["path"])
        selected = members[offset:required_end]
        if limit is not None and len(selected) != limit:
            return None
        if any(not _extracted_member_path(cache_root, str(member["path"])).exists() for member in selected):
            return None
        return selected

    def _ensure_extracted_sources(
        self,
        members: list[dict[str, Any]],
        *,
        offset: int,
        complete: bool,
    ) -> dict[str, Path]:
        cache_root = self.extracted_source_cache_root()
        if cache_root is None:
            return {}

        source_files = {
            str(member["path"]): _extracted_member_path(cache_root, str(member["path"]))
            for member in members
        }
        missing_members = [
            member for member in members if not source_files[str(member["path"])].exists()
        ]
        if missing_members:
            if self.archive_path is None or not self.archive_path.exists():
                missing_paths = ", ".join(str(member["path"]) for member in missing_members[:3])
                raise FileNotFoundError(
                    "extracted CADEvolve source cache is incomplete and archive is unavailable: "
                    f"{missing_paths}"
                )
            cache_root.mkdir(parents=True, exist_ok=True)
            with self.archive_path.open("rb") as raw_archive:
                for member in missing_members:
                    target = source_files[str(member["path"])]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = target.with_name(f"{target.name}.tmp")
                    tmp_path.write_bytes(_read_indexed_member(raw_archive, member))
                    tmp_path.replace(target)

        if self.archive_path is not None and self.archive_path.exists():
            _write_extracted_manifest(cache_root, self.archive_path, members, offset=offset, complete=complete)
        return source_files


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


def _source_object_record(
    source_path: str,
    code: str,
    index: int,
    config_hash: str,
) -> SourceObjectRecord:
    source_hash = sha256_text(code)
    normalized_code = _wrap_as_object_function(code)
    object_hash = sha256_json(
        {
            "source": "cadevolve",
            "source_path": source_path,
            "normalized_code": normalized_code,
        }
    )
    return SourceObjectRecord(
        object_id=f"obj_cadevolve_{index:06d}",
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
            config_hash=config_hash,
            prompt_hash=None,
        ),
    )


def _extracted_member_path(cache_root: Path, source_path: str) -> Path:
    normalized = _normalized_path(source_path)
    relative_path = Path(normalized)
    if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError(f"unsafe CADEvolve member path: {source_path}")
    return cache_root / relative_path


def _write_extracted_manifest(
    cache_root: Path,
    archive_path: Path,
    members: list[dict[str, Any]],
    *,
    offset: int,
    complete: bool,
) -> None:
    manifest_path = cache_root / "extraction_manifest.json"
    existing = _read_extracted_manifest(cache_root, archive_path) or {}
    members_by_path = {
        str(member["path"]): {
            "path": str(member["path"]),
            "size": int(member["size"]),
        }
        for member in existing.get("members", [])
    }
    members_by_path.update(
        {
            str(member["path"]): {
                "path": str(member["path"]),
                "size": int(member["size"]),
            }
            for member in members
        }
    )
    existing_prefix_count = int(existing.get("prefix_count", 0))
    prefix_count = existing_prefix_count
    if offset <= existing_prefix_count:
        prefix_count = max(existing_prefix_count, offset + len(members))
    complete = complete or existing.get("complete") is True

    tmp_path = manifest_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(
            {
                "schema": "intersectionqa_cadevolve_extracted_sources_v1",
                "archive": _archive_fingerprint(archive_path),
                "prefix_count": prefix_count,
                "complete": complete,
                "members": sorted(members_by_path.values(), key=lambda item: item["path"]),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(manifest_path)


def _read_extracted_manifest(cache_root: Path, archive_path: Path | None) -> dict[str, Any] | None:
    manifest_path = cache_root / "extraction_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if archive_path is not None and manifest.get("archive") != _archive_fingerprint(archive_path):
        return None
    if manifest.get("schema") != "intersectionqa_cadevolve_extracted_sources_v1":
        return None
    return manifest


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
