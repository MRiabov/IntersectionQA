"""CADEvolve tar archive source loader skeleton.

The MVP does not execute untrusted CADEvolve programs in-process. This loader
captures deterministic provenance and normalized source records for later
isolated validation. Missing archives are treated as an empty source rather than
blocking synthetic fixture smoke generation.
"""

from __future__ import annotations

import re
import tarfile
from pathlib import Path

from intersectionqa.hashing import sha256_json, sha256_text
from intersectionqa.schema import Hashes, SourceObjectRecord
from intersectionqa.sources.base import SourceLoadResult

EXECUTABLE_PREFIXES = ("CADEvolve-P/", "CADEvolve-C/")
RESULT_NAMES = ("result", "shape", "solid", "part")
OP_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)\(")


class CadevolveTarLoader:
    def __init__(self, archive_path: Path | None, config_hash: str) -> None:
        self.archive_path = archive_path
        self.config_hash = config_hash

    def load(self, limit: int | None = None) -> SourceLoadResult:
        if self.archive_path is None or not self.archive_path.exists():
            return SourceLoadResult(records=[], failures=[], scanned_count=0)

        records: list[SourceObjectRecord] = []
        scanned = 0
        with tarfile.open(self.archive_path, "r:*") as archive:
            members = sorted(
                (
                    member
                    for member in archive.getmembers()
                    if member.isfile()
                    and member.name.endswith(".py")
                    and _normalized_path(member.name).startswith(EXECUTABLE_PREFIXES)
                ),
                key=lambda member: _normalized_path(member.name),
            )
            for member in members:
                if limit is not None and len(records) >= limit:
                    break
                scanned += 1
                fileobj = archive.extractfile(member)
                if fileobj is None:
                    continue
                code = fileobj.read().decode("utf-8", errors="replace")
                source_path = _normalized_path(member.name)
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
        return SourceLoadResult(records=records, failures=[], scanned_count=scanned)


def _normalized_path(path: str) -> str:
    return path.removeprefix("./")


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
