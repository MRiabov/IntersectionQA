"""Source loader interfaces and source-level failure reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from intersectionqa.schema import FailureRecord, SourceObjectRecord


@dataclass(frozen=True)
class SourceLoadResult:
    records: list[SourceObjectRecord]
    failures: list[FailureRecord]
    scanned_count: int = 0


class SourceLoader(Protocol):
    def load(self, limit: int | None = None) -> SourceLoadResult:
        ...


def missing_archive_result(path: Path | None) -> SourceLoadResult:
    reason = "CADEvolve archive was not configured." if path is None else f"Archive not found: {path}"
    return SourceLoadResult(records=[], failures=[], scanned_count=0)
