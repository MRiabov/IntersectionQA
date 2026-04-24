"""Compare two generated dataset directories for reproducible release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import validate_dataset_dir
from intersectionqa.splits.grouped import DEFAULT_SPLITS

RELEASE_ARTIFACTS = [
    "metadata.json",
    "schema.json",
    "source_manifest.json",
    "split_manifest.json",
    "object_validation_manifest.jsonl",
    "failure_manifest.jsonl",
    *[f"{split}.jsonl" for split in DEFAULT_SPLITS],
]


def dataset_release_fingerprints(dataset_dir: Path) -> dict[str, str | None]:
    validate_dataset_dir(dataset_dir)
    return {name: _file_sha256(dataset_dir / name) for name in RELEASE_ARTIFACTS}


def compare_dataset_dirs(left: Path, right: Path) -> dict[str, Any]:
    left_fingerprints = dataset_release_fingerprints(left)
    right_fingerprints = dataset_release_fingerprints(right)
    mismatches = [
        {
            "artifact": name,
            "left": left_fingerprints.get(name),
            "right": right_fingerprints.get(name),
        }
        for name in RELEASE_ARTIFACTS
        if left_fingerprints.get(name) != right_fingerprints.get(name)
    ]
    return {
        "status": "pass" if not mismatches else "fail",
        "left": str(left),
        "right": str(right),
        "artifact_count": len(RELEASE_ARTIFACTS),
        "mismatches": mismatches,
    }


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()

    configure_logging()
    result = compare_dataset_dirs(args.left, args.right)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
