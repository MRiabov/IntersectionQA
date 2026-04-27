"""Validate exported IntersectionQA JSONL rows and split leakage."""

from __future__ import annotations

import argparse
from pathlib import Path

from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import validate_dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()

    configure_logging()
    rows = validate_dataset_dir(args.dataset_dir)
    print(f"validated_rows={len(rows)}")


if __name__ == "__main__":
    main()
