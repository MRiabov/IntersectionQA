"""Apply deterministic class balancing to an exported IntersectionQA dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.export.balance import balance_dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/IntersectionQA-90K"))
    parser.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = balance_dataset_dir(args.dataset_dir, backup=args.backup, dry_run=args.dry_run)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
