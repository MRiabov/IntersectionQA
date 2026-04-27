"""Inspect one exported dataset row by ID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import validate_dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("row_id")
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    configure_logging()
    rows = validate_dataset_dir(args.dataset_dir)
    matches = [row for row in rows if row.id == args.row_id]
    if not matches:
        raise SystemExit(f"row not found: {args.row_id}")
    row = matches[0]
    data = row.model_dump(mode="json")
    if not args.show_prompt:
        data["prompt"] = f"<hidden; {len(row.prompt)} chars>"
        data["script"] = f"<hidden; {len(row.script)} chars>"
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
