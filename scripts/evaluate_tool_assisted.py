"""Evaluate the exact-verifier tool-assisted upper bound."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.evaluation.tool_assisted import run_tool_assisted_upper_bound
from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import validate_dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()

    configure_logging()
    rows = validate_dataset_dir(args.dataset_dir)
    result = run_tool_assisted_upper_bound(rows)
    print(json.dumps(result.report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
