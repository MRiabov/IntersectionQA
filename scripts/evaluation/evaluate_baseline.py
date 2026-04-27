"""Evaluate binary-interference geometry baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.evaluation.aabb import evaluate_aabb_binary
from intersectionqa.evaluation.obb import evaluate_obb_binary
from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import validate_dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--baseline", choices=["aabb", "obb"], default="aabb")
    args = parser.parse_args()

    configure_logging()
    rows = validate_dataset_dir(args.dataset_dir)
    result = evaluate_obb_binary(rows) if args.baseline == "obb" else evaluate_aabb_binary(rows)
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
