"""Build JSON/Markdown comparison tables for baselines and saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.evaluation.aabb import evaluate_aabb_binary
from intersectionqa.evaluation.comparison import (
    comparison_rows_from_aabb,
    comparison_rows_from_metrics,
    comparison_rows_to_markdown,
    sort_comparison_rows,
)
from intersectionqa.evaluation.metrics import evaluate_predictions
from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import validate_dataset_dir
from scripts.evaluate_predictions import _read_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="SYSTEM=PATH",
        help="Add a saved prediction JSONL under a system name.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    configure_logging()
    rows = validate_dataset_dir(args.dataset_dir)
    comparison_rows = comparison_rows_from_aabb(evaluate_aabb_binary(rows))
    for system, path in [_parse_named_path(item) for item in args.prediction]:
        metrics = evaluate_predictions(rows, _read_predictions(path))
        comparison_rows.extend(comparison_rows_from_metrics(metrics, system=system))

    comparison_rows = sort_comparison_rows(comparison_rows)
    payload = [row.as_dict() for row in comparison_rows]
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    markdown_text = comparison_rows_to_markdown(comparison_rows)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json_text, encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown_text, encoding="utf-8")
    if args.json_output is None and args.markdown_output is None:
        print(json_text, end="")


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--prediction must use SYSTEM=PATH")
    system, raw_path = value.split("=", 1)
    system = system.strip()
    if not system:
        raise ValueError("--prediction system name must not be empty")
    return system, Path(raw_path)


if __name__ == "__main__":
    main()
