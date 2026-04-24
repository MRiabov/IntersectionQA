"""Summarize generation failures and optional prediction failure cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.evaluation.failure_analysis import failure_case_analysis
from intersectionqa.export.jsonl import read_failure_manifest, read_object_validation_manifest
from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import validate_dataset_dir
from scripts.evaluate_predictions import _read_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--predictions-jsonl", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    configure_logging()
    rows = validate_dataset_dir(args.dataset_dir)
    predictions = _read_predictions(args.predictions_jsonl) if args.predictions_jsonl else None
    report = failure_case_analysis(
        rows,
        read_object_validation_manifest(args.dataset_dir / "object_validation_manifest.jsonl"),
        read_failure_manifest(args.dataset_dir / "failure_manifest.jsonl"),
        predictions=predictions,
        max_examples=args.max_examples,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
