"""Evaluate saved model outputs against exported IntersectionQA rows.

Prediction JSONL format:

{"id": "intersectionqa_binary_000001", "output": "yes"}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.evaluation.metrics import evaluate_predictions
from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import validate_dataset_dir
from scripts.evaluation.internal.predictions import read_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("predictions_jsonl", type=Path)
    args = parser.parse_args()

    configure_logging()
    rows = validate_dataset_dir(args.dataset_dir)
    predictions = read_predictions(args.predictions_jsonl)
    metrics = evaluate_predictions(rows, predictions)
    print(json.dumps([metric.__dict__ for metric in metrics], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
