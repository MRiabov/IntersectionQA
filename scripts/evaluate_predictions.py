"""Evaluate saved model outputs against exported IntersectionQA rows.

Prediction JSONL format:

{"id": "intersectionqa_binary_000001", "output": "yes"}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.evaluation.metrics import Prediction, evaluate_predictions
from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import validate_dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("predictions_jsonl", type=Path)
    args = parser.parse_args()

    configure_logging()
    rows = validate_dataset_dir(args.dataset_dir)
    predictions = _read_predictions(args.predictions_jsonl)
    metrics = evaluate_predictions(rows, predictions)
    print(json.dumps([metric.__dict__ for metric in metrics], indent=2, sort_keys=True))


def _read_predictions(path: Path) -> list[Prediction]:
    predictions: list[Prediction] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            row_id = data.get("id") or data.get("row_id")
            output = data.get("output") or data.get("raw_completion")
            if not isinstance(row_id, str) or not isinstance(output, str):
                raise ValueError(f"{path}:{line_number}: expected string id/row_id and output/raw_completion fields")
            predictions.append(Prediction(row_id=row_id, output=output))
    return predictions


if __name__ == "__main__":
    main()
