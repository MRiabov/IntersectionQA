"""Verify repair_direction predictions by applying moves and remeasuring geometry.

Prediction JSONL format:

{"id": "intersectionedit_repair_direction_000001", "output": "+x"}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from intersectionqa.evaluation.repair import verify_repair_predictions
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
    result = verify_repair_predictions(rows, predictions)
    payload = {
        "report": result.report,
        "results": [asdict(item) for item in result.results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
