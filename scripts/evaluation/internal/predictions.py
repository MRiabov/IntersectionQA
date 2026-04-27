"""Shared prediction JSONL readers for evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path

from intersectionqa.evaluation.metrics import Prediction


def read_predictions(path: Path) -> list[Prediction]:
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
