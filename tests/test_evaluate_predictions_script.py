import json
import subprocess
import sys

from intersectionqa.config import DatasetConfig, SmokeConfig
from intersectionqa.enums import TaskType
from intersectionqa.pipeline import validate_dataset_dir, write_smoke_dataset
from scripts.evaluation.internal.predictions import read_predictions


def test_read_predictions_jsonl(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps({"id": "intersectionqa_binary_000001", "output": "yes"}) + "\n",
        encoding="utf-8",
    )
    predictions = read_predictions(path)
    assert len(predictions) == 1
    assert predictions[0].row_id == "intersectionqa_binary_000001"
    assert predictions[0].output == "yes"


def test_evaluate_predictions_script_handles_repair_direction_rows(tmp_path):
    dataset_dir = tmp_path / "dataset"
    write_smoke_dataset(
        DatasetConfig(
            output_dir=dataset_dir,
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                task_types=[
                    TaskType.BINARY_INTERFERENCE,
                    TaskType.REPAIR_DIRECTION,
                ],
            ),
        )
    )
    rows = validate_dataset_dir(dataset_dir)
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps({"id": row.id, "output": row.answer}) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluation.evaluate_predictions",
            str(dataset_dir),
            str(predictions_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metrics = json.loads(completed.stdout)

    repair_metrics = next(item for item in metrics if item["task_type"] == "repair_direction")
    assert repair_metrics["accuracy"] == 1.0
    assert repair_metrics["invalid_outputs"] == 0
