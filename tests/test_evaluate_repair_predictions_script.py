import json
import subprocess
import sys

from intersectionqa.config import DatasetConfig, SmokeConfig
from intersectionqa.enums import TaskType
from intersectionqa.pipeline import validate_dataset_dir, write_smoke_dataset


def test_evaluate_repair_predictions_script_reports_exact_repair_success(tmp_path):
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
    repair_rows = [row for row in rows if row.task_type == TaskType.REPAIR_DIRECTION]
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps({"id": row.id, "output": row.answer}) + "\n"
            for row in repair_rows
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluation.evaluate_repair_predictions",
            str(dataset_dir),
            str(predictions_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["report"]["system"] == "intersectionedit_repair_direction_exact_verifier"
    assert payload["report"]["row_count"] == len(repair_rows)
    assert payload["report"]["repair_success_rate"] == 1.0
    assert payload["report"]["exact_answer_accuracy"] == 1.0
    assert payload["report"]["by_output"]
    assert all(item["repaired"] for item in payload["results"])
