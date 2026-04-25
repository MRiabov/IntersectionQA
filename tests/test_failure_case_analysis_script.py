import json
import subprocess
import sys

from intersectionqa.config import DatasetConfig, SmokeConfig
from intersectionqa.enums import TaskType
from intersectionqa.pipeline import validate_dataset_dir, write_smoke_dataset


def test_failure_case_analysis_script_writes_prediction_failures(tmp_path):
    dataset_dir = tmp_path / "dataset"
    write_smoke_dataset(
        DatasetConfig(
            output_dir=dataset_dir,
            smoke=SmokeConfig(use_synthetic_fixtures=True, include_cadevolve_if_available=False),
        )
    )
    rows = validate_dataset_dir(dataset_dir)
    predictions_path = tmp_path / "bad_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"id": row.id, "output": "invalid prose"}) + "\n")

    output_path = tmp_path / "failure_analysis.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.failure_case_analysis",
            str(dataset_dir),
            "--predictions-jsonl",
            str(predictions_path),
            "--max-examples",
            "2",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["summary"]["rows"] == len(rows)
    assert report["prediction_failures"]["invalid_output_count"] == len(rows)
    assert len(report["prediction_failures"]["examples"]) == 2


def test_failure_case_analysis_script_includes_repair_verifier(tmp_path):
    dataset_dir = tmp_path / "dataset"
    write_smoke_dataset(
        DatasetConfig(
            output_dir=dataset_dir,
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                task_types=[TaskType.REPAIR_DIRECTION],
            ),
        )
    )
    rows = validate_dataset_dir(dataset_dir)
    predictions_path = tmp_path / "bad_repair_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"id": row.id, "output": "no_valid_move"}) + "\n")

    output_path = tmp_path / "failure_analysis.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.failure_case_analysis",
            str(dataset_dir),
            "--predictions-jsonl",
            str(predictions_path),
            "--max-examples",
            "2",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    verifier = report["repair_prediction_verifier"]
    assert verifier["row_count"] == len(rows)
    assert verifier["invalid_output_count"] == len(rows)
    assert verifier["repair_success_rate"] == 0.0
    assert len(verifier["failed_examples"]) == 2
