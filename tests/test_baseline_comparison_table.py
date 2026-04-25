import json
import subprocess
import sys

from intersectionqa.config import DatasetConfig, SmokeConfig
from intersectionqa.enums import TaskType
from intersectionqa.pipeline import validate_dataset_dir, write_smoke_dataset


def test_baseline_comparison_table_writes_json_and_markdown(tmp_path):
    dataset_dir = tmp_path / "dataset"
    write_smoke_dataset(
        DatasetConfig(
            output_dir=dataset_dir,
            smoke=SmokeConfig(use_synthetic_fixtures=True, include_cadevolve_if_available=False),
        )
    )
    rows = validate_dataset_dir(dataset_dir)
    predictions_path = tmp_path / "perfect_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"id": row.id, "output": row.answer}, sort_keys=True) + "\n")

    json_output = tmp_path / "comparison.json"
    markdown_output = tmp_path / "comparison.md"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.baseline_comparison_table",
            str(dataset_dir),
            "--prediction",
            f"perfect={predictions_path}",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert any(
        row["system"] == "aabb_overlap" and row["task_type"] == "binary_interference"
        for row in payload
    )
    assert any(
        row["system"] == "obb_overlap" and row["task_type"] == "binary_interference"
        for row in payload
    )
    assert any(row["system"] == "perfect" and row["accuracy"] == 1.0 for row in payload)
    markdown = markdown_output.read_text(encoding="utf-8")
    assert "| system | task_type |" in markdown
    assert "perfect" in markdown


def test_baseline_comparison_table_includes_repair_verifier_rows(tmp_path):
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
    predictions_path = tmp_path / "perfect_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"id": row.id, "output": row.answer}, sort_keys=True) + "\n")

    json_output = tmp_path / "comparison.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.baseline_comparison_table",
            str(dataset_dir),
            "--prediction",
            f"perfect={predictions_path}",
            "--json-output",
            str(json_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert any(
        row["system"] == "perfect_repair_verifier"
        and row["task_type"] == "repair_direction"
        and row["accuracy"] == 1.0
        for row in payload
    )
