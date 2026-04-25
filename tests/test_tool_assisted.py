import json
import subprocess
import sys

from intersectionqa.config import DatasetConfig, SmokeConfig
from intersectionqa.enums import TaskType
from intersectionqa.evaluation.metrics import Prediction
from intersectionqa.evaluation.repair import verify_repair_direction_output, verify_repair_predictions
from intersectionqa.evaluation.tool_assisted import (
    TOOL_ASSISTED_EVAL_VERSION,
    run_tool_assisted_upper_bound,
    tool_assisted_predict,
)
from intersectionqa.pipeline import build_smoke_rows
from intersectionqa.pipeline import write_smoke_dataset


def test_tool_assisted_upper_bound_executes_rows_and_reports_failures():
    rows, _ = build_smoke_rows(DatasetConfig())
    result = run_tool_assisted_upper_bound(rows)

    assert result.report["eval_version"] == TOOL_ASSISTED_EVAL_VERSION
    assert result.report["system"] == "tool_assisted_exact_verifier_upper_bound"
    assert result.report["tool_failure_count"] == 0
    assert result.metrics
    assert all(metric.accuracy == 1.0 for metric in result.metrics)


def test_tool_assisted_prediction_matches_exact_answer():
    rows, _ = build_smoke_rows(DatasetConfig())
    row = rows[0]

    prediction = tool_assisted_predict(row)

    assert prediction.status == "ok"
    assert prediction.output == row.answer


def test_tool_assisted_upper_bound_verifies_repair_direction_rows():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                task_types=[
                    TaskType.BINARY_INTERFERENCE,
                    TaskType.REPAIR_DIRECTION,
                ],
            )
        )
    )
    repair_rows = [row for row in rows if row.task_type == TaskType.REPAIR_DIRECTION]
    result = run_tool_assisted_upper_bound(rows)

    assert repair_rows
    assert result.report["tool_failure_count"] == 0
    assert all(metric.accuracy == 1.0 for metric in result.metrics)

    verification = verify_repair_direction_output(repair_rows[0], repair_rows[0].answer)
    assert verification.valid_output
    assert verification.repaired
    assert verification.relation_after_move not in {"intersecting", "contained"}

    invalid = verify_repair_direction_output(repair_rows[0], "no_valid_move")
    assert not invalid.valid_output
    assert not invalid.repaired


def test_repair_verifier_distinguishes_repair_success_from_exact_answer():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                task_types=[TaskType.REPAIR_DIRECTION],
            )
        )
    )
    predictions = [Prediction(row_id=row.id, output="+y") for row in rows]

    result = verify_repair_predictions(rows, predictions)

    assert result.report["row_count"] == len(rows)
    assert result.report["repair_success_rate"] == 1.0
    assert result.report["exact_answer_accuracy"] < result.report["repair_success_rate"]
    assert result.report["by_output"]["+y"]["total"] == len(rows)
    assert result.report["by_output"]["+y"]["repaired"] == len(rows)


def test_evaluate_tool_assisted_script_handles_repair_direction_rows(tmp_path):
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

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_tool_assisted",
            str(dataset_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["tool_failure_count"] == 0
    assert any(metric["task_type"] == "repair_direction" for metric in report["metrics"])
