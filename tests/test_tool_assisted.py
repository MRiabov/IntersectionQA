from intersectionqa.config import DatasetConfig
from intersectionqa.evaluation.tool_assisted import (
    TOOL_ASSISTED_EVAL_VERSION,
    run_tool_assisted_upper_bound,
    tool_assisted_predict,
)
from intersectionqa.pipeline import build_smoke_rows


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
