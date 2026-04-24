from intersectionqa.config import DatasetConfig
from intersectionqa.evaluation.aabb import evaluate_aabb_binary
from intersectionqa.pipeline import build_smoke_rows


def test_aabb_baseline_runs_on_binary_rows():
    rows, _ = build_smoke_rows(DatasetConfig())
    result = evaluate_aabb_binary(rows)
    assert result.total > 0
    assert 0.0 <= result.accuracy <= 1.0
