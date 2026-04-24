from intersectionqa.config import DatasetConfig
from intersectionqa.evaluation.metrics import Prediction, dataset_stats, evaluate_predictions, manifest_stats
from intersectionqa.evaluation.parsing import parse_answer
from intersectionqa.pipeline import build_smoke_rows


def test_parse_answer_is_strict_by_task():
    assert parse_answer("binary_interference", "yes") == "yes"
    assert parse_answer("binary_interference", "yes\n") == "yes"
    assert parse_answer("binary_interference", "yes.") is None
    assert parse_answer("relation_classification", "near_miss") == "near_miss"
    assert parse_answer("volume_bucket", "(0.20, 0.50]") == "(0.20, 0.50]"


def test_evaluate_predictions_reports_invalid_outputs():
    rows, _ = build_smoke_rows(DatasetConfig())
    predictions = [
        Prediction(row_id=row.id, output=row.answer if index % 2 == 0 else "invalid prose")
        for index, row in enumerate(rows)
    ]
    metrics = evaluate_predictions(rows, predictions)
    assert {item.task_type for item in metrics} == {
        "binary_interference",
        "relation_classification",
        "volume_bucket",
    }
    assert any(item.invalid_outputs > 0 for item in metrics)
    assert all(0.0 <= item.accuracy <= 1.0 for item in metrics)


def test_dataset_stats_counts_rows():
    rows, _ = build_smoke_rows(DatasetConfig())
    stats = dataset_stats(rows)
    assert stats["total_rows"] == len(rows)
    assert stats["by_task"]["binary_interference"] == 7
    assert stats["by_task_answer"]["binary_interference"]["yes"] > 0
    assert stats["by_candidate_strategy"]["golden_box_fixture"] > 0
    assert stats["by_source_subtree"]["unknown"] == len(rows)
    assert sum(sum(counts.values()) for counts in stats["by_split_relation"].values()) == len(rows)


def test_manifest_stats_counts_validation_records():
    stats = manifest_stats([], [])
    assert stats["object_validation_records"] == 0
    assert stats["failure_manifest_records"] == 0
