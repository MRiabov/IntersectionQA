from intersectionqa.config import DatasetConfig, SmokeConfig
from intersectionqa.enums import TaskType
from intersectionqa.evaluation.failure_analysis import failure_case_analysis
from intersectionqa.evaluation.metrics import Prediction, dataset_stats, evaluate_predictions, manifest_stats
from intersectionqa.evaluation.parsing import parse_answer
from intersectionqa.pipeline import build_smoke_rows


def _synthetic_config() -> DatasetConfig:
    return DatasetConfig(smoke=SmokeConfig(include_cadevolve_if_available=False))


def test_parse_answer_is_strict_by_task():
    assert parse_answer("binary_interference", "yes") == "yes"
    assert parse_answer("binary_interference", "yes\n") == "yes"
    assert parse_answer("binary_interference", "yes.") is None
    assert parse_answer("relation_classification", "near_miss") == "near_miss"
    assert parse_answer("volume_bucket", "(0.20, 0.50]") == "(0.20, 0.50]"


def test_evaluate_predictions_reports_invalid_outputs():
    rows, _ = build_smoke_rows(_synthetic_config())
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
    rows, _ = build_smoke_rows(_synthetic_config())
    stats = dataset_stats(rows)
    assert stats["total_rows"] == len(rows)
    assert stats["by_task"]["binary_interference"] == 7
    assert stats["by_task_answer"]["binary_interference"]["yes"] > 0
    assert stats["by_candidate_strategy"]["golden_box_fixture"] > 0
    assert stats["by_source_subtree"]["unknown"] == len(rows)
    assert sum(sum(counts.values()) for counts in stats["by_split_relation"].values()) == len(rows)
    assert stats["repair_direction"]["row_count"] == 0


def test_dataset_stats_summarizes_repair_direction_rows():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                task_types=[TaskType.REPAIR_DIRECTION],
            )
        )
    )

    stats = dataset_stats(rows)
    repair_stats = stats["repair_direction"]

    assert repair_stats["row_count"] == len(rows)
    assert repair_stats["by_policy"] == {
        "conservative_aabb_separating_translation_v01": len(rows)
    }
    assert sum(repair_stats["by_selected_direction"].values()) == len(rows)
    assert repair_stats["selected_magnitude_mm"]["min"] is not None
    assert repair_stats["selected_magnitude_mm"]["max"] >= repair_stats["selected_magnitude_mm"]["min"]


def test_manifest_stats_counts_validation_records():
    stats = manifest_stats([], [])
    assert stats["object_validation_records"] == 0
    assert stats["failure_manifest_records"] == 0


def test_failure_case_analysis_counts_prediction_failures():
    rows, _ = build_smoke_rows(_synthetic_config())
    predictions = [
        Prediction(row_id=row.id, output=row.answer if index % 2 == 0 else "invalid prose")
        for index, row in enumerate(rows)
    ]

    report = failure_case_analysis(rows, [], [], predictions=predictions, max_examples=3)

    prediction_failures = report["prediction_failures"]
    assert prediction_failures["incorrect_count"] > 0
    assert prediction_failures["invalid_output_count"] > 0
    assert len(prediction_failures["examples"]) == 3


def test_failure_case_analysis_summarizes_repair_verifier_failures():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                task_types=[TaskType.REPAIR_DIRECTION],
            )
        )
    )
    predictions = [Prediction(row_id=row.id, output="no_valid_move") for row in rows]

    report = failure_case_analysis(rows, [], [], predictions=predictions, max_examples=2)

    verifier = report["repair_prediction_verifier"]
    assert verifier["row_count"] == len(rows)
    assert verifier["invalid_output_count"] == len(rows)
    assert verifier["repair_success_rate"] == 0.0
    assert len(verifier["failed_examples"]) == 2
