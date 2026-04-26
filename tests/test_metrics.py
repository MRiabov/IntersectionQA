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


def test_evaluate_predictions_accepts_reasoning_answer_tags():
    rows, _ = build_smoke_rows(_synthetic_config())
    predictions = [
        Prediction(row_id=row.id, output=f"<think>Check the geometry.</think><answer>{row.answer}</answer>")
        for row in rows
    ]

    metrics = evaluate_predictions(rows, predictions)

    assert metrics
    assert all(item.invalid_outputs == 0 for item in metrics)
    assert all(item.accuracy == 1.0 for item in metrics)


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


def test_epic15_metrics_report_numeric_and_candidate_scores():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                geometry_limit=3,
                task_types=[
                    TaskType.AXIS_ALIGNED_REPAIR,
                    TaskType.AXIS_ALIGNED_REPAIR_VECTOR,
                    TaskType.AXIS_ALIGNED_REPAIR_PROGRAM,
                    TaskType.TARGET_CLEARANCE_REPAIR,
                    TaskType.TARGET_CLEARANCE_MOVE,
                    TaskType.TARGET_CONTACT_MOVE,
                    TaskType.CENTROID_DISTANCE_MOVE,
                    TaskType.EDIT_CANDIDATE_SELECTION,
                    TaskType.EDIT_CANDIDATE_RANKING,
                ],
            )
        )
    )
    predictions = [Prediction(row_id=row.id, output=row.answer) for row in rows]

    metrics = {metric.task_type: metric for metric in evaluate_predictions(rows, predictions)}
    stats = dataset_stats(rows)

    assert metrics[TaskType.AXIS_ALIGNED_REPAIR].within_tolerance_accuracy == 1.0
    assert metrics[TaskType.AXIS_ALIGNED_REPAIR_VECTOR].numeric_mae_mm == 0.0
    assert metrics[TaskType.AXIS_ALIGNED_REPAIR_PROGRAM].within_tolerance_accuracy == 1.0
    assert metrics[TaskType.TARGET_CLEARANCE_REPAIR].numeric_mae_mm == 0.0
    assert metrics[TaskType.TARGET_CLEARANCE_MOVE].within_tolerance_accuracy == 1.0
    assert metrics[TaskType.TARGET_CONTACT_MOVE].within_tolerance_accuracy == 1.0
    assert metrics[TaskType.CENTROID_DISTANCE_MOVE].within_tolerance_accuracy == 1.0
    assert metrics[TaskType.EDIT_CANDIDATE_RANKING].pairwise_ranking_accuracy == 1.0
    assert stats["axis_aligned_repair"]["row_count"] > 0
    assert stats["axis_aligned_repair_vector"]["row_count"] > 0
    assert stats["axis_aligned_repair_program"]["row_count"] > 0
    assert stats["by_edit_difficulty_tag"]["axis_aligned_intersection_repair_vector"] > 0
    assert stats["by_edit_difficulty_tag"]["output_edit_program"] > 0
    assert stats["by_edit_difficulty_tag"]["centroid_distance_move"] > 0
    assert stats["target_clearance_repair"]["row_count"] > 0
    assert stats["target_clearance_move"]["row_count"] > 0
    assert stats["target_contact_move"]["row_count"] > 0
    assert stats["centroid_distance_move"]["row_count"] > 0
    assert stats["edit_candidate_selection"]["row_count"] > 0
    assert stats["edit_candidate_ranking"]["row_count"] > 0


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


def test_failure_case_analysis_summarizes_intersectionedit_failure_modes():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                geometry_limit=3,
                task_types=[
                    TaskType.AXIS_ALIGNED_REPAIR,
                    TaskType.AXIS_ALIGNED_REPAIR_VECTOR,
                    TaskType.AXIS_ALIGNED_REPAIR_PROGRAM,
                    TaskType.TARGET_CLEARANCE_MOVE,
                    TaskType.CENTROID_DISTANCE_MOVE,
                    TaskType.EDIT_CANDIDATE_SELECTION,
                    TaskType.EDIT_CANDIDATE_RANKING,
                ],
            )
        )
    )
    predictions = []
    for row in rows:
        if row.task_type == TaskType.AXIS_ALIGNED_REPAIR:
            predictions.append(Prediction(row_id=row.id, output="direction=-z, distance_mm=9.9"))
        elif row.task_type == TaskType.AXIS_ALIGNED_REPAIR_VECTOR:
            predictions.append(Prediction(row_id=row.id, output="dx=9.9, dy=0.0, dz=0.0"))
        elif row.task_type == TaskType.AXIS_ALIGNED_REPAIR_PROGRAM:
            predictions.append(Prediction(row_id=row.id, output="object_b = object_b.translate((9.9, 0.0, 0.0))"))
        elif row.task_type == TaskType.EDIT_CANDIDATE_SELECTION:
            predictions.append(Prediction(row_id=row.id, output="A" if row.answer != "A" else "B"))
        elif row.task_type == TaskType.EDIT_CANDIDATE_RANKING:
            predictions.append(Prediction(row_id=row.id, output=row.answer[::-1]))
        else:
            predictions.append(Prediction(row_id=row.id, output="distance_mm=9.9"))

    report = failure_case_analysis(rows, [], [], predictions=predictions, max_examples=4)

    edit_report = report["intersectionedit_prediction_failures"]
    categories = edit_report["by_category"]
    assert edit_report["row_count"] == len(rows)
    assert categories["wrong_direction"] > 0
    assert categories["excessive_movement"] > 0
    assert categories["centroid_distance_error"] > 0
    assert categories["candidate_ranking_error"] > 0
    assert len(edit_report["examples"]) == 4
