from intersectionqa.config import DatasetConfig, SmokeConfig
from intersectionqa.enums import TaskType
from intersectionqa.evaluation.rewards import reward_prediction
from intersectionqa.pipeline import build_smoke_rows


def test_axis_repair_reward_gives_partial_numeric_credit():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                geometry_limit=3,
                task_types=[TaskType.AXIS_ALIGNED_REPAIR],
            )
        )
    )
    row = rows[0]
    exact = reward_prediction(row, row.answer)
    wrong_distance = reward_prediction(
        row,
        row.answer.replace("distance_mm=", "distance_mm=9", 1)
        if "distance_mm=0" in row.answer
        else "direction=+x, distance_mm=9.9",
    )
    invalid = reward_prediction(row, "direction=+x, distance_mm=1.23")

    assert exact.reward == 1.0
    assert exact.components["within_tolerance"] == 1.0
    assert 0.0 <= wrong_distance.reward < exact.reward
    assert invalid.reward == 0.0
    assert invalid.failure_reason == "invalid_output"


def test_rewards_accept_reasoning_answer_tags():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                geometry_limit=3,
                task_types=[TaskType.AXIS_ALIGNED_REPAIR],
            )
        )
    )
    row = rows[0]
    result = reward_prediction(row, f"<think>Use the constrained edit policy.</think><answer>{row.answer}</answer>")

    assert result.reward == 1.0
    assert result.parsed_output == row.answer
    assert result.components["answer_tag"] == 1.0
    assert result.components["reasoning_format"] == 1.0


def test_axis_repair_vector_and_program_rewards_use_vector_error():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                geometry_limit=3,
                task_types=[
                    TaskType.AXIS_ALIGNED_REPAIR_VECTOR,
                    TaskType.AXIS_ALIGNED_REPAIR_PROGRAM,
                ],
            )
        )
    )

    assert rows
    for row in rows:
        exact = reward_prediction(row, row.answer)
        wrong = reward_prediction(
            row,
            "object_b = object_b.translate((9.9, 0.0, 0.0))"
            if row.task_type == TaskType.AXIS_ALIGNED_REPAIR_PROGRAM
            else "dx=9.9, dy=0.0, dz=0.0",
        )
        assert exact.reward == 1.0
        assert exact.components["within_tolerance"] == 1.0
        assert 0.0 <= wrong.reward < exact.reward


def test_candidate_selection_reward_scores_valid_non_best_candidates():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                geometry_limit=3,
                task_types=[TaskType.EDIT_CANDIDATE_SELECTION],
            )
        )
    )
    row = rows[0]
    best = reward_prediction(row, row.answer)
    alternatives = [
        reward_prediction(row, label)
        for label in "ABCD"
        if label != row.answer
    ]

    assert best.reward == 1.0
    assert any(0.0 < item.reward < 1.0 for item in alternatives)


def test_target_clearance_move_reward_gives_signed_numeric_credit():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                geometry_limit=7,
                task_types=[TaskType.TARGET_CLEARANCE_MOVE],
            )
        )
    )
    row = rows[0]
    exact = reward_prediction(row, row.answer)
    wrong = reward_prediction(row, "distance_mm=9.9")
    invalid = reward_prediction(row, "distance_mm=9.99")

    assert exact.reward == 1.0
    assert exact.components["within_tolerance"] == 1.0
    assert 0.0 <= wrong.reward < exact.reward
    assert invalid.reward == 0.0


def test_target_contact_move_reward_uses_signed_distance():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                geometry_limit=7,
                task_types=[TaskType.TARGET_CONTACT_MOVE],
            )
        )
    )
    row = rows[0]
    exact = reward_prediction(row, row.answer)
    wrong = reward_prediction(row, "distance_mm=9.9")

    assert exact.reward == 1.0
    assert exact.components["within_tolerance"] == 1.0
    assert 0.0 <= wrong.reward < exact.reward


def test_centroid_distance_move_reward_uses_signed_distance():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                geometry_limit=7,
                task_types=[TaskType.CENTROID_DISTANCE_MOVE],
            )
        )
    )
    row = rows[0]
    exact = reward_prediction(row, row.answer)
    wrong = reward_prediction(row, "distance_mm=9.9")

    assert exact.reward == 1.0
    assert exact.components["within_tolerance"] == 1.0
    assert 0.0 <= wrong.reward < exact.reward


def test_candidate_ranking_reward_uses_pairwise_partial_credit():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                geometry_limit=3,
                task_types=[TaskType.EDIT_CANDIDATE_RANKING],
            )
        )
    )
    row = rows[0]
    exact = reward_prediction(row, row.answer)
    reversed_order = reward_prediction(row, row.answer[::-1])

    assert exact.reward == 1.0
    assert exact.components["pairwise_ranking"] == 1.0
    assert 0.0 < reversed_order.reward < 1.0
