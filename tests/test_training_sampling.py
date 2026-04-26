from collections import Counter

from intersectionqa.training.sampling import select_diverse_low_reward_samples, select_rows


def test_stratified_task_sampling_covers_small_task_groups():
    rows = (
        [{"task_type": "qa", "id": f"qa_{index}"} for index in range(30)]
        + [{"task_type": "repair", "id": f"repair_{index}"} for index in range(4)]
        + [{"task_type": "move", "id": f"move_{index}"} for index in range(4)]
    )

    selected = select_rows(
        rows,
        limit=9,
        seed=7,
        strategy="stratified_task",
        key=lambda row: row["task_type"],
    )
    counts = Counter(row["task_type"] for row in selected)

    assert counts == {"qa": 3, "repair": 3, "move": 3}


def test_random_sampling_preserves_previous_cap_behavior():
    rows = [{"task_type": "qa", "id": str(index)} for index in range(20)]

    selected = select_rows(
        rows,
        limit=5,
        seed=7,
        strategy="random",
        key=lambda row: row["task_type"],
    )

    assert len(selected) == 5
    assert {row["id"] for row in selected} != {str(index) for index in range(5)}


def test_debug_sample_selection_prioritizes_low_reward_task_diversity():
    samples = [
        {"row_id": "easy_a", "task_type": "a", "reward": 1.0},
        {"row_id": "bad_a", "task_type": "a", "reward": 0.05},
        {"row_id": "bad_b", "task_type": "b", "reward": 0.0},
        {"row_id": "mid_b", "task_type": "b", "reward": 0.4},
        {"row_id": "bad_c", "task_type": "c", "reward": 0.2},
    ]

    selected = select_diverse_low_reward_samples(samples, 3)

    assert [sample["task_type"] for sample in selected] == ["b", "a", "c"]
    assert [sample["row_id"] for sample in selected] == ["bad_b", "bad_a", "bad_c"]
