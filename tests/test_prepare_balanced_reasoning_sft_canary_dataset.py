from __future__ import annotations

from collections import Counter

from scripts.training.prepare_balanced_reasoning_sft_canary_dataset import balanced_train_sample


def test_balanced_train_sample_caps_majority_and_upsamples_minority() -> None:
    rows = [
        {"id": f"yes-{index}", "task_type": "binary_interference", "answer": "yes"}
        for index in range(2)
    ] + [
        {"id": f"no-{index}", "task_type": "binary_interference", "answer": "no"}
        for index in range(10)
    ]

    sampled, report = balanced_train_sample(
        rows,
        seed=1,
        target_per_answer=4,
        max_repeat_per_row=3,
    )

    counts = Counter(row["answer"] for row in sampled)
    assert counts == {"yes": 4, "no": 4}
    assert report["source_row_count"] == 12
    assert report["sampled_row_count"] == 8


def test_balanced_train_sample_respects_repeat_cap_for_singletons() -> None:
    rows = [{"id": "rare", "task_type": "volume_bucket", "answer": ">0.50"}]

    sampled, report = balanced_train_sample(
        rows,
        seed=1,
        target_per_answer=10,
        max_repeat_per_row=3,
    )

    assert len(sampled) == 3
    assert report["strata"][0]["max_possible_with_repeat_cap"] == 3
