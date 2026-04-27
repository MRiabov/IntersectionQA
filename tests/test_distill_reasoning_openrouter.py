from __future__ import annotations

import json

from scripts.training.distill_reasoning_openrouter import _is_valid_ranking_answer, _sample_rows


def test_sample_rows_round_robin_balances_tasks(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "train.jsonl").write_text(
        "\n".join(
            json.dumps({"id": row_id, "task_type": task_type})
            for row_id, task_type in [
                ("z1", "zeta"),
                ("a1", "alpha"),
                ("z2", "zeta"),
                ("a2", "alpha"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _sample_rows(
        dataset_dir,
        splits=["train"],
        task_types={"alpha", "zeta"},
        max_rows=10,
        rows_per_task=1,
        seed=7,
    )

    assert [row["task_type"] for row in rows] == ["alpha", "zeta"]
    assert len(rows) == 2
    assert {row["id"] for row in rows} <= {"a1", "a2", "z1", "z2"}


def test_ranking_answer_validation_accepts_unique_abcd_permutations():
    assert _is_valid_ranking_answer("ABC")
    assert _is_valid_ranking_answer("ABCD")
    assert _is_valid_ranking_answer("ABCDE")
    assert not _is_valid_ranking_answer("AAB")
    assert not _is_valid_ranking_answer("AB")
    assert not _is_valid_ranking_answer("ABCDEF")
    assert not _is_valid_ranking_answer("ABCF")
