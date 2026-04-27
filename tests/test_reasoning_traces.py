import json

from intersectionqa.config import DatasetConfig, SmokeConfig
from intersectionqa.enums import TaskType
from intersectionqa.pipeline import build_smoke_rows
from intersectionqa.training.reasoning_traces import add_reasoning_target, reasoning_target_text, write_reasoning_sft_dataset
from scripts.training.prepare_rejection_sampled_reasoning_sft_dataset import accepted_reasoning_payload, rejection_reason


def test_reasoning_target_preserves_canonical_answer_for_repair_direction():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                geometry_limit=3,
                task_types=[TaskType.REPAIR_DIRECTION],
            )
        )
    )
    row = rows[0]

    payload = add_reasoning_target(row)

    assert payload["answer"] == row.answer
    assert payload["canonical_answer"] == row.answer
    assert payload["target_text"].startswith("<think>")
    assert payload["target_text"].endswith(f"<answer>{row.answer}</answer>")
    assert "smallest" in payload["target_text"]


def test_reasoning_target_uses_signed_distance_metadata():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                geometry_limit=3,
                task_types=[TaskType.TARGET_CLEARANCE_MOVE],
            )
        )
    )
    row = rows[0]

    target = reasoning_target_text(row)

    assert row.answer in target
    assert "signed" in target
    assert "<answer>distance_mm=" in target


def test_write_reasoning_sft_dataset_writes_report_and_target_text(tmp_path):
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                geometry_limit=3,
                task_types=[TaskType.BINARY_INTERFERENCE, TaskType.REPAIR_TRANSLATION],
            )
        )
    )
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    with (dataset_dir / "inner_train.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")

    output_dir = tmp_path / "reasoning"
    report = write_reasoning_sft_dataset(dataset_dir=dataset_dir, output_dir=output_dir, splits=["inner_train"])
    output_rows = [
        json.loads(line)
        for line in (output_dir / "inner_train.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert report["splits"]["inner_train"]["row_count"] == len(rows)
    assert (output_dir / "reasoning_sft_report.json").exists()
    assert all(row["answer"] == row["canonical_answer"] for row in output_rows)
    assert all(row["target_text"].startswith("<think>") for row in output_rows)


def test_rejection_sampled_reasoning_accepts_only_correct_reasoned_outputs():
    rows, _ = build_smoke_rows(
        DatasetConfig(
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                geometry_limit=3,
                task_types=[TaskType.BINARY_INTERFERENCE],
            )
        )
    )
    row = rows[0].model_dump(mode="json")
    accepted = f"<think>Use the relation label to decide if positive volume overlap exists.</think><answer>{row['answer']}</answer>"
    wrong = "<think>Use the relation label to decide if positive volume overlap exists.</think><answer>not_an_answer</answer>"
    trivial = f"<think>yes</think><answer>{row['answer']}</answer>"

    payload = accepted_reasoning_payload(row, accepted)

    assert payload is not None
    assert payload["target_text"] == accepted
    assert payload["canonical_answer"] == row["answer"]
    assert accepted_reasoning_payload(row, wrong) is None
    assert rejection_reason(row, wrong) == "answer_incorrect_or_parse_invalid"
    assert accepted_reasoning_payload(row, trivial) is None
    assert rejection_reason(row, trivial) == "missing_or_trivial_reasoning"
