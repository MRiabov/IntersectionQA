import json

from intersectionqa.config import DatasetConfig
from intersectionqa.enums import Split, TaskType
from intersectionqa.evaluation.model_runner import (
    DecodingSettings,
    FEW_SHOT_PROMPT_VERSION,
    ModelOutput,
    ModelSpec,
    ZERO_SHOT_PROMPT_VERSION,
    few_shot_request_record,
    run_few_shot_evaluation,
    run_zero_shot_evaluation,
    select_few_shot_examples,
    select_rows,
    write_few_shot_request_jsonl,
    write_predictions_jsonl,
    write_report,
    write_request_jsonl,
    zero_shot_request_record,
)
from intersectionqa.pipeline import build_smoke_rows


class RecordingClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.messages = []

    def generate(self, messages, settings):
        self.messages.append((messages, settings))
        return ModelOutput(text=self.outputs.pop(0), raw_response={"finish_reason": "stop"})


def test_zero_shot_request_records_are_versioned_and_stable():
    rows, _ = build_smoke_rows(DatasetConfig())
    row = rows[0]
    spec = ModelSpec(provider="openai-chat", model="frontier-test", api_key_env="OPENAI_API_KEY")
    settings = DecodingSettings(temperature=0.0, max_tokens=16, top_p=1.0)

    first = zero_shot_request_record(row, spec, settings)
    second = zero_shot_request_record(row, spec, settings)

    assert first == second
    assert first["eval_prompt_version"] == ZERO_SHOT_PROMPT_VERSION
    assert first["prompt_template_version"] == row.metadata["prompt_template_version"]
    assert first["prompt_hash"] == row.hashes.prompt_hash
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][1]["content"] == row.prompt
    assert first["eval_request_hash"].startswith("sha256:")


def test_zero_shot_runner_reports_invalid_outputs():
    rows, _ = build_smoke_rows(DatasetConfig())
    binary_rows = select_rows(
        rows,
        task_types={TaskType.BINARY_INTERFERENCE},
        limit=2,
    )
    client = RecordingClient([binary_rows[0].answer, "invalid prose"])
    spec = ModelSpec(provider="openai-chat", model="frontier-test", api_key_env="OPENAI_API_KEY")
    settings = DecodingSettings(temperature=0.0, max_tokens=8, top_p=1.0, seed=123)

    result = run_zero_shot_evaluation(binary_rows, client, spec, settings)

    assert len(result.predictions) == 2
    assert result.report["parser_policy"] == "strict_exact_answer_v01"
    assert result.report["decoding"] == {
        "temperature": 0.0,
        "max_tokens": 8,
        "top_p": 1.0,
        "seed": 123,
    }
    assert result.metrics[0].invalid_outputs == 1
    assert result.metrics[0].invalid_output_rate == 0.5
    assert client.messages[0][0][0]["role"] == "system"


def test_few_shot_examples_are_train_split_and_group_safe():
    rows, _ = build_smoke_rows(DatasetConfig())
    target = next(
        row
        for row in rows
        if row.task_type == TaskType.BINARY_INTERFERENCE and row.split != Split.TRAIN
    )

    examples = select_few_shot_examples(rows, target, shot_count=3)

    assert examples
    assert all(example.split == Split.TRAIN for example in examples)
    assert all(example.task_type == target.task_type for example in examples)
    assert all(example.metadata["split_group"] != target.metadata["split_group"] for example in examples)


def test_few_shot_request_records_include_examples_and_runner_reports_policy():
    rows, _ = build_smoke_rows(DatasetConfig())
    target = next(
        row
        for row in rows
        if row.task_type == TaskType.BINARY_INTERFERENCE and row.split != Split.TRAIN
    )
    examples = select_few_shot_examples(rows, target, shot_count=1)
    spec = ModelSpec(provider="openai-chat", model="frontier-test", api_key_env="OPENAI_API_KEY")
    settings = DecodingSettings(temperature=0.0, max_tokens=8, top_p=1.0)

    request = few_shot_request_record(target, examples, spec, settings)

    assert request["eval_prompt_version"] == FEW_SHOT_PROMPT_VERSION
    assert request["few_shot_example_ids"] == [example.id for example in examples]
    assert "Example 1" in request["messages"][1]["content"]
    assert "Target prompt:" in request["messages"][1]["content"]

    selected = select_rows(rows, task_types={TaskType.BINARY_INTERFERENCE}, limit=2)
    client = RecordingClient([row.answer for row in selected])
    result = run_few_shot_evaluation(
        rows,
        client,
        spec,
        settings,
        shot_count=1,
        task_types={TaskType.BINARY_INTERFERENCE},
        limit=2,
    )

    assert len(result.predictions) == 2
    assert result.report["eval_prompt_version"] == FEW_SHOT_PROMPT_VERSION
    assert result.report["comparison_baseline_eval_version"] == "zero_shot_eval_v01"
    assert result.report["few_shot_count"] == 1
    assert result.report["few_shot_examples_by_row_id"]


def test_zero_shot_artifact_writers(tmp_path):
    rows, _ = build_smoke_rows(DatasetConfig())
    selected = select_rows(rows, task_types={TaskType.BINARY_INTERFERENCE}, limit=1)
    spec = ModelSpec(provider="huggingface-chat", model="open-code-test", api_key_env="HF_TOKEN")
    settings = DecodingSettings(max_tokens=4)
    request_path = tmp_path / "requests.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "report.json"

    assert write_request_jsonl(selected, spec, settings, request_path) == 1
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["model"] == "open-code-test"
    assert request["eval_prompt_version"] == ZERO_SHOT_PROMPT_VERSION

    client = RecordingClient([selected[0].answer])
    result = run_zero_shot_evaluation(selected, client, spec, settings)
    assert write_predictions_jsonl(result.predictions, prediction_path) == 1
    write_report(result.report, report_path)

    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert prediction["row_id"] == selected[0].id
    assert report["prediction_count"] == 1


def test_few_shot_request_writer_uses_full_row_pool(tmp_path):
    rows, _ = build_smoke_rows(DatasetConfig())
    selected = select_rows(rows, task_types={TaskType.BINARY_INTERFERENCE}, limit=1)
    spec = ModelSpec(provider="huggingface-chat", model="open-code-test", api_key_env="HF_TOKEN")
    settings = DecodingSettings(max_tokens=4)
    request_path = tmp_path / "fewshot_requests.jsonl"

    assert (
        write_few_shot_request_jsonl(
            selected,
            spec,
            settings,
            request_path,
            shot_count=1,
            all_rows=rows,
        )
        == 1
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["eval_prompt_version"] == FEW_SHOT_PROMPT_VERSION
    assert request["few_shot_example_ids"]
