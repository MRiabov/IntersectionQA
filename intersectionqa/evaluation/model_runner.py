"""Zero-shot model evaluation helpers."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from intersectionqa.enums import Split, TaskType
from intersectionqa.evaluation.metrics import Prediction, TaskMetrics, evaluate_predictions
from intersectionqa.hashing import sha256_json
from intersectionqa.experiments import prediction_record
from intersectionqa.schema import PublicTaskRow

ZERO_SHOT_EVAL_VERSION = "zero_shot_eval_v01"
ZERO_SHOT_PROMPT_VERSION = "closed_book_zero_shot_v01"
FEW_SHOT_EVAL_VERSION = "few_shot_eval_v01"
FEW_SHOT_PROMPT_VERSION = "closed_book_few_shot_v01"
PARSER_POLICY_VERSION = "strict_exact_answer_v01"
ZERO_SHOT_SYSTEM_PROMPT = (
    "You are evaluating code-only CAD spatial reasoning prompts from the "
    "IntersectionQA/IntersectionEdit benchmark family. "
    "Do not execute code, call tools, or ask for clarification. Return only the exact final "
    "answer token requested by the user prompt."
)
FEW_SHOT_SYSTEM_PROMPT = (
    "You are evaluating code-only CAD spatial reasoning prompts from the "
    "IntersectionQA/IntersectionEdit benchmark family. "
    "Study the provided solved examples, then answer the final target prompt. Do not execute "
    "code, call tools, or ask for clarification. Return only the exact final answer token "
    "requested by the target prompt."
)


@dataclass(frozen=True)
class DecodingSettings:
    temperature: float = 0.0
    max_tokens: int = 32
    top_p: float = 1.0
    seed: int | None = None

    def as_request_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelOutput:
    text: str
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelPrediction:
    row_id: str
    output: str
    provider: str
    model: str
    prompt_hash: str | None
    prompt_template_version: str | None
    eval_prompt_version: str
    eval_request_hash: str
    decoding: dict[str, Any]
    response_metadata: dict[str, Any] = field(default_factory=dict)

    def as_prediction(self) -> Prediction:
        return Prediction(row_id=self.row_id, output=self.output)

    def as_json_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ZeroShotRunResult:
    report: dict[str, Any]
    predictions: list[ModelPrediction]
    metrics: list[TaskMetrics]


@dataclass(frozen=True)
class FewShotRunResult:
    report: dict[str, Any]
    predictions: list[ModelPrediction]
    metrics: list[TaskMetrics]


class ModelClient(Protocol):
    def generate(self, messages: list[dict[str, str]], settings: DecodingSettings) -> ModelOutput:
        """Return a single assistant output for the supplied closed-book prompt."""


class OpenAIChatCompletionsClient:
    """Minimal Chat Completions client using the documented REST shape."""

    def __init__(self, spec: ModelSpec, timeout_seconds: float = 60.0) -> None:
        if not spec.api_key_env:
            raise ValueError("openai-chat provider requires api_key_env")
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key environment variable: {spec.api_key_env}")
        self._api_key = api_key
        self._model = spec.model
        self._base_url = (spec.base_url or "https://api.openai.com/v1").rstrip("/")
        self._timeout_seconds = timeout_seconds

    def generate(self, messages: list[dict[str, str]], settings: DecodingSettings) -> ModelOutput:
        payload = {
            "model": self._model,
            "messages": messages,
            **settings.as_request_payload(),
        }
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"chat completion failed with HTTP {exc.code}: {body}") from exc
        choice = data.get("choices", [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        return ModelOutput(
            text=content if isinstance(content, str) else "",
            raw_response={
                "id": data.get("id"),
                "created": data.get("created"),
                "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage"),
                "system_fingerprint": data.get("system_fingerprint"),
            },
        )


class HuggingFaceChatClient:
    """Chat-completion client for Hugging Face Inference Providers."""

    def __init__(self, spec: ModelSpec, timeout_seconds: float = 60.0) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:  # pragma: no cover - dependency is present in project env
            raise RuntimeError("huggingface-chat provider requires huggingface-hub") from exc
        token = os.environ.get(spec.api_key_env) if spec.api_key_env else None
        self._client = InferenceClient(
            model=spec.model,
            provider=spec.provider_options.get("hf_provider"),
            token=token,
            timeout=timeout_seconds,
            base_url=spec.base_url,
        )
        self._model = spec.model

    def generate(self, messages: list[dict[str, str]], settings: DecodingSettings) -> ModelOutput:
        response = self._client.chat_completion(
            messages=messages,
            model=self._model,
            stream=False,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            seed=settings.seed,
        )
        choice = response.choices[0]
        return ModelOutput(
            text=choice.message.content or "",
            raw_response={
                "id": getattr(response, "id", None),
                "created": getattr(response, "created", None),
                "finish_reason": choice.finish_reason,
                "usage": getattr(response, "usage", None),
            },
        )


def build_model_client(spec: ModelSpec, timeout_seconds: float = 60.0) -> ModelClient:
    if spec.provider == "openai-chat":
        return OpenAIChatCompletionsClient(spec, timeout_seconds=timeout_seconds)
    if spec.provider == "huggingface-chat":
        return HuggingFaceChatClient(spec, timeout_seconds=timeout_seconds)
    raise ValueError(f"unsupported zero-shot provider: {spec.provider}")


def zero_shot_messages(row: PublicTaskRow) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
        {"role": "user", "content": row.prompt},
    ]


def few_shot_messages(row: PublicTaskRow, examples: list[PublicTaskRow]) -> list[dict[str, str]]:
    example_text = "\n\n".join(
        [
            f"Example {index}\nPrompt:\n{example.prompt}\nAnswer:\n{example.answer}"
            for index, example in enumerate(examples, start=1)
        ]
    )
    user_content = (
        f"{example_text}\n\nTarget prompt:\n{row.prompt}"
        if example_text
        else f"Target prompt:\n{row.prompt}"
    )
    return [
        {"role": "system", "content": FEW_SHOT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def zero_shot_request_record(
    row: PublicTaskRow,
    spec: ModelSpec,
    settings: DecodingSettings,
) -> dict[str, Any]:
    messages = zero_shot_messages(row)
    record = {
        "id": row.id,
        "provider": spec.provider,
        "model": spec.model,
        "eval_version": ZERO_SHOT_EVAL_VERSION,
        "eval_prompt_version": ZERO_SHOT_PROMPT_VERSION,
        "prompt_template_version": row.metadata.get("prompt_template_version"),
        "prompt_hash": row.hashes.prompt_hash,
        "messages": messages,
        "decoding": settings.as_request_payload(),
    }
    record["eval_request_hash"] = sha256_json(record)
    return record


def few_shot_request_record(
    row: PublicTaskRow,
    examples: list[PublicTaskRow],
    spec: ModelSpec,
    settings: DecodingSettings,
) -> dict[str, Any]:
    messages = few_shot_messages(row, examples)
    record = {
        "id": row.id,
        "provider": spec.provider,
        "model": spec.model,
        "eval_version": FEW_SHOT_EVAL_VERSION,
        "eval_prompt_version": FEW_SHOT_PROMPT_VERSION,
        "prompt_template_version": row.metadata.get("prompt_template_version"),
        "prompt_hash": row.hashes.prompt_hash,
        "few_shot_example_ids": [example.id for example in examples],
        "few_shot_example_prompt_hashes": [example.hashes.prompt_hash for example in examples],
        "messages": messages,
        "decoding": settings.as_request_payload(),
    }
    record["eval_request_hash"] = sha256_json(record)
    return record


def select_rows(
    rows: Iterable[PublicTaskRow],
    *,
    splits: set[Split] | None = None,
    task_types: set[TaskType] | None = None,
    limit: int | None = None,
) -> list[PublicTaskRow]:
    selected = [
        row
        for row in rows
        if (splits is None or row.split in splits)
        and (task_types is None or row.task_type in task_types)
    ]
    selected = sorted(selected, key=lambda row: row.id)
    return selected[:limit] if limit is not None else selected


def select_few_shot_examples(
    rows: Iterable[PublicTaskRow],
    target: PublicTaskRow,
    *,
    shot_count: int,
    example_splits: set[Split] | None = None,
) -> list[PublicTaskRow]:
    if shot_count <= 0:
        return []
    example_splits = example_splits or {Split.TRAIN}
    target_group = _row_split_group(target)
    candidates = [
        row
        for row in rows
        if row.id != target.id
        and row.task_type == target.task_type
        and row.split in example_splits
        and _row_split_group(row) != target_group
    ]
    return sorted(candidates, key=lambda row: (_answer_sort_key(row), row.id))[:shot_count]


def run_zero_shot_evaluation(
    rows: Iterable[PublicTaskRow],
    client: ModelClient,
    spec: ModelSpec,
    settings: DecodingSettings,
    *,
    splits: set[Split] | None = None,
    task_types: set[TaskType] | None = None,
    limit: int | None = None,
) -> ZeroShotRunResult:
    selected = select_rows(rows, splits=splits, task_types=task_types, limit=limit)
    predictions: list[ModelPrediction] = []
    started = time.time()
    for row in selected:
        request_record = zero_shot_request_record(row, spec, settings)
        output = client.generate(request_record["messages"], settings)
        predictions.append(
            ModelPrediction(
                row_id=row.id,
                output=output.text,
                provider=spec.provider,
                model=spec.model,
                prompt_hash=row.hashes.prompt_hash,
                prompt_template_version=row.metadata.get("prompt_template_version"),
                eval_prompt_version=ZERO_SHOT_PROMPT_VERSION,
                eval_request_hash=request_record["eval_request_hash"],
                decoding=settings.as_request_payload(),
                response_metadata=output.raw_response,
            )
        )
    metrics = evaluate_predictions(selected, [prediction.as_prediction() for prediction in predictions])
    report = build_zero_shot_report(
        rows=selected,
        predictions=predictions,
        metrics=metrics,
        spec=spec,
        settings=settings,
        elapsed_seconds=time.time() - started,
    )
    return ZeroShotRunResult(report=report, predictions=predictions, metrics=metrics)


def run_few_shot_evaluation(
    rows: Iterable[PublicTaskRow],
    client: ModelClient,
    spec: ModelSpec,
    settings: DecodingSettings,
    *,
    shot_count: int = 3,
    example_splits: set[Split] | None = None,
    splits: set[Split] | None = None,
    task_types: set[TaskType] | None = None,
    limit: int | None = None,
) -> FewShotRunResult:
    all_rows = list(rows)
    selected = select_rows(all_rows, splits=splits, task_types=task_types, limit=limit)
    predictions: list[ModelPrediction] = []
    examples_by_row_id: dict[str, list[str]] = {}
    started = time.time()
    for row in selected:
        examples = select_few_shot_examples(
            all_rows,
            row,
            shot_count=shot_count,
            example_splits=example_splits,
        )
        examples_by_row_id[row.id] = [example.id for example in examples]
        request_record = few_shot_request_record(row, examples, spec, settings)
        output = client.generate(request_record["messages"], settings)
        predictions.append(
            ModelPrediction(
                row_id=row.id,
                output=output.text,
                provider=spec.provider,
                model=spec.model,
                prompt_hash=row.hashes.prompt_hash,
                prompt_template_version=row.metadata.get("prompt_template_version"),
                eval_prompt_version=FEW_SHOT_PROMPT_VERSION,
                eval_request_hash=request_record["eval_request_hash"],
                decoding=settings.as_request_payload(),
                response_metadata={
                    **output.raw_response,
                    "few_shot_example_ids": examples_by_row_id[row.id],
                },
            )
        )
    metrics = evaluate_predictions(selected, [prediction.as_prediction() for prediction in predictions])
    report = build_few_shot_report(
        rows=selected,
        predictions=predictions,
        metrics=metrics,
        spec=spec,
        settings=settings,
        elapsed_seconds=time.time() - started,
        shot_count=shot_count,
        examples_by_row_id=examples_by_row_id,
    )
    return FewShotRunResult(report=report, predictions=predictions, metrics=metrics)


def build_zero_shot_report(
    *,
    rows: list[PublicTaskRow],
    predictions: list[ModelPrediction],
    metrics: list[TaskMetrics],
    spec: ModelSpec,
    settings: DecodingSettings,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "eval_version": ZERO_SHOT_EVAL_VERSION,
        "eval_prompt_version": ZERO_SHOT_PROMPT_VERSION,
        "parser_policy": PARSER_POLICY_VERSION,
        "provider": spec.provider,
        "model": spec.model,
        "provider_options": spec.provider_options,
        "decoding": settings.as_request_payload(),
        "row_count": len(rows),
        "prediction_count": len(predictions),
        "elapsed_seconds": elapsed_seconds,
        "dataset_versions": sorted({row.dataset_version for row in rows}),
        "splits": sorted({row.split for row in rows}),
        "task_types": sorted({row.task_type for row in rows}),
        "prompt_template_versions": sorted(
            {
                row.metadata.get("prompt_template_version")
                for row in rows
                if row.metadata.get("prompt_template_version")
            }
        ),
        "request_hashes": [prediction.eval_request_hash for prediction in predictions],
        "metrics": [_json_ready(asdict(metric)) for metric in metrics],
    }


def build_few_shot_report(
    *,
    rows: list[PublicTaskRow],
    predictions: list[ModelPrediction],
    metrics: list[TaskMetrics],
    spec: ModelSpec,
    settings: DecodingSettings,
    elapsed_seconds: float,
    shot_count: int,
    examples_by_row_id: dict[str, list[str]],
) -> dict[str, Any]:
    report = build_zero_shot_report(
        rows=rows,
        predictions=predictions,
        metrics=metrics,
        spec=spec,
        settings=settings,
        elapsed_seconds=elapsed_seconds,
    )
    report.update(
        {
            "eval_version": FEW_SHOT_EVAL_VERSION,
            "eval_prompt_version": FEW_SHOT_PROMPT_VERSION,
            "comparison_baseline_eval_version": ZERO_SHOT_EVAL_VERSION,
            "few_shot_count": shot_count,
            "few_shot_selection_policy": "same_task_train_split_excluding_target_split_group",
            "few_shot_examples_by_row_id": examples_by_row_id,
        }
    )
    return report


def write_request_jsonl(
    rows: Iterable[PublicTaskRow],
    spec: ModelSpec,
    settings: DecodingSettings,
    path: Path,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(zero_shot_request_record(row, spec, settings), sort_keys=True) + "\n")
            count += 1
    return count


def write_few_shot_request_jsonl(
    rows: Iterable[PublicTaskRow],
    spec: ModelSpec,
    settings: DecodingSettings,
    path: Path,
    *,
    shot_count: int = 3,
    example_splits: set[Split] | None = None,
    all_rows: Iterable[PublicTaskRow] | None = None,
) -> int:
    selected = list(rows)
    source_rows = list(all_rows) if all_rows is not None else selected
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in selected:
            examples = select_few_shot_examples(
                source_rows,
                row,
                shot_count=shot_count,
                example_splits=example_splits,
            )
            handle.write(
                json.dumps(few_shot_request_record(row, examples, spec, settings), sort_keys=True)
                + "\n"
            )
            count += 1
    return count


def write_predictions_jsonl(
    predictions: Iterable[ModelPrediction],
    path: Path,
    *,
    rows: Iterable[PublicTaskRow] | None = None,
    include_prompt: bool = False,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_id = {row.id: row for row in rows} if rows is not None else {}
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            record = prediction.as_json_record()
            row = rows_by_id.get(prediction.row_id)
            if row is not None:
                record = {
                    **record,
                    **prediction_record(
                        row,
                        {"row_id": prediction.row_id, "output": prediction.output},
                        prompt=include_prompt,
                    ),
                }
            handle.write(json.dumps(_json_ready(record), sort_keys=True) + "\n")
            count += 1
    return count


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if hasattr(value, "model_dump"):
        return _json_ready(value.model_dump())
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _row_split_group(row: PublicTaskRow) -> str:
    return str(
        row.metadata.get("split_group")
        or row.counterfactual_group_id
        or row.assembly_group_id
        or row.base_object_pair_id
        or row.id
    )


def _answer_sort_key(row: PublicTaskRow) -> tuple[str, str]:
    return (str(row.answer), row.id)
