"""Distill native OpenRouter reasoning traces for IntersectionQA rows."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any
import urllib.error
import urllib.request

DEFAULT_SYSTEM_PROMPT = (
    "You are a CAD spatial reasoning evaluator. Use only the provided benchmark prompt. "
    "Do not execute code, call external tools, or ask for clarification."
)
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
ALLOWED_BY_TASK = {
    "binary_interference": {"yes", "no"},
    "relation_classification": {
        "disjoint",
        "touching",
        "near_miss",
        "intersecting",
        "contained",
        "invalid",
    },
    "volume_bucket": {
        "0",
        "(0, 0.01]",
        "(0.01, 0.05]",
        "(0.05, 0.20]",
        "(0.20, 0.50]",
        ">0.50",
    },
}
DEFAULT_TASK_TYPES = [
    "binary_interference",
    "relation_classification",
    "volume_bucket",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/IntersectionQA-90K"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--fallback-models", nargs="*", default=[])
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--task-types", nargs="+", default=DEFAULT_TASK_TYPES)
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--reasoning-max-tokens", type=int, default=8192)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-base-sleep-seconds", type=float, default=10.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Delete existing per-row JSONL artifacts before starting.")
    parser.add_argument("--provider-sort", choices=["price", "throughput", "latency"], default="throughput")
    parser.add_argument("--no-provider-routing", action="store_true")
    parser.add_argument("--json-schema", action="store_true", help="Opt into JSON response_format; off by default.")
    parser.add_argument("--no-json-schema-fallback", action="store_true")
    parser.add_argument("--hf-bucket")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if len([args.model, *args.fallback_models]) > 3:
        raise ValueError("OpenRouter supports at most 3 total models in a fallback request")

    _load_env_file(Path(".env"))
    api_key_env = args.api_key_env or _default_api_key_env()
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"missing OpenRouter API key environment variable: {api_key_env}")
    base_url = (args.base_url or _default_base_url()).rstrip("/")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _sample_rows(
        args.dataset_dir,
        splits=args.splits,
        task_types=set(args.task_types),
        max_rows=args.max_rows,
        seed=args.seed,
    )
    request_config = {
        "model": args.model,
        "fallback_models": args.fallback_models,
        "base_url": base_url,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "reasoning": _reasoning_config(args.reasoning_max_tokens),
        "provider": None if args.no_provider_routing else _provider_routing(args.provider_sort),
        "json_schema": args.json_schema,
        "json_schema_fallback": not args.no_json_schema_fallback,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
    (args.output_dir / "request_config.json").write_text(
        json.dumps({**request_config, "api_key_env": api_key_env}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    raw_path = args.output_dir / "raw_responses.jsonl"
    trace_path = args.output_dir / "traces.jsonl"
    accepted_path = args.output_dir / "accepted_reasoning_sft.jsonl"
    if args.fresh:
        for path in (raw_path, trace_path, accepted_path, args.output_dir / "report.json"):
            path.unlink(missing_ok=True)
    completed_ids = _completed_ids(trace_path)
    counters: Counter[str] = Counter()
    usage_totals: Counter[str] = Counter()
    accepted_reasoning_lengths: list[int] = []
    observed_reasoning_lengths: list[int] = []

    pending_rows = [(index, row) for index, row in enumerate(rows, start=1) if row["id"] not in completed_ids]
    counters["skipped_existing"] = len(rows) - len(pending_rows)

    with raw_path.open("a", encoding="utf-8") as raw_handle, trace_path.open(
        "a", encoding="utf-8"
    ) as trace_handle, accepted_path.open("a", encoding="utf-8") as accepted_handle:
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            for batch in _batched(pending_rows, args.batch_size):
                futures = [
                    executor.submit(
                        _process_row,
                        index,
                        row,
                        api_key=api_key,
                        base_url=base_url,
                        args=args,
                        request_config=request_config,
                    )
                    for index, row in batch
                ]
                for future in as_completed(futures):
                    result = future.result()
                    row_id = result["id"]
                    raw_handle.write(json.dumps(result["raw_record"], sort_keys=True) + "\n")
                    raw_handle.flush()
                    trace = result["trace"]
                    trace_handle.write(json.dumps(trace, sort_keys=True) + "\n")
                    trace_handle.flush()
                    completed_ids.add(row_id)

                    counters["rows"] += 1
                    if result.get("error_type"):
                        counters["request_errors"] += 1
                        counters[f"error_type:{result['error_type']}"] += 1
                    counters[f"parse_status:{trace['parse_status']}"] += 1
                    counters[f"reasoning_status:{trace['reasoning_status']}"] += 1
                    if trace["reasoning_text"]:
                        observed_reasoning_lengths.append(len(trace["reasoning_text"]))
                    if trace["correct"]:
                        counters["correct"] += 1
                    if trace["accepted"]:
                        counters["accepted"] += 1
                        accepted_reasoning_lengths.append(len(trace["reasoning_text"]))
                        accepted_handle.write(json.dumps(result["accepted_payload"], sort_keys=True) + "\n")
                        accepted_handle.flush()
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "row_count": len(rows),
        "counters": dict(sorted(counters.items())),
        "usage_totals": dict(sorted(usage_totals.items())),
        "accepted_reasoning_length_chars": _length_summary(accepted_reasoning_lengths),
        "observed_reasoning_length_chars": _length_summary(observed_reasoning_lengths),
        "artifacts": {
            "accepted_reasoning_sft": str(accepted_path),
            "raw_responses": str(raw_path),
            "request_config": str(args.output_dir / "request_config.json"),
            "traces": str(trace_path),
        },
        "request_config": request_config,
    }
    if args.hf_bucket:
        try:
            report["hf_bucket_upload"] = _sync_to_hf_bucket(args.output_dir, args.hf_bucket)
        except Exception as exc:
            report["hf_bucket_upload"] = {
                "status": "failed",
                "target": args.hf_bucket,
                "error": str(exc),
            }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _process_row(
    index: int,
    row: dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    args: argparse.Namespace,
    request_config: dict[str, Any],
) -> dict[str, Any]:
    try:
        response = _request_completion(
            api_key=api_key,
            base_url=base_url,
            model=args.model,
            fallback_models=args.fallback_models,
            provider_sort=None if args.no_provider_routing else args.provider_sort,
            row=row,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            reasoning_max_tokens=args.reasoning_max_tokens,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
            use_json_schema=args.json_schema,
            allow_json_schema_fallback=not args.no_json_schema_fallback,
            retry_base_sleep_seconds=args.retry_base_sleep_seconds,
        )
    except Exception as exc:
        if not args.continue_on_error:
            raise
        error_record = {
            "id": row["id"],
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        return {
            "id": row["id"],
            "raw_record": error_record,
            "trace": _error_trace(row, error_record, index),
            "error_type": type(exc).__name__,
            "accepted_payload": None,
        }
    trace = _trace_from_response(row, response)
    trace["index"] = index
    trace["request_hash"] = _sha256_json(
        {
            "row_id": row["id"],
            "prompt_hash": row.get("hashes", {}).get("prompt_hash"),
            "config": request_config,
        }
    )
    return {
        "id": row["id"],
        "raw_record": {"id": row["id"], "response": response},
        "trace": trace,
        "error_type": None,
        "accepted_payload": _accepted_payload(row, trace) if trace["accepted"] else None,
    }


def _default_api_key_env() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return "OPENROUTER_API_KEY"
    return "OPENAI_API_KEY"


def _default_base_url() -> str:
    if os.environ.get("OPENROUTER_BASE_URL"):
        return str(os.environ["OPENROUTER_BASE_URL"])
    if os.environ.get("OPENROUTER_API_BASE"):
        return str(os.environ["OPENROUTER_API_BASE"])
    if os.environ.get("OPENROUTER_API_KEY"):
        return "https://openrouter.ai/api/v1"
    return os.environ.get("OPENAI_API_URL", "https://openrouter.ai/api/v1")


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = record.get("id")
            if isinstance(row_id, str):
                completed.add(row_id)
    return completed


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _sample_rows(
    dataset_dir: Path,
    *,
    splits: list[str],
    task_types: set[str],
    max_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing split file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("task_type") in task_types:
                    rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:max_rows]


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _request_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    fallback_models: list[str],
    provider_sort: str | None,
    row: dict[str, Any],
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning_max_tokens: int,
    timeout_seconds: float,
    retries: int,
    use_json_schema: bool,
    allow_json_schema_fallback: bool,
    retry_base_sleep_seconds: float,
) -> dict[str, Any]:
    return _request_completion_pydantic_ai(
        api_key=api_key,
        model=model,
        fallback_models=fallback_models,
        provider_sort=provider_sort,
        row=row,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning_max_tokens=reasoning_max_tokens,
        timeout_seconds=timeout_seconds,
    )


def _request_completion_pydantic_ai(
    *,
    api_key: str,
    model: str,
    fallback_models: list[str],
    provider_sort: str | None,
    row: dict[str, Any],
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning_max_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    from pydantic_ai import Agent
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    provider = OpenRouterProvider(
        api_key=api_key,
        app_url="https://github.com/Problemologist/IntersectionQA",
        app_title="IntersectionQA reasoning distill",
    )
    model_obj = OpenRouterModel(model, provider=provider)
    settings: OpenRouterModelSettings = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "timeout": timeout_seconds,
        "openrouter_reasoning": _reasoning_config(reasoning_max_tokens),
        "openrouter_usage": {"include": True},
    }
    if fallback_models:
        models = [model, *fallback_models]
        if len(models) > 3:
            raise ValueError("OpenRouter supports at most 3 total models in a fallback request")
        settings["openrouter_models"] = models
    if provider_sort is not None:
        settings["openrouter_provider"] = {
            "allow_fallbacks": True,
            "require_parameters": True,
            "sort": provider_sort,
        }
    agent = Agent(model_obj, output_type=str, system_prompt=DEFAULT_SYSTEM_PROMPT)
    result = agent.run_sync(row["prompt"], model_settings=settings)
    messages = result.all_messages()
    reasoning_parts = _thinking_parts_from_messages(messages)
    return {
        "client": "pydantic-ai",
        "output": result.output,
        "reasoning_text": "\n\n".join(part.get("content", "") for part in reasoning_parts if part.get("content")),
        "reasoning_details": reasoning_parts,
        "usage": _jsonable(result.usage()),
        "provider_details": _jsonable(getattr(result.response, "provider_details", None)),
        "messages": _messages_json(result),
    }


def _request_completion_urllib(
    *,
    api_key: str,
    base_url: str,
    model: str,
    fallback_models: list[str],
    provider_sort: str | None,
    row: dict[str, Any],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_seconds: float,
    retries: int,
    use_json_schema: bool,
    allow_json_schema_fallback: bool,
    retry_base_sleep_seconds: float,
) -> dict[str, Any]:
    payload = _request_payload(
        model=model,
        fallback_models=fallback_models,
        provider_sort=provider_sort,
        row=row,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        use_json_schema=use_json_schema,
    )
    for attempt in range(retries + 1):
        try:
            return _post_chat_completion(api_key, base_url, payload, timeout_seconds)
        except RuntimeError as exc:
            if (
                use_json_schema
                and allow_json_schema_fallback
                and "HTTP 400" in str(exc)
                and payload.pop("response_format", None) is not None
            ):
                continue
            if attempt >= retries:
                raise
            delay = retry_base_sleep_seconds * (2**attempt) if "HTTP 429" in str(exc) else 2**attempt
            time.sleep(delay)
    raise AssertionError("unreachable retry loop")


def _request_payload(
    *,
    model: str,
    fallback_models: list[str],
    provider_sort: str | None,
    row: dict[str, Any],
    temperature: float,
    top_p: float,
    max_tokens: int,
    use_json_schema: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": True},
    }
    models = [model, *fallback_models]
    if len(models) == 1:
        payload["model"] = models[0]
    else:
        payload["models"] = models
    if provider_sort is not None:
        payload["provider"] = _provider_routing(provider_sort)
    if use_json_schema:
        payload["response_format"] = _final_answer_response_format(row)
    return payload


def _provider_routing(sort: str) -> dict[str, Any]:
    return {
        "allow_fallbacks": True,
        "require_parameters": True,
        "sort": {
            "by": sort,
            "partition": "none",
        },
    }


def _reasoning_config(max_tokens: int | None) -> dict[str, Any]:
    if max_tokens is None or max_tokens <= 0:
        return {"enabled": True}
    return {"max_tokens": max_tokens}


def _final_answer_response_format(row: dict[str, Any]) -> dict[str, Any]:
    allowed = sorted(ALLOWED_BY_TASK[row["task_type"]])
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "intersectionqa_final_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "final_answer": {
                        "type": "string",
                        "enum": allowed,
                    }
                },
                "required": ["final_answer"],
            },
        },
    }


def _post_chat_completion(
    api_key: str,
    base_url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Problemologist/IntersectionQA",
            "X-Title": "IntersectionQA reasoning distill",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"chat completion failed with HTTP {exc.code}: {body}") from exc


def _trace_from_response(row: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    if response.get("client") == "pydantic-ai":
        content = response.get("output")
        final_answer = _final_answer_from_content(content if isinstance(content, str) else "")
        parsed = _parse_answer(row["task_type"], final_answer or "")
        reasoning_text = response.get("reasoning_text") if isinstance(response.get("reasoning_text"), str) else ""
        correct = parsed == row["answer"]
        reasoning_status = "native_reasoning_present" if reasoning_text.strip() else "missing_native_reasoning"
        parse_status = "parse_valid" if parsed is not None else "parse_invalid"
        return {
            "id": row["id"],
            "split": row.get("split"),
            "task_type": row["task_type"],
            "answer": row["answer"],
            "final_answer": final_answer,
            "parsed_answer": parsed,
            "correct": correct,
            "accepted": bool(correct and reasoning_text.strip()),
            "parse_status": parse_status,
            "reasoning_status": reasoning_status,
            "reasoning_text": reasoning_text,
            "reasoning_details": response.get("reasoning_details"),
            "content": content,
            "finish_reason": None,
            "usage": response.get("usage"),
            "response_id": None,
            "model": response.get("provider_details", {}).get("model_name") if isinstance(response.get("provider_details"), dict) else None,
            "provider_details": response.get("provider_details"),
        }
    if isinstance(response.get("error"), dict):
        error_record = {
            "id": row["id"],
            "error": json.dumps(response["error"], sort_keys=True),
            "error_type": "OpenRouterResponseError",
        }
        return _error_trace(row, error_record, index=0)
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else ""
    final_answer = _final_answer_from_content(content if isinstance(content, str) else "")
    parsed = _parse_answer(row["task_type"], final_answer or "")
    reasoning_text = _reasoning_text(message)
    reasoning_details = message.get("reasoning_details")
    correct = parsed == row["answer"]
    reasoning_status = "native_reasoning_present" if reasoning_text.strip() else "missing_native_reasoning"
    parse_status = "parse_valid" if parsed is not None else "parse_invalid"
    return {
        "id": row["id"],
        "split": row.get("split"),
        "task_type": row["task_type"],
        "answer": row["answer"],
        "final_answer": final_answer,
        "parsed_answer": parsed,
        "correct": correct,
        "accepted": bool(correct and reasoning_text.strip()),
        "parse_status": parse_status,
        "reasoning_status": reasoning_status,
        "reasoning_text": reasoning_text,
        "reasoning_details": reasoning_details,
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
        "response_id": response.get("id"),
        "model": response.get("model"),
    }


def _error_trace(row: dict[str, Any], error_record: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": row["id"],
        "index": index,
        "split": row.get("split"),
        "task_type": row["task_type"],
        "answer": row["answer"],
        "final_answer": "",
        "parsed_answer": None,
        "correct": False,
        "accepted": False,
        "parse_status": "request_error",
        "reasoning_status": "request_error",
        "reasoning_text": "",
        "reasoning_details": None,
        "content": "",
        "finish_reason": None,
        "usage": None,
        "response_id": None,
        "model": None,
        "error": error_record,
    }


def _final_answer_from_content(content: str) -> str:
    stripped = content.strip()
    if not stripped:
        return ""
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        candidate = _canonical_answer_candidate(stripped)
        return candidate.strip()
    if isinstance(decoded, dict):
        value = decoded.get("final_answer")
        if isinstance(value, str):
            return value.strip()
    if isinstance(decoded, str):
        return decoded.strip()
    return stripped


def _parse_answer(task_type: str, output: str) -> str | None:
    stripped = output.strip(" \t\r\n")
    allowed = ALLOWED_BY_TASK.get(task_type)
    if allowed is None:
        raise ValueError(f"unsupported task type for this self-contained distill script: {task_type}")
    return stripped if stripped in allowed else None


def _canonical_answer_candidate(output: str) -> str:
    match = ANSWER_TAG_RE.search(output)
    if match is None:
        return output
    return match.group(1).strip()


def _reasoning_text(message: dict[str, Any]) -> str:
    direct = message.get("reasoning") or message.get("reasoning_content")
    if isinstance(direct, str):
        return direct.strip()
    details = message.get("reasoning_details")
    if not isinstance(details, list):
        return ""
    pieces: list[str] = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        for key in ("text", "summary", "content"):
            value = detail.get(key)
            if isinstance(value, str) and value.strip():
                pieces.append(value.strip())
                break
    return "\n\n".join(pieces).strip()


def _accepted_payload(row: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload["target_text"] = f"<think>{trace['reasoning_text'].strip()}</think><answer>{trace['final_answer']}</answer>"
    payload["canonical_answer"] = row["answer"]
    payload["supervision"] = {
        "target_text_format": "native_reasoning_think_answer_v01",
        "target_text_source": "openrouter_native_reasoning",
        "reasoning_model": trace.get("model"),
        "acceptance_policy": "parse_valid_answer_correct_native_reasoning_present_v01",
    }
    return payload


def _length_summary(lengths: list[int]) -> dict[str, float | int | None]:
    if not lengths:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
    }


def _sync_to_hf_bucket(source: Path, target: str) -> dict[str, Any]:
    if not os.environ.get("HF_TOKEN") and os.environ.get("HF_TOKEN_PUSH"):
        os.environ["HF_TOKEN"] = os.environ["HF_TOKEN_PUSH"]
    from huggingface_hub import sync_bucket

    result = sync_bucket(str(source), target)
    summary = getattr(result, "summary", None)
    if callable(summary):
        return {"target": target, "summary": dict(summary())}
    return {"target": target}


def _thinking_parts_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for message in messages:
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", None) != "thinking":
                continue
            parts.append(
                {
                    "content": getattr(part, "content", ""),
                    "id": getattr(part, "id", None),
                    "provider_details": _jsonable(getattr(part, "provider_details", None)),
                    "provider_name": getattr(part, "provider_name", None),
                    "signature": getattr(part, "signature", None),
                }
            )
    return parts


def _messages_json(result: Any) -> Any:
    try:
        payload = result.all_messages_json()
    except Exception:
        return None
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    try:
        return json.loads(payload)
    except Exception:
        return payload


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _sha256_json(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
