"""Run or export a zero-shot closed-book model evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from intersectionqa.enums import Split, TaskType
from intersectionqa.evaluation.model_runner import (
    DecodingSettings,
    ModelSpec,
    build_model_client,
    run_zero_shot_evaluation,
    select_rows,
    write_predictions_jsonl,
    write_report,
    write_request_jsonl,
)
from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import validate_dataset_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--provider", choices=["openai-chat", "huggingface-chat"], default="openai-chat")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--hf-provider", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", action="append", choices=[item.value for item in Split])
    parser.add_argument("--task-type", action="append", choices=[item.value for item in TaskType])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("eval/zero_shot"))
    parser.add_argument("--predictions-jsonl", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument(
        "--requests-jsonl",
        type=Path,
        default=None,
        help="Write versioned request records before running, or as the only artifact with --export-requests-only.",
    )
    parser.add_argument("--export-requests-only", action="store_true")
    args = parser.parse_args(argv)

    configure_logging()
    _load_env_file(args.env_file)
    rows = validate_dataset_dir(args.dataset_dir)
    selected = select_rows(
        rows,
        splits={Split(item) for item in args.split} if args.split else None,
        task_types={TaskType(item) for item in args.task_type} if args.task_type else None,
        limit=args.limit,
    )
    settings = DecodingSettings(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        seed=args.seed,
    )
    spec = ModelSpec(
        provider=args.provider,
        model=args.model,
        api_key_env=args.api_key_env or _default_api_key_env(args.provider),
        base_url=args.base_url,
        provider_options={"hf_provider": args.hf_provider} if args.hf_provider else {},
    )

    if args.requests_jsonl is not None:
        write_request_jsonl(selected, spec, settings, args.requests_jsonl)
    if args.export_requests_only:
        return

    client = build_model_client(spec, timeout_seconds=args.timeout_seconds)
    result = run_zero_shot_evaluation(selected, client, spec, settings)
    artifact_stem = f"{args.provider}_{_path_token(args.model)}"
    predictions_path = args.predictions_jsonl or args.output_dir / f"{artifact_stem}_predictions.jsonl"
    report_path = args.report_json or args.output_dir / f"{artifact_stem}_report.json"
    write_predictions_jsonl(result.predictions, predictions_path)
    write_report(result.report, report_path)
    print(report_path)


def _default_api_key_env(provider: str) -> str | None:
    if provider == "openai-chat":
        return "OPENAI_API_KEY"
    if provider == "huggingface-chat":
        return "HF_TOKEN"
    return None


def _path_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    load_dotenv(path, override=False)


if __name__ == "__main__":
    main()
