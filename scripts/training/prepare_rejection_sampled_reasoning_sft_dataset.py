"""Build a reasoning-SFT dataset from accepted model-generated completions."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from intersectionqa.evaluation.parsing import canonical_answer_candidate, parse_answer
from intersectionqa.training.prompt_features import PROMPT_FEATURE_MODES, prompt_for_mapping


SYSTEM_PROMPT = (
    "Solve the CAD spatial-reasoning task without executing code. "
    "Return exactly one completion in this format: "
    "<think>brief useful reasoning</think><answer>canonical answer</answer>"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="unsloth/Qwen3.5-4B")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--task-types", nargs="+")
    parser.add_argument("--prompt-feature-mode", choices=PROMPT_FEATURE_MODES, default="none")
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--attempts-per-row", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--min-accepted-train-rows", type=int, default=1)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    if args.attempts_per_row <= 0:
        raise ValueError("--attempts-per-row must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = _load_tokenizer(args.model, args.adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization = None
    if not args.no_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    model.eval()

    task_types = set(args.task_types) if args.task_types else None
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "attempts_per_row": args.attempts_per_row,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "prompt_feature_mode": args.prompt_feature_mode,
        "splits": {},
    }

    for split in args.splits:
        rows = _load_rows(args.dataset_dir / f"{split}.jsonl", task_types=task_types)
        if args.max_rows_per_split is not None:
            rows = rows[: args.max_rows_per_split]
        accepted_rows: list[dict[str, Any]] = []
        rejection_reasons: Counter[str] = Counter()
        trace_path = args.output_dir / f"{split}_trace_samples.jsonl"
        with trace_path.open("w", encoding="utf-8") as trace_handle:
            remaining = list(rows)
            for attempt_index in range(args.attempts_per_row):
                next_remaining: list[dict[str, Any]] = []
                for batch in _batched(remaining, args.batch_size):
                    completions = _generate_reasoning_completions(
                        model,
                        tokenizer,
                        batch,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=args.seed + attempt_index,
                        prompt_feature_mode=args.prompt_feature_mode,
                    )
                    for row, completion in zip(batch, completions, strict=True):
                        accepted = accepted_reasoning_payload(row, completion)
                        reason = None if accepted else rejection_reason(row, completion)
                        trace_handle.write(
                            json.dumps(
                                {
                                    "id": row["id"],
                                    "split": split,
                                    "attempt_index": attempt_index,
                                    "accepted": accepted is not None,
                                    "raw_completion": completion,
                                    "rejection_reason": reason,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        trace_handle.flush()
                        if accepted is not None:
                            accepted_rows.append(accepted)
                        elif attempt_index + 1 < args.attempts_per_row:
                            next_remaining.append(row)
                        else:
                            rejection_reasons[reason or "unknown"] += 1
                remaining = next_remaining
                if not remaining:
                    break

        output_path = args.output_dir / f"{split}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for row in accepted_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        report["splits"][split] = {
            "input_rows": len(rows),
            "accepted_rows": len(accepted_rows),
            "acceptance_rate": len(accepted_rows) / len(rows) if rows else 0.0,
            "output_path": str(output_path),
            "trace_path": str(trace_path),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "accepted_task_counts": dict(sorted(Counter(row["task_type"] for row in accepted_rows).items())),
        }

    train_accepted = report["splits"].get("train", {}).get("accepted_rows", 0)
    if train_accepted < args.min_accepted_train_rows:
        raise RuntimeError(
            f"accepted train rows {train_accepted} below minimum {args.min_accepted_train_rows}"
        )

    (args.output_dir / "rejection_sampled_reasoning_sft_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def accepted_reasoning_payload(row: dict[str, Any], completion: str) -> dict[str, Any] | None:
    candidate, _format_components = canonical_answer_candidate(completion)
    parsed = parse_answer(row["task_type"], candidate)
    if parsed != row["answer"]:
        return None
    if not _has_nontrivial_reasoning(completion):
        return None
    payload = dict(row)
    payload["target_text"] = completion.strip()
    payload["canonical_answer"] = row["answer"]
    payload["supervision"] = {
        "target_text_format": "think_answer_v01",
        "target_text_source": "rejection_sampled_model_completion",
        "acceptance_policy": "parse_valid_answer_correct_nontrivial_reasoning_v01",
    }
    return payload


def rejection_reason(row: dict[str, Any], completion: str) -> str:
    candidate, _format_components = canonical_answer_candidate(completion)
    parsed = parse_answer(row["task_type"], candidate)
    if parsed != row["answer"]:
        return "answer_incorrect_or_parse_invalid"
    if not _has_nontrivial_reasoning(completion):
        return "missing_or_trivial_reasoning"
    return "unknown"


def _has_nontrivial_reasoning(completion: str) -> bool:
    lower = completion.lower()
    start = lower.find("<think>")
    end = lower.find("</think>")
    if start < 0 or end <= start:
        return False
    reasoning = completion[start + len("<think>") : end].strip()
    if len(reasoning.split()) < 4:
        return False
    unique_tokens = set(reasoning.lower().split())
    return len(unique_tokens) >= 4


def _load_tokenizer(model: str, adapter_dir: Path | None) -> Any:
    from transformers import AutoTokenizer

    if adapter_dir is not None:
        try:
            return AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
        except (OSError, ValueError):
            pass
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def _load_rows(path: Path, *, task_types: set[str] | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing split file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if task_types is not None and row.get("task_type") not in task_types:
                continue
            rows.append(row)
    return rows


def _generate_reasoning_completions(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    prompt_feature_mode: str,
) -> list[str]:
    import torch

    torch.manual_seed(seed)
    texts = [_reasoning_prompt_text(tokenizer, row, prompt_feature_mode=prompt_feature_mode) for row in rows]
    encoded = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_width = encoded["input_ids"].shape[-1]
    completions = output[:, prompt_width:]
    return [completion.strip() for completion in tokenizer.batch_decode(completions, skip_special_tokens=True)]


def _reasoning_prompt_text(tokenizer: Any, row: dict[str, Any], *, prompt_feature_mode: str) -> str:
    prompt = prompt_for_mapping(row, mode=prompt_feature_mode).rstrip()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"### System:\n{SYSTEM_PROMPT}\n\n### User:\n{prompt}\n\n### Assistant:\n"


def _batched(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


if __name__ == "__main__":
    main()
