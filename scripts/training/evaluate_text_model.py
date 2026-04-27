"""Exact-answer generation evaluator for text-only IntersectionQA models."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from intersectionqa.evaluation.parsing import canonical_answer_candidate, parse_answer
from intersectionqa.training.prompt_features import PROMPT_FEATURE_MODES, prompt_for_mapping


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--splits", nargs="+", default=["validation", "test_random"])
    parser.add_argument("--max-rows-per-split", type=int, default=200)
    parser.add_argument("--max-rows-per-task-per-split", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prompt-feature-mode", choices=PROMPT_FEATURE_MODES, default="none")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--predictions-jsonl", type=Path)
    parser.add_argument("--metrics-jsonl", type=Path)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir or args.model, trust_remote_code=True)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    rows = load_rows(
        args.dataset_dir,
        args.splits,
        args.max_rows_per_split,
        max_rows_per_task_per_split=args.max_rows_per_task_per_split,
        seed=args.seed,
    )
    by_split = defaultdict(Counter)
    by_task = defaultdict(Counter)
    labels_by_task: dict[str, set[str]] = defaultdict(set)
    confusion_by_task: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    examples = []
    prediction_records = []
    for row in rows:
        prediction = generate_answer(
            model,
            tokenizer,
            row,
            max_new_tokens=args.max_new_tokens,
            prompt_feature_mode=args.prompt_feature_mode,
        )
        candidate, _format_components = canonical_answer_candidate(prediction)
        parsed = parse_answer(row["task_type"], candidate)
        normalized_prediction = normalize_answer(prediction)
        normalized_answer = normalize_answer(row["answer"])
        exact = parsed == row["answer"]
        by_split[row["split"]]["total"] += 1
        by_split[row["split"]]["correct"] += int(exact)
        by_task[row["task_type"]]["total"] += 1
        by_task[row["task_type"]]["correct"] += int(exact)
        labels_by_task[row["task_type"]].add(normalized_answer)
        labels_by_task[row["task_type"]].add(normalized_prediction)
        confusion_by_task[row["task_type"]][(normalized_answer, normalized_prediction)] += 1
        prediction_records.append(
            {
                "id": row["id"],
                "row_id": row["id"],
                "split": row["split"],
                "task_type": row["task_type"],
                "prompt_hash": (row.get("hashes") or {}).get("prompt_hash"),
                "raw_completion": prediction,
                "output": prediction,
                "parsed_answer": parsed,
                "canonical_answer": row["answer"],
                "parse_valid": parsed is not None,
                "correct": exact,
            }
        )
        if len(examples) < 50 and not exact:
            examples.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "task_type": row["task_type"],
                    "answer": row["answer"],
                    "prediction": prediction,
                    "normalized_prediction": normalized_prediction,
                }
            )

    payload = {
        "model": args.model,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "row_count": len(rows),
        "by_split": summarize(by_split),
        "by_task": summarize(by_task),
        "classification_by_task": classification_summary(confusion_by_task, labels_by_task),
        "mismatches": examples,
        "prompt_feature_mode": args.prompt_feature_mode,
    }
    output_json = args.output_json
    predictions_jsonl = args.predictions_jsonl
    metrics_jsonl = args.metrics_jsonl
    if args.output_dir is not None:
        output_json = output_json or args.output_dir / "eval_report.json"
        predictions_jsonl = predictions_jsonl or args.output_dir / "predictions" / "final_eval.jsonl"
        metrics_jsonl = metrics_jsonl or args.output_dir / "eval_metrics.jsonl"
    if predictions_jsonl is not None:
        predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with predictions_jsonl.open("w", encoding="utf-8") as handle:
            for record in prediction_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    if metrics_jsonl is not None:
        metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with metrics_jsonl.open("w", encoding="utf-8") as handle:
            for split, counts in sorted(by_split.items()):
                handle.write(
                    json.dumps(
                        {
                            "scope": "split",
                            "split": split,
                            "correct": counts["correct"],
                            "total": counts["total"],
                            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            for task_type, counts in sorted(by_task.items()):
                handle.write(
                    json.dumps(
                        {
                            "scope": "task_type",
                            "task_type": task_type,
                            "correct": counts["correct"],
                            "total": counts["total"],
                            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text, encoding="utf-8")
    print(text)


def load_rows(
    dataset_dir: Path,
    splits: list[str],
    limit_per_split: int,
    *,
    max_rows_per_task_per_split: int | None,
    seed: int,
) -> list[dict]:
    rows = []
    for split in splits:
        split_rows = []
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["diagnostics"].get("label_status") != "ok":
                    continue
                split_rows.append(row)
        rng = random.Random(f"{seed}:{split}")
        if max_rows_per_task_per_split is not None:
            by_task: dict[str, list[dict]] = defaultdict(list)
            for row in split_rows:
                by_task[row["task_type"]].append(row)
            for task_type, task_rows in sorted(by_task.items()):
                rng.shuffle(task_rows)
                rows.extend(task_rows[:max_rows_per_task_per_split])
        else:
            rng.shuffle(split_rows)
            rows.extend(split_rows[:limit_per_split])
    return rows


def generate_answer(
    model,
    tokenizer,
    row: dict,
    *,
    max_new_tokens: int,
    prompt_feature_mode: str = "none",
) -> str:
    import torch

    messages = [
        {
            "role": "system",
            "content": "Answer the IntersectionQA prompt with only the canonical answer token.",
        },
        {"role": "user", "content": prompt_for_mapping(row, mode=prompt_feature_mode).rstrip()},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = output[0, encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def normalize_answer(value: str) -> str:
    text = value.strip().splitlines()[0].strip()
    text = re.sub(r"^['\"`]+|['\"`.,;:]+$", "", text)
    return text.strip().lower()


def summarize(groups: dict[str, Counter]) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
        }
        for key, counts in sorted(groups.items())
    }


def classification_summary(
    confusion_by_task: dict[str, Counter[tuple[str, str]]],
    labels_by_task: dict[str, set[str]],
) -> dict[str, dict]:
    return {
        task_type: summarize_labels(confusion, sorted(labels_by_task[task_type]))
        for task_type, confusion in sorted(confusion_by_task.items())
    }


def summarize_labels(confusion: Counter[tuple[str, str]], labels: list[str]) -> dict:
    per_label = {}
    support_total = 0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    for label in labels:
        true_positive = confusion[(label, label)]
        false_positive = sum(count for (gold, pred), count in confusion.items() if pred == label and gold != label)
        false_negative = sum(count for (gold, pred), count in confusion.items() if gold == label and pred != label)
        support = true_positive + false_negative
        precision = safe_div(true_positive, true_positive + false_positive)
        recall = safe_div(true_positive, true_positive + false_negative)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": true_positive + false_positive,
        }
        support_total += support
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    label_count = len(labels)
    return {
        "macro_precision": safe_div(macro_precision, label_count),
        "macro_recall": safe_div(macro_recall, label_count),
        "macro_f1": safe_div(macro_f1, label_count),
        "weighted_precision": safe_div(weighted_precision, support_total),
        "weighted_recall": safe_div(weighted_recall, support_total),
        "weighted_f1": safe_div(weighted_f1, support_total),
        "per_label": per_label,
        "confusion": [
            {"answer": gold, "prediction": pred, "count": count}
            for (gold, pred), count in sorted(confusion.items())
        ],
    }


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
