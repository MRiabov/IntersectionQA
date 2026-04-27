"""Unsloth text-only LoRA SFT runner for IntersectionQA JSONL exports."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import inspect
import json
import random
import shutil
from pathlib import Path
from typing import Any

from unsloth import FastModel
from peft import PeftModel
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from transformers import Trainer, TrainerCallback

from intersectionqa.evaluation.parsing import parse_answer
from intersectionqa.evaluation.parsing import canonical_answer_candidate
from intersectionqa.training.prompt_features import PROMPT_FEATURE_MODES, prompt_for_mapping


DEFAULT_TASK_TYPES = (
    "binary_interference",
    "clearance_bucket",
    "pairwise_interference",
    "ranking_normalized_intersection",
    "relation_classification",
    "tolerance_fit",
    "volume_bucket",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", default="unsloth/Qwen3.5-4B")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/intersectionqa_unsloth_sft"))
    parser.add_argument("--train-splits", nargs="+", default=["train"])
    parser.add_argument("--eval-splits", nargs="+", default=["validation"])
    parser.add_argument("--task-types", nargs="+", default=list(DEFAULT_TASK_TYPES))
    parser.add_argument("--require-counterfactual-group", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt-feature-mode", choices=PROMPT_FEATURE_MODES, default="none")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--eval-strategy", choices=["no", "steps"], default="steps")
    parser.add_argument("--final-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pack-tokenized", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--assistant-only-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--adapter-init-dir", type=Path)
    parser.add_argument("--finetune-language-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-attention-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune-mlp-modules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metrics-log-file", default="train_metrics.jsonl")
    parser.add_argument("--print-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-eval-steps", type=int, default=0)
    parser.add_argument("--quality-eval-max-rows", type=int, default=64)
    parser.add_argument("--quality-metrics-log-file", default="quality_metrics.jsonl")
    parser.add_argument("--quality-predictions-dir", default="predictions")
    parser.add_argument("--quality-max-new-tokens", type=int, default=16)
    parser.add_argument("--final-adapter-save-mode", choices=["trainer", "checkpoint"], default="trainer")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()

    task_types = set(args.task_types)
    train_rows = load_rows(
        args.dataset_dir,
        args.train_splits,
        task_types=task_types,
        limit=args.max_train_rows,
        seed=args.seed,
        require_counterfactual_group=args.require_counterfactual_group,
    )
    eval_rows = load_rows(
        args.dataset_dir,
        args.eval_splits,
        task_types=task_types,
        limit=args.max_eval_rows,
        seed=args.seed + 1,
        require_counterfactual_group=False,
    )
    if not train_rows:
        raise ValueError("No training rows matched the requested splits and task types.")
    if not eval_rows:
        raise ValueError("No eval rows matched the requested splits and task types.")

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=args.load_in_16bit,
        full_finetuning=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_examples = [
        to_sft_example(tokenizer, row, prompt_feature_mode=args.prompt_feature_mode)
        for row in train_rows
    ]
    eval_examples = [
        to_sft_example(tokenizer, row, prompt_feature_mode=args.prompt_feature_mode)
        for row in eval_rows
    ]
    if args.pack_tokenized:
        train_dataset = Dataset.from_list(pack_tokenized_examples(tokenizer, train_examples, args.max_seq_length))
        eval_dataset = Dataset.from_list(pack_tokenized_examples(tokenizer, eval_examples, args.max_seq_length))
    else:
        train_dataset = Dataset.from_list(train_examples)
        eval_dataset = Dataset.from_list(eval_examples)

    if args.adapter_init_dir:
        model = PeftModel.from_pretrained(model, str(args.adapter_init_dir), is_trainable=True)
    else:
        model = FastModel.get_peft_model(
            model,
            finetune_language_layers=args.finetune_language_layers,
            finetune_attention_modules=args.finetune_attention_modules,
            finetune_mlp_modules=args.finetune_mlp_modules,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            bias="none",
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
        )

    FastModel.for_training(model)
    if args.pack_tokenized:
        trainer = Trainer(
            model=model,
            args=build_sft_config(args),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=lambda features: causal_lm_collator(features, tokenizer.pad_token_id),
        )
    else:
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "args": build_sft_config(args),
        }
        trainer_signature = inspect.signature(SFTTrainer.__init__).parameters
        if "processing_class" in trainer_signature:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_signature:
            trainer_kwargs["tokenizer"] = tokenizer
        trainer = SFTTrainer(**trainer_kwargs)
    checkpoint = resolve_checkpoint(args)
    metrics_log_path = args.output_dir / args.metrics_log_file
    trainer.add_callback(MetricsJsonlCallback(metrics_log_path, append=checkpoint is not None, print_metrics=args.print_metrics))
    if args.quality_eval_steps > 0:
        quality_rows = eval_rows[: args.quality_eval_max_rows]
        trainer.add_callback(
            GenerationQualityCallback(
                path=args.output_dir / args.quality_metrics_log_file,
                predictions_dir=args.output_dir / args.quality_predictions_dir,
                rows=quality_rows,
                tokenizer=tokenizer,
                every_steps=args.quality_eval_steps,
                max_new_tokens=args.quality_max_new_tokens,
                prompt_feature_mode=args.prompt_feature_mode,
                append=checkpoint is not None,
                print_metrics=args.print_metrics,
            )
        )
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    eval_metrics = trainer.evaluate() if args.final_eval else {}
    adapter_dir = args.output_dir / "adapter"
    if args.final_adapter_save_mode == "checkpoint":
        checkpoint_adapter = latest_checkpoint(args.output_dir)
        if checkpoint_adapter is None:
            raise RuntimeError("final adapter save mode 'checkpoint' requires at least one checkpoint")
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        shutil.copytree(checkpoint_adapter, adapter_dir)
    else:
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(adapter_dir)

    payload = {
        "status": "ok",
        "model": args.model,
        "dataset_dir": str(args.dataset_dir),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_sequences": len(train_dataset),
        "eval_sequences": len(eval_dataset),
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "train_loss": result.training_loss,
        "eval_metrics": eval_metrics,
        "eval_strategy": args.eval_strategy,
        "final_eval": args.final_eval,
        "output_dir": str(args.output_dir),
        "resumed_from_checkpoint": str(checkpoint) if checkpoint else None,
        "load_in_4bit": args.load_in_4bit,
        "load_in_16bit": args.load_in_16bit,
        "finetune_language_layers": args.finetune_language_layers,
        "finetune_attention_modules": args.finetune_attention_modules,
        "finetune_mlp_modules": args.finetune_mlp_modules,
        "adapter_init_dir": str(args.adapter_init_dir) if args.adapter_init_dir else None,
        "require_counterfactual_group": args.require_counterfactual_group,
        "prompt_feature_mode": args.prompt_feature_mode,
        "packing": args.packing,
        "pack_tokenized": args.pack_tokenized,
        "assistant_only_loss": args.assistant_only_loss,
        "trl_assistant_only_loss": trl_assistant_only_loss(args),
        "loss_masking": loss_masking_mode(args),
        "metrics_log_file": args.metrics_log_file,
        "quality_eval_steps": args.quality_eval_steps,
        "quality_eval_max_rows": args.quality_eval_max_rows,
        "quality_metrics_log_file": args.quality_metrics_log_file,
        "quality_predictions_dir": args.quality_predictions_dir,
        "final_adapter_save_mode": args.final_adapter_save_mode,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_result.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_sft_config(args: argparse.Namespace) -> SFTConfig:
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "dataset_text_field": "text",
        "max_length": args.max_seq_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_strategy": "steps",
        "report_to": [],
        "seed": args.seed,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "optim": "adamw_8bit",
        "weight_decay": 0.001,
        "lr_scheduler_type": "linear",
        "packing": args.packing,
        "assistant_only_loss": trl_assistant_only_loss(args),
    }
    signature = inspect.signature(SFTConfig.__init__).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = args.eval_strategy
    elif "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = args.eval_strategy
    return SFTConfig(**{key: value for key, value in kwargs.items() if key in signature})


def trl_assistant_only_loss(args: argparse.Namespace) -> bool:
    if args.pack_tokenized:
        return False
    return bool(args.assistant_only_loss)


def loss_masking_mode(args: argparse.Namespace) -> str:
    if args.pack_tokenized:
        return "packed_assistant_tokens_only"
    if args.assistant_only_loss:
        return "trl_assistant_only_loss"
    return "full_sequence_prompt_and_answer"


def pack_tokenized_examples(tokenizer: Any, examples: list[dict], max_seq_length: int) -> list[dict]:
    text_tokenizer = get_text_tokenizer(tokenizer)
    eos_token_id = text_tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for token packing.")
    token_stream: list[int] = []
    label_stream: list[int] = []
    packed: list[dict] = []
    for example in examples:
        full_token_ids = encode_text(text_tokenizer, example["text"])
        if example.get("prompt_text"):
            prompt_token_ids = encode_text(text_tokenizer, example["prompt_text"])
        else:
            prompt_token_ids = []
        label_ids = full_token_ids.copy()
        supervised_start = min(len(prompt_token_ids), len(label_ids))
        label_ids[:supervised_start] = [-100] * supervised_start
        token_stream.extend(full_token_ids)
        label_stream.extend(label_ids)
        token_stream.append(eos_token_id)
        label_stream.append(eos_token_id)
        while len(token_stream) >= max_seq_length:
            chunk = token_stream[:max_seq_length]
            label_chunk = label_stream[:max_seq_length]
            packed.append(
                {
                    "input_ids": chunk,
                    "attention_mask": [1] * len(chunk),
                    "labels": label_chunk,
                }
            )
            token_stream = token_stream[max_seq_length:]
            label_stream = label_stream[max_seq_length:]
    if token_stream:
        packed.append(
            {
                "input_ids": token_stream,
                "attention_mask": [1] * len(token_stream),
                "labels": label_stream,
            }
        )
    return packed


def get_text_tokenizer(tokenizer_or_processor: Any) -> Any:
    return getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)


def encode_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text=text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        return list(input_ids[0])
    return list(input_ids)


def causal_lm_collator(features: list[dict], pad_token_id: int | None) -> dict[str, torch.Tensor]:
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for packed training.")
    max_len = max(len(feature["input_ids"]) for feature in features)
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    for feature in features:
        input_ids = list(feature["input_ids"])
        attention_mask = list(feature["attention_mask"])
        labels = list(feature["labels"])
        pad_len = max_len - len(input_ids)
        batch_input_ids.append(input_ids + [pad_token_id] * pad_len)
        batch_attention_mask.append(attention_mask + [0] * pad_len)
        batch_labels.append(labels + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }


class MetricsJsonlCallback(TrainerCallback):
    def __init__(self, path: Path, *, append: bool, print_metrics: bool) -> None:
        self.path = path
        self.print_metrics = print_metrics
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not append and self.path.exists():
            self.path.unlink()

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not logs:
            return
        payload = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "global_step": state.global_step,
            "epoch": state.epoch,
            **{key: json_safe(value) for key, value in logs.items()},
        }
        line = json.dumps(payload, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        if self.print_metrics:
            print(f"[metrics] {line}", flush=True)


class GenerationQualityCallback(TrainerCallback):
    def __init__(
        self,
        path: Path,
        predictions_dir: Path,
        *,
        rows: list[dict],
        tokenizer: Any,
        every_steps: int,
        max_new_tokens: int,
        prompt_feature_mode: str,
        append: bool,
        print_metrics: bool,
    ) -> None:
        if every_steps <= 0:
            raise ValueError("every_steps must be positive")
        self.path = path
        self.predictions_dir = predictions_dir
        self.rows = rows
        self.tokenizer = tokenizer
        self.every_steps = every_steps
        self.max_new_tokens = max_new_tokens
        self.prompt_feature_mode = prompt_feature_mode
        self.print_metrics = print_metrics
        self._seen_steps: set[int] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        if not append and self.path.exists():
            self.path.unlink()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        step = int(state.global_step or 0)
        if step <= 0 or step % self.every_steps != 0 or step in self._seen_steps:
            return
        model = kwargs.get("model")
        if model is None:
            return
        self._seen_steps.add(step)
        payload = evaluate_generation_quality(
            model,
            self.tokenizer,
            self.rows,
            max_new_tokens=self.max_new_tokens,
            prompt_feature_mode=self.prompt_feature_mode,
            prediction_path=self.predictions_dir / f"quality_step_{step}.jsonl",
        )
        payload.update(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "global_step": step,
                "epoch": state.epoch,
            }
        )
        line = json.dumps(payload, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        if self.print_metrics:
            print(f"[quality] {line}", flush=True)


def evaluate_generation_quality(
    model: Any,
    tokenizer: Any,
    rows: list[dict],
    *,
    max_new_tokens: int,
    prompt_feature_mode: str = "none",
    prediction_path: Path | None = None,
) -> dict:
    by_split: dict[str, Counter] = defaultdict(Counter)
    by_task: dict[str, Counter] = defaultdict(Counter)
    labels_by_task: dict[str, set[str]] = defaultdict(set)
    confusion_by_task: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    examples = []
    prediction_records = []
    was_training = bool(getattr(model, "training", False))
    FastModel.for_inference(model)
    model.eval()
    try:
        for row in rows:
            prediction = generate_canonical_answer(
                model,
                tokenizer,
                row,
                max_new_tokens=max_new_tokens,
                prompt_feature_mode=prompt_feature_mode,
            )
            normalized_prediction = normalize_answer(prediction)
            normalized_answer = normalize_answer(str(row["answer"]))
            exact = normalized_prediction == normalized_answer
            candidate, _format_components = canonical_answer_candidate(prediction)
            parsed = parse_answer(row["task_type"], candidate)
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
                    "correct": parsed == row["answer"],
                }
            )
            by_split[row["split"]]["total"] += 1
            by_split[row["split"]]["correct"] += int(exact)
            by_task[row["task_type"]]["total"] += 1
            by_task[row["task_type"]]["correct"] += int(exact)
            labels_by_task[row["task_type"]].add(normalized_answer)
            labels_by_task[row["task_type"]].add(normalized_prediction)
            confusion_by_task[row["task_type"]][(normalized_answer, normalized_prediction)] += 1
            if len(examples) < 10 and not exact:
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
    finally:
        if was_training:
            FastModel.for_training(model)
            model.train()
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("w", encoding="utf-8") as handle:
            for record in prediction_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "row_count": len(rows),
        "prediction_file": str(prediction_path) if prediction_path is not None else None,
        "by_split": summarize_accuracy(by_split),
        "by_task": summarize_accuracy(by_task),
        "classification_by_task": classification_summary(confusion_by_task, labels_by_task),
        "mismatches": examples,
    }


def generate_canonical_answer(
    model: Any,
    tokenizer_or_processor: Any,
    row: dict,
    *,
    max_new_tokens: int,
    prompt_feature_mode: str = "none",
) -> str:
    text_tokenizer = get_text_tokenizer(tokenizer_or_processor)
    prompt = prompt_only_text(tokenizer_or_processor, row, prompt_feature_mode=prompt_feature_mode)
    encoded = text_tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=text_tokenizer.pad_token_id,
            eos_token_id=text_tokenizer.eos_token_id,
        )
    completion = output[0, encoded["input_ids"].shape[-1] :]
    return text_tokenizer.decode(completion, skip_special_tokens=True).strip()


def prompt_only_text(tokenizer: Any, row: dict, *, prompt_feature_mode: str = "none") -> str:
    prompt = prompt_for_mapping(row, mode=prompt_feature_mode).rstrip()
    user_message = {"role": "user", "content": prompt}
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template([user_message], tokenize=False, add_generation_prompt=True)
    return f"### User:\n{prompt}\n\n### Assistant:\n"


def normalize_answer(value: str) -> str:
    import re

    candidate, _format_components = canonical_answer_candidate(value)
    text = candidate.strip().splitlines()[0].strip()
    text = re.sub(r"^['\"`]+|['\"`.,;:]+$", "", text)
    return text.strip().lower()


def summarize_accuracy(groups: dict[str, Counter]) -> dict[str, dict[str, float | int]]:
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


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float | int | str | bool) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def resolve_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_from_checkpoint:
        return args.resume_from_checkpoint
    if not args.resume:
        return None
    return latest_checkpoint(args.output_dir)


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else -1,
    )
    return checkpoints[-1] if checkpoints else None


def load_rows(
    dataset_dir: Path,
    splits: list[str],
    *,
    task_types: set[str],
    limit: int | None,
    seed: int,
    require_counterfactual_group: bool,
) -> list[dict]:
    rows: list[dict] = []
    for split in splits:
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("task_type") not in task_types:
                    continue
                if require_counterfactual_group and not row.get("counterfactual_group_id"):
                    continue
                rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit] if limit else rows


def to_sft_example(tokenizer: Any, row: dict, *, prompt_feature_mode: str = "none") -> dict:
    prompt = prompt_for_mapping(row, mode=prompt_feature_mode)
    user_message = {"role": "user", "content": prompt}
    assistant_text = str(row.get("target_text", row["answer"]))
    assistant_message = {"role": "assistant", "content": assistant_text}
    messages = [
        user_message,
        assistant_message,
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template([user_message], tokenize=False, add_generation_prompt=True)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        prompt_text = f"### User:\n{prompt}\n\n### Assistant:\n"
        text = f"{prompt_text}{assistant_text}"
    return {
        "text": text,
        "prompt_text": prompt_text,
        "id": row["id"],
        "task_type": row["task_type"],
        "split": row["split"],
    }


if __name__ == "__main__":
    main()
