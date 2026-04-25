"""Unsloth/TRL GRPO runner for IntersectionQA and IntersectionEdit JSONL exports."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import inspect
import json
import random
from pathlib import Path
from typing import Any

from unsloth import FastModel
from datasets import Dataset
import torch
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from intersectionqa.evaluation.metrics import Prediction, evaluate_predictions
from intersectionqa.evaluation.rewards import reward_from_fields
from intersectionqa.schema import PublicTaskRow

SYSTEM_PROMPT = (
    "Solve the CAD spatial-reasoning task. Think briefly in <think>...</think>, "
    "then put only the canonical answer string inside <answer>...</answer>."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", default="unsloth/Qwen3.5-4B")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/intersectionqa_edit_grpo"))
    parser.add_argument("--train-splits", nargs="+", default=["inner_train", "train"])
    parser.add_argument("--eval-splits", nargs="+", default=["inner_eval", "validation"])
    parser.add_argument("--task-types", nargs="+")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-16bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--importance-sampling-level", choices=["token", "sequence"], default="sequence")
    parser.add_argument("--loss-type", default="dapo")
    parser.add_argument("--scale-rewards", default="group")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--metrics-log-file", default="train_metrics.jsonl")
    parser.add_argument("--quality-metrics-log-file", default="quality_metrics.jsonl")
    parser.add_argument("--quality-eval-steps", type=int, default=50)
    parser.add_argument("--quality-eval-max-rows", type=int, default=64)
    parser.add_argument("--quality-max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()

    task_types = set(args.task_types) if args.task_types else None
    train_rows = load_rows(args.dataset_dir, args.train_splits, task_types=task_types, limit=args.max_train_rows, seed=args.seed)
    eval_rows = load_rows(args.dataset_dir, args.eval_splits, task_types=task_types, limit=args.max_eval_rows, seed=args.seed + 1)
    if not train_rows:
        raise ValueError("No training rows matched the requested splits and task types.")
    if not eval_rows:
        raise ValueError("No eval rows matched the requested splits and task types.")

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_prompt_length + args.max_completion_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=args.load_in_16bit,
        full_finetuning=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastModel.get_peft_model(
        model,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    FastModel.for_training(model)

    train_dataset = Dataset.from_list([to_grpo_example(row) for row in train_rows])
    eval_dataset = Dataset.from_list([to_grpo_example(row) for row in eval_rows])
    training_args = build_grpo_config(args)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "reward_funcs": row_reward,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    trainer_signature = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)

    checkpoint = resolve_checkpoint(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.add_callback(MetricsJsonlCallback(args.output_dir / args.metrics_log_file, append=checkpoint is not None))
    if args.quality_eval_steps > 0:
        trainer.add_callback(
            ReasoningQualityCallback(
                path=args.output_dir / args.quality_metrics_log_file,
                rows=eval_rows[: args.quality_eval_max_rows],
                tokenizer=tokenizer,
                every_steps=args.quality_eval_steps,
                max_new_tokens=args.quality_max_new_tokens,
                append=checkpoint is not None,
            )
        )
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    adapter_dir = args.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    payload = {
        "status": "ok",
        "model": args.model,
        "dataset_dir": str(args.dataset_dir),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_task_counts": dict(sorted(Counter(str(row.task_type) for row in train_rows).items())),
        "eval_task_counts": dict(sorted(Counter(str(row.task_type) for row in eval_rows).items())),
        "max_steps": args.max_steps,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "importance_sampling_level": args.importance_sampling_level,
        "loss_type": args.loss_type,
        "scale_rewards": args.scale_rewards,
        "train_loss": result.training_loss,
        "output_dir": str(args.output_dir),
        "resumed_from_checkpoint": str(checkpoint) if checkpoint else None,
    }
    (args.output_dir / "train_result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_grpo_config(args: argparse.Namespace) -> GRPOConfig:
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_strategy": "steps",
        "temperature": args.temperature,
        "report_to": [],
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    }
    signature = inspect.signature(GRPOConfig).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = "steps"
    if "eval_steps" in signature:
        kwargs["eval_steps"] = args.eval_steps
    optional = {
        "importance_sampling_level": args.importance_sampling_level,
        "loss_type": args.loss_type,
        "scale_rewards": args.scale_rewards,
    }
    for key, value in optional.items():
        if key in signature:
            kwargs[key] = value
    return GRPOConfig(**kwargs)


def row_reward(completions, answer, id, task_type, metadata, **_kwargs) -> list[float]:
    rewards: list[float] = []
    for completion, expected, row_id, row_task_type, row_metadata in zip(
        completions,
        answer,
        id,
        task_type,
        metadata,
        strict=True,
    ):
        text = completion[-1].get("content", "") if isinstance(completion, list) and completion else str(completion)
        result = reward_from_fields(
            row_id=row_id,
            task_type=row_task_type,
            answer=expected,
            metadata=json.loads(row_metadata) if isinstance(row_metadata, str) else row_metadata,
            output=text,
        )
        rewards.append(result.reward)
    return rewards


def to_grpo_example(row: PublicTaskRow) -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row.prompt},
        ],
        "answer": row.answer,
        "id": row.id,
        "task_type": str(row.task_type),
        "metadata": json.dumps(row.metadata, sort_keys=True),
    }


def load_rows(
    dataset_dir: Path,
    splits: list[str],
    *,
    task_types: set[str] | None,
    limit: int | None,
    seed: int,
) -> list[PublicTaskRow]:
    rows: list[PublicTaskRow] = []
    for split in splits:
        path = dataset_dir / f"{split}.jsonl"
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        rows.append(PublicTaskRow.model_validate_json(line))
                    except Exception as exc:
                        raise ValueError(f"{path}:{line_number}: invalid public row: {exc}") from exc
    if task_types is not None:
        rows = [row for row in rows if str(row.task_type) in task_types]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit] if limit else rows


def resolve_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_from_checkpoint:
        return args.resume_from_checkpoint
    if not args.resume:
        return None
    checkpoints = sorted(args.output_dir.glob("checkpoint-*"), key=lambda path: path.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


class MetricsJsonlCallback(TrainerCallback):
    def __init__(self, path: Path, *, append: bool) -> None:
        self.path = path
        self.append = append

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.append:
            self.path.write_text("", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step": state.global_step,
            **logs,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


class ReasoningQualityCallback(TrainerCallback):
    def __init__(
        self,
        *,
        path: Path,
        rows: list[PublicTaskRow],
        tokenizer: Any,
        every_steps: int,
        max_new_tokens: int,
        append: bool,
    ) -> None:
        self.path = path
        self.rows = rows
        self.tokenizer = tokenizer
        self.every_steps = every_steps
        self.max_new_tokens = max_new_tokens
        self.append = append
        self._last_step = -1

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.append:
            self.path.write_text("", encoding="utf-8")

    def on_step_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
        if model is None or not self.rows or state.global_step == self._last_step:
            return
        if state.global_step == 0 or state.global_step % self.every_steps != 0:
            return
        self._last_step = state.global_step
        predictions = generate_predictions(
            model,
            self.tokenizer,
            self.rows,
            max_new_tokens=self.max_new_tokens,
        )
        metrics = evaluate_predictions(self.rows, predictions)
        reward_values = [
            reward_from_fields(
                row_id=row.id,
                task_type=row.task_type,
                answer=row.answer,
                metadata=row.metadata,
                output=prediction.output,
            ).reward
            for row, prediction in zip(self.rows, predictions, strict=True)
        ]
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step": state.global_step,
            "rows": len(self.rows),
            "reward_mean": sum(reward_values) / len(reward_values) if reward_values else 0.0,
            "metrics": [metric.__dict__ for metric in metrics],
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def generate_predictions(
    model: Any,
    tokenizer: Any,
    rows: list[PublicTaskRow],
    *,
    max_new_tokens: int,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row.prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            completion_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
            predictions.append(Prediction(row_id=row.id, output=tokenizer.decode(completion_ids, skip_special_tokens=True).strip()))
    model.train()
    return predictions


if __name__ == "__main__":
    main()
