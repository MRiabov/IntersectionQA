"""Text-only LoRA/QLoRA SFT runner for IntersectionQA JSONL exports."""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


DEFAULT_TASK_TYPES = (
    "binary_interference",
    "relation_classification",
    "volume_bucket",
    "clearance_bucket",
    "tolerance_fit",
    "pairwise_interference",
    "ranking_normalized_intersection",
)
PUBLIC_ROW_REQUIRED_KEYS = {
    "id",
    "dataset_version",
    "split",
    "task_type",
    "prompt",
    "answer",
    "script",
    "geometry_ids",
    "source",
    "base_object_pair_id",
    "assembly_group_id",
    "labels",
    "diagnostics",
    "difficulty_tags",
    "label_policy",
    "hashes",
    "metadata",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/intersectionqa_text_sft"))
    parser.add_argument("--train-splits", nargs="+", default=["train"])
    parser.add_argument("--eval-splits", nargs="+", default=["validation"])
    parser.add_argument("--task-types", nargs="+", default=list(DEFAULT_TASK_TYPES))
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    task_types = set(args.task_types)
    train_rows = load_rows(
        args.dataset_dir,
        args.train_splits,
        task_types=task_types,
        limit=args.max_train_rows,
        seed=args.seed,
    )
    eval_rows = load_rows(
        args.dataset_dir,
        args.eval_splits,
        task_types=task_types,
        limit=args.max_eval_rows,
        seed=args.seed + 1,
    )
    if not train_rows:
        raise ValueError("No training rows matched the requested splits and task types.")
    if not eval_rows:
        raise ValueError("No eval rows matched the requested splits and task types.")

    train_dataset = Dataset.from_list([to_sft_example(tokenizer, row) for row in train_rows])
    eval_dataset = Dataset.from_list([to_sft_example(tokenizer, row) for row in eval_rows])

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
    if hasattr(model, "config"):
        model.config.use_cache = False
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    training_args = build_sft_config(args)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    checkpoint = resolve_checkpoint(args)
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(args.output_dir / "adapter")

    payload = {
        "status": "ok",
        "model": args.model,
        "dataset_dir": str(args.dataset_dir),
        "train_splits": args.train_splits,
        "eval_splits": args.eval_splits,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "train_loss": result.training_loss,
        "eval_metrics": eval_metrics,
        "output_dir": str(args.output_dir),
        "resumed_from_checkpoint": str(checkpoint) if checkpoint else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
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
    }
    signature = inspect.signature(SFTConfig.__init__).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = "steps"
    return SFTConfig(**{key: value for key, value in kwargs.items() if key in signature})


def resolve_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_from_checkpoint:
        return args.resume_from_checkpoint
    if not args.resume:
        return None
    checkpoints = sorted(
        args.output_dir.glob("checkpoint-*"),
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
) -> list[dict]:
    rows: list[dict] = []
    for split in splits:
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = PUBLIC_ROW_REQUIRED_KEYS - set(row)
                if missing:
                    raise ValueError(f"{path}:{line_number}: missing public-row keys: {sorted(missing)}")
                if row["diagnostics"].get("label_status") != "ok":
                    continue
                if row["task_type"] not in task_types:
                    continue
                rows.append(row)
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def to_sft_example(tokenizer, row: dict) -> dict:
    messages = [
        {
            "role": "system",
            "content": "Answer the IntersectionQA prompt with only the canonical answer token.",
        },
        {"role": "user", "content": row["prompt"].rstrip()},
        {"role": "assistant", "content": row["answer"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    return {"text": text, "id": row["id"], "task_type": row["task_type"], "answer": row["answer"]}


if __name__ == "__main__":
    main()
