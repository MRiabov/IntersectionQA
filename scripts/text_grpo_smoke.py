"""Tiny GRPO reward-learning smoke test for IntersectionQA JSONL exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from intersectionqa.evaluation.rewards import reward_from_fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/intersectionqa_grpo_smoke"))
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = _load_rows(args.dataset_dir, limit=args.max_rows)
    dataset = Dataset.from_list(
        [
            {
                "prompt": [
                    {
                        "role": "system",
                        "content": "Answer the IntersectionQA prompt with only the canonical answer token.",
                    },
                    {"role": "user", "content": row["prompt"]},
                ],
                "answer": row["answer"],
                "id": row["id"],
                "task_type": row["task_type"],
                "metadata": row.get("metadata", {}),
            }
            for row in rows
        ]
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=16,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=_row_reward,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    result = trainer.train()
    print(
        json.dumps(
            {
                "status": "ok",
                "model": args.model,
                "rows": len(rows),
                "max_steps": args.max_steps,
                "train_loss": result.training_loss,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _row_reward(completions, answer, id, task_type, metadata, **_kwargs) -> list[float]:
    rewards = []
    for completion, expected, row_id, row_task_type, row_metadata in zip(
        completions,
        answer,
        id,
        task_type,
        metadata,
        strict=True,
    ):
        if isinstance(completion, list) and completion:
            text = completion[-1].get("content", "")
        else:
            text = str(completion)
        result = reward_from_fields(
            row_id=row_id,
            task_type=row_task_type,
            answer=expected,
            metadata=row_metadata,
            output=text,
        )
        rewards.append(result.reward)
    return rewards


def _load_rows(dataset_dir: Path, *, limit: int) -> list[dict]:
    rows: list[dict] = []
    for split in ("train", "validation", "test_random", "test_object_pair_heldout", "test_near_boundary"):
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    return rows
    return rows


if __name__ == "__main__":
    main()
