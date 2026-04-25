"""Tiny text-only SFT smoke test for IntersectionQA JSONL exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/intersectionqa_sft_smoke"))
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = _load_rows(args.dataset_dir, limit=args.max_rows)
    dataset = Dataset.from_list(
        [{"text": _chat_text(tokenizer, row), "answer": row["answer"]} for row in rows]
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
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        dataset_text_field="text",
        max_length=args.max_seq_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
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


def _chat_text(tokenizer, row: dict) -> str:
    messages = [
        {
            "role": "system",
            "content": "Answer the IntersectionQA prompt with only the canonical answer token.",
        },
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["answer"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


if __name__ == "__main__":
    main()
