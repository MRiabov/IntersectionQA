from __future__ import annotations

import importlib
import sys
import types
from argparse import Namespace
from collections import Counter

import pytest


def import_sft_module(monkeypatch):
    class DummyCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def is_bf16_supported() -> bool:
            return False

    class DummyTorch:
        cuda = DummyCuda()
        long = "long"

        @staticmethod
        def tensor(values, dtype=None):
            return {"values": values, "dtype": dtype}

    class DummySFTConfig:
        def __init__(
            self,
            output_dir=None,
            packing=None,
            assistant_only_loss=None,
            eval_strategy=None,
            **kwargs,
        ):
            self.output_dir = output_dir
            self.packing = packing
            self.assistant_only_loss = assistant_only_loss
            self.eval_strategy = eval_strategy
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "torch", DummyTorch)
    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(FastModel=object()))
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=object()))
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=object()))
    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=DummySFTConfig, SFTTrainer=object))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(Trainer=object, TrainerCallback=object),
    )
    sys.modules.pop("scripts.training.text_sft_train_unsloth", None)
    return importlib.import_module("scripts.training.text_sft_train_unsloth")


class CharacterTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(char) for char in text]}


def test_pack_tokenized_examples_masks_prompt_tokens(monkeypatch):
    sft = import_sft_module(monkeypatch)

    packed = sft.pack_tokenized_examples(
        CharacterTokenizer(),
        [{"text": "promptanswer", "prompt_text": "prompt"}],
        max_seq_length=32,
    )

    assert len(packed) == 1
    assert packed[0]["input_ids"] == [ord(char) for char in "promptanswer"] + [0]
    assert packed[0]["labels"] == [-100] * len("prompt") + [ord(char) for char in "answer"] + [0]


def test_pack_tokenized_examples_drops_prompt_only_chunks(monkeypatch):
    sft = import_sft_module(monkeypatch)

    packed = sft.pack_tokenized_examples(
        CharacterTokenizer(),
        [{"text": "longpromptanswer", "prompt_text": "longprompt"}],
        max_seq_length=6,
    )

    assert packed
    assert all(any(label != -100 for label in chunk["labels"]) for chunk in packed)
    assert packed[0]["input_ids"] == [ord(char) for char in "omptan"]
    assert packed[0]["labels"] == [-100, -100, -100, -100, ord("a"), ord("n")]


def test_pack_tokenized_examples_fails_when_context_never_reaches_answer(monkeypatch):
    sft = import_sft_module(monkeypatch)

    with pytest.raises(ValueError, match="No supervised packed chunks"):
        sft.pack_tokenized_examples(
            CharacterTokenizer(),
            [{"text": "prompt", "prompt_text": "prompt"}],
            max_seq_length=4,
        )


def test_causal_lm_collator_can_pad_to_fixed_length(monkeypatch):
    sft = import_sft_module(monkeypatch)

    batch = sft.causal_lm_collator(
        [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]}],
        pad_token_id=0,
        pad_to_length=5,
    )

    assert batch["input_ids"]["values"] == [[1, 2, 0, 0, 0]]
    assert batch["attention_mask"]["values"] == [[1, 1, 0, 0, 0]]
    assert batch["labels"]["values"] == [[-100, 2, -100, -100, -100]]


def test_summarize_quality_format_reports_invalid_rate(monkeypatch):
    sft = import_sft_module(monkeypatch)

    summary = sft.summarize_quality_format(
        Counter(
            {
                "total": 4,
                "parse_valid": 3,
                "invalid": 1,
                "parsed_correct": 2,
                "answer_tag": 1,
                "reasoning_format": 1,
            }
        )
    )

    assert summary["parse_valid_rate"] == 0.75
    assert summary["invalid_rate"] == 0.25
    assert summary["parsed_accuracy"] == 0.5
    assert summary["answer_tag_rate"] == 0.25


def test_pack_tokenized_uses_manual_masking_not_trl_assistant_loss(monkeypatch, tmp_path):
    sft = import_sft_module(monkeypatch)
    args = Namespace(
        output_dir=tmp_path,
        max_seq_length=2048,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        max_steps=20,
        num_train_epochs=1.0,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        seed=3407,
        packing=False,
        pack_tokenized=True,
        assistant_only_loss=True,
        eval_strategy="no",
    )

    config = sft.build_sft_config(args)

    assert config.assistant_only_loss is False
    assert sft.loss_masking_mode(args) == "packed_assistant_tokens_only"
