from __future__ import annotations

import importlib
import sys
import types
from argparse import Namespace


def import_sft_module(monkeypatch):
    class DummyCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def is_bf16_supported() -> bool:
            return False

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

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=DummyCuda()))
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
