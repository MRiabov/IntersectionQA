# Qwen3.5 4B Tuning Notes

This note captures reusable Qwen3.5 4B tuning guidance for IntersectionQA and
IntersectionEdit. Individual dated experiment records live under
`docs/experiments/`.

## Experiment Records

- [April 25, 2026 Qwen3.5 4B IntersectionQA SFT](experiments/apr-25-qwen3p5-4b-intersectionqa-sft.md)
- [April 25-26, 2026 Qwen3.5 4B IntersectionEdit GRPO](experiments/apr-25-26-qwen3p5-4b-intersectionedit-grpo.md)

## Model Choice

`unsloth/Qwen3.5-4B` is the default budget-conscious model for current tuning
experiments. It supports full language, attention, and MLP LoRA while keeping
A100/H100 runs tractable. Larger 25-35B candidates were deprioritized because
attention-only LoRA or very slow step times made them poor fits for short budget
runs. The 9B path is viable but was still too slow for the original time budget.

## Hardware

Use an A100 80GB-class or H100-class instance for meaningful runs. Small 16GB
instances are useful only for smoke tests. For Vast instances, destroy failed or
finished contracts after artifacts are uploaded and verified.

Typical teardown command:

```bash
printf 'y\n' | rtk uv run vastai destroy instance <contract_id>
```

## Runtime Stack

The stable stack used by the later GRPO runs was:

- Torch `2.10.0+cu128`
- transformers `5.5.0`
- TRL `0.24.0`
- Unsloth `2026.4.8`
- bf16 LoRA on A100/H100

Qwen3.5 is processor-backed. For text-only training and evaluation, use the
underlying text tokenizer rather than passing the processor through generation or
`GRPOTrainer`; otherwise multimodal preprocessing overhead and warnings can leak
into text-only runs.

## SFT Guidance

For answer-only supervised fine-tuning on IntersectionQA-style rows:

- Keep `--assistant-only-loss` enabled so prompt tokens do not dominate the loss.
- Use manual packed-token training where the processor path prevents built-in
  packing.
- Short-answer SFT can score very well on exact-answer metrics, but it does not
  teach reusable reasoning behavior by itself.

For future reasoning-preserving SFT:

- Generate a medium-sized trace set from labels, metadata, and tool-derived
  geometry rules.
- Train the model to emit
  `<think>...geometry reasoning...</think><answer>...</answer>` reliably before
  RL.
- Early SFT traces may be around `128-256` tokens when they contain real
  geometry reasoning.

## GRPO Guidance

For GRPO/GSPO on IntersectionEdit:

- Use group-safe internal train/eval splits derived only from public `train`
  rows.
- Keep task and answer balancing in capped canaries; random caps can starve rare
  repair and movement families.
- Avoid duplicate generation-heavy eval paths. Pick either TRL internal eval or
  the quality callback for a given budget run. If using the quality callback, set
  `--eval-strategy no` and keep `--quality-eval-steps` at a deliberate cadence.
- Use batched quality generation via `--quality-generation-batch-size`.
- Try larger `--per-device-train-batch-size` and `--generation-batch-size` when
  VRAM is underused.
- Treat vLLM guided decoding / structured outputs as a separate canary because it
  changes the rollout distribution and learning problem.

## Reasoning Preservation

Do not rely on a blind minimum generation length. It can reward filler. Prefer a
structured reasoning-length band:

- full reward requires a nonempty `<think>` section;
- reasoning length is bounded, initially around `64-192` useful tokens for RL;
- repeated or filler reasoning is penalized;
- the final answer remains short and schema-valid.

## Logging Requirements

Future GRPO runs should log enough data to support post-mortems and SFT mining:

- every quality-eval prediction;
- sampled rollout completions for debug/eval batches;
- reward components per completion;
- prompt token length, completion length, and truncation status;
- group-relative ranks or selected winners;
- best checkpoint copied to `checkpoint-best`.

Do not rely only on `save_total_limit`; it can delete the best checkpoint before
post-mortem analysis.

## Known Failure Modes

- 120GB disks are too small to keep multiple large Qwen model caches. Clear failed
  model caches before switching large candidates.
- Generic PEFT QLoRA on `Qwen/Qwen3.6-35B-A3B` OOMed on A100 80GB during
  k-bit preparation. Use Unsloth bf16 LoRA for MoE experiments.
- Unsloth bf16 LoRA on `unsloth/Qwen3.6-35B-A3B` with attention and MLP modules
  enabled fit at 2048 context but trained about 953M parameters and was too slow.
- Dense 27B QLoRA and Qwen3.5 9B are viable but were too slow for the original
  four-hour full-dataset budget.
