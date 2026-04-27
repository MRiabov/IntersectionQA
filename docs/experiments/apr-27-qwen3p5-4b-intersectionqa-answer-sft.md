# April 27, 2026 Qwen3.5 4B IntersectionQA Answer SFT Recovery

## Summary

- Date: April 27, 2026
- Remote instance: Vast `35682721`, `NVIDIA A100 80GB PCIe`
- Repo commit used for stopped run: `88c9635`
- Dataset: `MRiabov/IntersectionQA-90K`, materialized at `data/IntersectionQA-90K`
- Model: `unsloth/Qwen3.5-4B`
- Run: `answer_sft_full`
- Config: `configs/overnight_experiment_suite.yaml`
- Method: bf16/16-bit LoRA over language, attention, and MLP modules
- Context length: `2048`
- Packing: `--pack-tokenized`
- Effective loss: `packed_assistant_tokens_only`
- Batch: `--per-device-train-batch-size 32`, `--gradient-accumulation-steps 1`
- Planned budget: one epoch over public `train`
- Actual stop: early stop after step 400 quality regression
- Best checkpoint: `runs/answer_sft_full/checkpoint-best`, copied from `checkpoint-200`

The run was stopped intentionally because the fixed 64-row quality probe peaked
at step 200 and regressed at steps 300 and 400. The Vast instance was left
running for follow-up work.

## Why This Run Happened

The first April 27 SFT attempt accidentally trained with prompt-token loss. The
suite command used `--no-assistant-only-loss` without also using the packed
tokenized path, so the model spent most updates reconstructing the long prompt
instead of learning the short canonical answer.

The fix restored the April 25 answer-only recipe:

- use `--pack-tokenized`;
- keep manual label masking for prompt tokens (`-100`);
- use plain `transformers.Trainer` on packed token IDs;
- pass `assistant_only_loss=false` only to TRL config in packed mode, because
  the loss mask is already applied manually;
- record `loss_masking=packed_assistant_tokens_only` in `train_result.json`.

Regression coverage was added in `tests/test_text_sft_train_unsloth.py`.

## Commands

The run was launched by the experiment orchestrator:

```bash
python -m scripts.experiments.run_experiment_suite \
  configs/overnight_experiment_suite.yaml \
  --run answer_sft_full \
  --with-dependencies
```

The resolved SFT command in the manifest was:

```bash
python -m scripts.training.text_sft_train_unsloth \
  --dataset-dir data/IntersectionQA-90K \
  --output-dir runs/answer_sft_full \
  --pack-tokenized \
  --per-device-train-batch-size 32 \
  --gradient-accumulation-steps 1 \
  --eval-strategy no \
  --no-final-eval \
  --final-adapter-save-mode checkpoint \
  --resume \
  --quality-eval-steps 100
```

The run was stopped with `Ctrl-C` through the `iqa` tmux session after
`checkpoint-400` had been written and synced.

## Results

Quality evaluation used a fixed 64-row validation probe every 100 optimizer
steps.

| Step | Correct | Accuracy | Notes |
| --- | ---: | ---: | --- |
| 100 | 47 / 64 | 73.44% | First corrected full-run probe |
| 200 | 52 / 64 | 81.25% | Best observed checkpoint |
| 300 | 51 / 64 | 79.69% | Slight regression |
| 400 | 49 / 64 | 76.56% | Continued regression; early stop |

The best observed checkpoint was step 200. After stopping, `checkpoint-200` was
copied to `checkpoint-best` and uploaded.

## Artifact Locations

HF bucket target:

```text
hf://buckets/MRIabov/intersectionqa-qwen3p5-4b-grpo-artifacts/intersectionqa-overnight-2026-04-27/vast-35682721/runs/answer_sft_full
```

Preserved artifacts include:

- `checkpoint-200`
- `checkpoint-300`
- `checkpoint-400`
- `checkpoint-best`
- `quality_metrics.jsonl`
- `predictions/quality_step_100.jsonl`
- `predictions/quality_step_200.jsonl`
- `predictions/quality_step_300.jsonl`
- `predictions/quality_step_400.jsonl`
- `train_metrics.jsonl`
- `logs/stdout.log`
- `logs/stderr.log`
- `early_stop_summary.json`
- `status.json`

The final `status.json` marks the run as stopped with reason
`quality_regression_after_step_200`.

## Important Blockers Found

### Rejection-Sampled Reasoning Stage

The downstream `rejection_sampled_reasoning_dataset` stage was not allowed to
start automatically. Its current implementation is not guided/constrained
generation. It only:

- prompts for `<think>...</think><answer>...</answer>`;
- samples with normal `model.generate`;
- applies post-hoc rejection for answer correctness and nontrivial reasoning.

This does not satisfy the guided-generation expectations in
`specs/research-experiment-spec.md` for serious training/evaluation runs.
The automatic chain watcher was stopped before the stage could start.

The manifest was also corrected so this stage samples from the base
`unsloth/Qwen3.5-4B` model rather than from the answer-SFT adapter.

### One-Epoch Budget

The spec does call for answer-only full SFT to run one epoch over public
`train`, or an equivalent budget. This run was intentionally stopped early
because the validation probe peaked at step 200 and then regressed twice. For
paper reporting, this should be described as an early-stopped answer-SFT run,
not a completed one-epoch SFT.

## Follow-Ups

- Implement guided or constrained final-answer generation for the
  rejection-sampled reasoning dataset builder before launching reasoning SFT.
- Run a larger held-out evaluation for `checkpoint-best`, because the 64-row
  quality probe is useful for stop decisions but too small for a paper table.
- Decide whether to compare `checkpoint-200` against the April 25 SFT result
  before spending more A100 time on another supervised run.
