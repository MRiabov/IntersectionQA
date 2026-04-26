# Qwen3.5 4B Tuning Experiment

This experiment record keeps the Qwen3.5 4B tuning decisions and launch state
out of the general text fine-tuning runbook.

## Summary

### April 25, 2026 GRPO/GSPO Pilot

- Dataset: `data/intersectionedit_grpo_pilot_inner_all`
- Source release candidate: `data/intersectionedit_grpo_pilot`
- Model: `unsloth/Qwen3.5-4B`
- Method: Unsloth/TRL GRPO with sequence-level importance sampling
  (`--importance-sampling-level sequence`, GSPO-style) and DAPO loss
- Training split hygiene: public `train` only, then group-safe
  `inner_train`/`inner_eval`
- Internal rows: 1,627 train, 200 eval
- Public release-candidate rows: 2,565 total from 500 CADEvolve-backed geometry
  records
- Public validation: `scripts.validate_dataset` passed and leakage audit status
  is `pass`
- Status: A100 canaries and a 50-step continuation completed, with
  checkpointed artifacts. The 300-step pilot remains blocked because accuracy is
  still concentrated in easy tolerance/relation rows and does not cover repair,
  movement, or bucket tasks.

The initial `configs/repair_smoke.yaml` path was confirmed to be a tiny smoke
configuration (`geometry_limit: 100`). Exact `axis_aligned_repair` and candidate
tasks were too slow on raw CADEvolve rows for the overnight budget, so the pilot
uses fast mixed QA+Edit task families:

- `binary_interference`
- `relation_classification`
- `volume_bucket`
- `clearance_bucket`
- `tolerance_fit`
- `repair_direction`
- `repair_translation`
- `target_clearance_move`
- `target_contact_move`
- `centroid_distance_move`

Prepared split report:

```json
{
  "input_rows": 1827,
  "selected_rows": 1827,
  "inner_train_rows": 1627,
  "inner_eval_rows": 200,
  "scope": "all"
}
```

Vast canary environment:

- Contract: `35599309`
- GPU: `NVIDIA A100-SXM4-80GB`
- Host/port used: `80.188.223.202:14366`
- Approx price: `$1.10/hr`
- Working stack: Torch `2.11.0+cu130`, CUDA toolkit `13.0`,
  transformers `5.6.2`, TRL `0.23.0`, Unsloth `2026.4.8`
- Required runtime environment:
  `LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cu13/lib`
  plus `PYTHONPATH=/root/IntersectionQA`

First canary command:

```bash
cd /root/IntersectionQA
nohup python scripts/text_grpo_train_unsloth.py \
  --dataset-dir /root/intersectionedit_grpo_pilot_inner_all \
  --model unsloth/Qwen3.5-4B \
  --output-dir /root/outputs/grpo_qwen3p5_4b_intersectionqa_edit_canary \
  --train-splits inner_train \
  --eval-splits inner_eval \
  --max-train-rows 128 \
  --max-eval-rows 64 \
  --max-steps 20 \
  --max-prompt-length 2048 \
  --max-completion-length 512 \
  --num-generations 4 \
  --eval-steps 10 \
  --save-steps 10 \
  --quality-eval-steps 10 \
  --quality-eval-max-rows 32 \
  --importance-sampling-level sequence \
  --loss-type dapo \
  > /root/grpo_qwen3p5_4b_intersectionqa_edit_canary.log 2>&1 &
```

If the canary has non-degenerate rewards and valid formatted completions, extend
to the 300-step pilot:

```bash
cd /root/IntersectionQA
nohup python scripts/text_grpo_train_unsloth.py \
  --dataset-dir /root/intersectionedit_grpo_pilot_inner_all \
  --model unsloth/Qwen3.5-4B \
  --output-dir /root/outputs/grpo_qwen3p5_4b_intersectionqa_edit_pilot \
  --train-splits inner_train \
  --eval-splits inner_eval \
  --max-steps 300 \
  --max-prompt-length 2048 \
  --max-completion-length 1024 \
  --num-generations 4 \
  --eval-steps 50 \
  --save-steps 50 \
  --quality-eval-steps 50 \
  --quality-eval-max-rows 64 \
  --importance-sampling-level sequence \
  --loss-type dapo \
  --resume \
  > /root/grpo_qwen3p5_4b_intersectionqa_edit_pilot.log 2>&1 &
```

Actual canary results:

- The first 512-token, 4-generation canary reached step 10, but the generation
  quality callback crashed because Qwen3.5's processor was used for text-only
  generation. The runner was patched to use the underlying text tokenizer for
  quality generation.
- The faster canary used `max_completion_length=192`, `num_generations=2`,
  128 train rows, and 64 eval rows. It saved
  `/root/outputs/grpo_qwen3p5_4b_intersectionqa_edit_canary_fast/checkpoint-10`.
- Step 5 train reward mean: `0.0500`; clipped completion ratio: `0.75`.
- Step 10 train reward mean: `0.08925`; reward std: `0.1262`; clipped
  completion ratio: `0.60`; grad norm: `1.0286`.
- Step 10 internal eval reward mean: `0.08164`; eval clipped completion ratio:
  `0.7344`; eval runtime: `648.7s` for 64 rows.
- A metrics bug was found after the step-10 quality log: reward parsing accepted
  `<answer>...</answer>`, but `evaluate_predictions` parsed raw completion text.
  Shared answer-tag canonicalization was added so quality metrics evaluate the
  same answer candidate as the reward function.
- A corrected step-11 resume quality probe over 8 rows reported reward mean
  `0.2125`. Tolerance-fit rows were `2/2` correct with zero invalid outputs;
  clearance, relation, and repair-translation rows were still invalid. This is
  enough to preserve the canary artifact, but not enough to justify the 300-step
  pilot.
- A second performance issue was fixed in the runner: `GRPOTrainer` now receives
  the text tokenizer instead of the Qwen processor, avoiding the processor path
  and associated warnings in future runs.
- Local artifact mirror:
  `data/training_artifacts/grpo_qwen3p5_4b_intersectionqa_edit_canary_fast/`
  contains train/quality JSONL logs, remote command logs, and the step-10 LoRA
  adapter checkpoint files. HF bucket upload was skipped because no bucket
  target was specified for this canary.

Stop decision: do not launch the 300-step pilot from this checkpoint. The next
run should first improve output-format reliability, reduce quality-eval cost,
and then rerun a small canary with the text-tokenizer trainer path.

Format-reward follow-up:

- Added small format-scaffold reward for `<answer>...</answer>` and
  `<think>...</think><answer>...</answer>` outputs, tightened the system prompt,
  reduced quality-eval defaults, and logged representative quality samples.
- On a fresh A100 contract `35601616`, the clean Torch `2.5.1` stack loaded but
  produced zero trainable LoRA parameters. Upgrading Unsloth installed Torch
  `2.10.0+cu128`, transformers `5.5.0`, TRL `0.23.0`, and Unsloth
  `2026.4.8`; this stack trained with `38,756,352` LoRA parameters.
- Qwen3.5/TRL needed a compatibility shim that attaches `warnings_issued` to
  the Unsloth-wrapped model before `GRPOTrainer` initialization.
- The 10-step format canary used `max_completion_length=128`,
  `num_generations=2`, 128 train rows, and 32 eval rows. Step 10 quality over 8
  rows had reward mean `0.3491` and invalid-output rate `0.0`. Correct rows were
  still only tolerance-fit (`2/2`) and one relation row (`1/2`); repair and
  clearance bucket remained wrong.
- A bounded 50-step continuation resumed from checkpoint 10 in the same output
  directory. Step 25 train reward peaked at `0.4942`, but the 16-row quality
  probe was only `0.3319`. Step 50 quality fell to `0.3116`: tolerance-fit
  remained `3/3`, relation was `1/2`, and binary interference, centroid-distance
  move, clearance bucket, repair translation, target-clearance move, and volume
  bucket were all `0%` exact. Invalid-output rate stayed `0.0`, so the remaining
  issue is task competence, not parsing.
- Final internal eval reward at step 50 was `0.3066` with clipped completion
  ratio `0.0625`, while KL rose to `0.6432`. Do not extend this checkpoint to a
  300-step run without changing the data mix, prompt, or reward shaping for the
  underperforming task families.
- Local artifact mirror:
  `data/training_artifacts/grpo_qwen3p5_4b_intersectionqa_edit_format_pilot50/`
  contains logs, `train_metrics.jsonl`, `quality_metrics.jsonl`, `checkpoint-50`,
  the final adapter, and the compressed remote artifact bundle.
- Vast instance `35601616` was destroyed after artifact retrieval; `show
  instances --raw` returned `[]`.

Next canary preparation:

- The failed 50-step continuation exposed a data-starvation issue in capped
  canaries: the previous random 128-row cap contained only 7
  `repair_translation`, 5 `target_clearance_move`, and 3 `repair_direction`
  rows. The runner now defaults to task-stratified row sampling, which gives
  roughly 12-13 rows per task family under the same 128-row cap.
- Quality eval rows are selected from the stratified eval cap, and logged samples
  now prioritize low-reward task diversity rather than the first rows. This
  should make the next canary's failure log useful for repair/movement/bucket
  debugging without increasing generation cost.
- The next GPU check should be another short canary with the same cheap shape as
  the format run, but with default stratified sampling:

```bash
python scripts/text_grpo_train_unsloth.py \
  --dataset-dir /root/intersectionedit_grpo_pilot_inner_all \
  --model unsloth/Qwen3.5-4B \
  --output-dir /root/outputs/grpo_qwen3p5_4b_intersectionqa_edit_stratified_canary \
  --train-splits inner_train \
  --eval-splits inner_eval \
  --max-train-rows 128 \
  --max-eval-rows 32 \
  --max-steps 20 \
  --max-prompt-length 2048 \
  --max-completion-length 128 \
  --num-generations 2 \
  --eval-steps 20 \
  --save-steps 20 \
  --quality-eval-steps 20 \
  --quality-eval-max-rows 16 \
  --quality-max-new-tokens 96 \
  --quality-sample-count 16 \
  --importance-sampling-level sequence \
  --loss-type dapo \
  --temperature 0.7
```

### Earlier April 25, 2026 SFT Run

- Date: April 25, 2026
- Dataset: `data/IntersectionQA-90K`
- Model: `unsloth/Qwen3.5-4B`
- Context length: `2048`
- Method: bf16/16-bit LoRA with full language, attention, and MLP modules
- Loss: answer-only loss via `--assistant-only-loss`
- Packing: manual token packing via `--pack-tokenized`
- Training target: one epoch over the current train split
- Output directory:
  `/root/outputs/sft_unsloth_qwen3p5_4b_intersectionqa_90k_2048_tpack_answer_b32`

Manual token packing is used because TRL/Unsloth built-in packing is skipped
for Qwen3.5, which is processor-based. Optional stepped quality probes via
`--quality-eval-steps` run generation on a small fixed eval sample and log exact
accuracy plus precision/recall/F1 to `quality_metrics.jsonl`.

## Hardware

Use an A100 80GB-class instance for this time-budgeted SFT experiment. The cheap
16GB RTX instance was useful only for smoke tests and was destroyed.

Confirmed working A100 contract during the April 25, 2026 run:

- Contract: `35563228`
- GPU: `NVIDIA A100-SXM4-80GB`
- SSH: `ssh7.vast.ai`, port `13228`
- Approx total price: about `$1.08/hr` with 120GB disk

Always destroy old or failed instances after moving to a new one:

```bash
printf 'y\n' | rtk uv run vastai destroy instance <contract_id>
```

## Model Decision

`unsloth/Qwen3.5-4B` was selected because it keeps the experiment focused on
proper full-module supervised fine-tuning instead of a mostly capacity-bound
canary while still fitting the four-hour budget better than the larger
candidates.

Rejected or deprioritized paths:

- The 25-35B path was the wrong tradeoff for this experiment. It fit only with
  attention-only LoRA or had step times too slow for a useful 4-hour run.
- The 9B model is viable but too slow for the current time budget when training
  full language, attention, and MLP LoRA modules over real rows.
- `unsloth/Qwen3.5-4B` full-module LoRA at 4096 context fit comfortably in VRAM
  but settled around 26-30 seconds per optimizer step, too slow for the
  four-hour budget.
- `unsloth/Qwen3.5-9B` full-module LoRA at 4096 context fit in VRAM but early
  optimizer steps were around 30 seconds. At 2048 context with batch 4 and
  grad-accum 4 it still trended around 24-25 seconds per optimizer step.
- `Qwen/Qwen3.6-27B` dense 4-bit QLoRA fits on A100 80GB at 4096 context, but
  the canary showed early optimizer steps taking tens of seconds. A full epoch
  over 22,505 train rows would not fit the 4-hour budget.
- `Qwen/Qwen3.6-35B-A3B` MoE with generic PEFT QLoRA OOMed during
  `prepare_model_for_kbit_training` on A100 80GB.
- `unsloth/Qwen3.6-35B-A3B` bf16 LoRA at 2048 context loaded successfully and
  used about 67.5GB VRAM before training, but full attention+MLP LoRA trained
  about 953M parameters and was too slow.

## Remote Setup

Upload files:

```bash
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.ssh/vastai_wireseghr -P 13228 \
  data/IntersectionQA-90K.tar.gz \
  scripts/text_sft_train.py \
  scripts/text_sft_train_unsloth.py \
  scripts/evaluate_text_model.py \
  scripts/text_sft_smoke.py \
  scripts/text_grpo_smoke.py \
  root@ssh7.vast.ai:/root/
```

Prepare the A100 box:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.ssh/vastai_wireseghr -p 13228 root@ssh7.vast.ai

cd /root
tar -xzf IntersectionQA-90K.tar.gz
python -m pip install --upgrade transformers datasets "trl>=0.22.0" peft bitsandbytes accelerate huggingface_hub
python -m pip install --upgrade unsloth
```

Unsloth may replace Torch with its pinned CUDA build. After installation,
confirm CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

## Canary Run

For the 4B path, launch the full one-epoch run directly and monitor the first
10-20 optimizer steps. The 4096-context run was too slow, so use 2048 context.
If batch 32 OOMs, reduce `--per-device-train-batch-size` to `16`.
Always keep `--assistant-only-loss` enabled. The dataset rows contain long
prompts and tiny answers, so prompt-token loss wastes most of the update budget.

The older 35B-A3B attention-only canary command was:

```bash
cd /root
nohup python text_sft_train_unsloth.py \
  --dataset-dir /root/IntersectionQA-90K \
  --model unsloth/Qwen3.6-35B-A3B \
  --output-dir /root/outputs/sft_unsloth_qwen3p6_35b_a3b_90k_canary \
  --max-train-rows 512 \
  --max-eval-rows 64 \
  --max-steps 10 \
  --max-seq-length 2048 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --no-finetune-mlp-modules \
  --logging-steps 1 \
  --eval-steps 10 \
  --save-steps 10 \
  > /root/sft_unsloth_qwen3p6_35b_a3b_canary.log 2>&1 &
```

Monitor:

```bash
tail -n 160 /root/sft_unsloth_qwen3p6_35b_a3b_canary.log
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
df -h /root
```

## Full Run

Current 4B full-module run:

```bash
cd /root
nohup python text_sft_train_unsloth.py \
  --dataset-dir /root/IntersectionQA-90K \
  --model unsloth/Qwen3.5-4B \
  --output-dir /root/outputs/sft_unsloth_qwen3p5_4b_intersectionqa_90k_2048_tpack_answer_b32 \
  --max-eval-rows 128 \
  --num-train-epochs 1 \
  --max-steps -1 \
  --max-seq-length 2048 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 2e-4 \
  --warmup-ratio 0.03 \
  --logging-steps 10 \
  --eval-strategy no \
  --no-final-eval \
  --save-steps 100 \
  --save-total-limit 3 \
  --pack-tokenized \
  --assistant-only-loss \
  --metrics-log-file train_metrics.jsonl \
  --quality-eval-steps 100 \
  --quality-eval-max-rows 64 \
  --quality-metrics-log-file quality_metrics.jsonl \
  --resume \
  > /root/sft_unsloth_qwen3p5_4b_90k_2048_tpack_answer_b32.log 2>&1 &
```

This command disables in-training eval and final eval to keep the GPU focused on
SFT during the time budget. Run `scripts/evaluate_text_model.py` afterward on a
small held-out slice if time remains.

Future runs write durable JSONL metrics to
`<output-dir>/train_metrics.jsonl` and also print `[metrics] ...` lines to
stdout. The active April 25 run was paused and resumed from `checkpoint-100`,
so its metrics file starts at step 110 rather than step 0.

When `--quality-eval-steps` is nonzero, future runs also write
`<output-dir>/quality_metrics.jsonl` and print `[quality] ...` lines. Keep the
sample small, for example 64 rows every 100 steps, because these are generation
metrics and cost real GPU time.

## Known Failure Modes

- 120GB disk is too small to keep both `Qwen/Qwen3.6-27B` and
  `Qwen/Qwen3.6-35B-A3B` caches at once. Clear failed model caches before
  switching:

```bash
rm -rf /root/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B
rm -rf /root/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B
rm -rf /root/.cache/huggingface/hub/.locks/models--Qwen--Qwen3.6-27B
rm -rf /root/.cache/huggingface/hub/.locks/models--Qwen--Qwen3.6-35B-A3B
```

- Generic PEFT QLoRA on `Qwen/Qwen3.6-35B-A3B` OOMed on A100 80GB during
  k-bit preparation. Use Unsloth bf16 LoRA for MoE.
- Unsloth bf16 LoRA on `unsloth/Qwen3.6-35B-A3B` with both attention and MLP
  modules enabled fit at 2048 context but trained about 953M parameters and was
  too slow. Attention-only was a canary, not the preferred experiment.
- Dense 27B QLoRA is viable but too slow for a 4-hour full-dataset run at 4096
  context.
- `unsloth/Qwen3.5-9B` full-module LoRA fit but remained too slow at both 4096
  and 2048 context for the current budget.
- `unsloth/Qwen3.5-4B` full-module LoRA at 4096 context also remained too slow;
  use the 2048/batch-32 run unless later logs prove it is still outside budget.
- Plain single-field SFT without `--assistant-only-loss` likely trains mostly on
  prompt reconstruction. Use the packed tokenized path, which masks labels
  before the assistant answer.
- Keep secrets out of logs and docs. The HF token was provided in chat but must
  not be written into repository files.
