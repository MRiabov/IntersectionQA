# IntersectionQA Text Fine-Tuning Runbook

This note captures the current text-only dataset and training state so future
agents can resume without rediscovering the same failure modes.

## Dataset State

- Local 15K export: `data/IntersectionQA-15K`
- Local 90K export: `data/IntersectionQA-90K`
- 90K archive for remote transfer: `data/IntersectionQA-90K.tar.gz`
- Hugging Face datasets:
  - `MRiabov/IntersectionQA-15K`
  - `MRiabov/IntersectionQA-90K`

The 90K export has 90,000 final public rows. The train split was augmented on
April 25, 2026 to include a leakage-safe subset of counterfactual groups from
`test_near_boundary`. The active A100 baseline run that started earlier that
morning is still training on the pre-augmentation split, because its dataloader
was already constructed before this local dataset rewrite.

Current local `data/IntersectionQA-90K` split counts:

- `train`: 44,149 rows, including 21,644 rows with
  `counterfactual_group_id`
- `validation`: 2,895 rows
- `test_random`: 2,345 rows
- `test_object_pair_heldout`: 3,095 rows
- `test_near_boundary`: 37,516 rows, including 37,096 rows with
  `counterfactual_group_id`

Current train task counts:

- `binary_interference`
- `clearance_bucket`
- `pairwise_interference`
- `ranking_normalized_intersection`
- `relation_classification`
- `tolerance_fit`
- `volume_bucket`

Exact train counts are 8,496 `binary_interference`, 8,494
`clearance_bucket`, 839 `pairwise_interference`, 838
`ranking_normalized_intersection`, 8,494 `relation_classification`, 8,494
`tolerance_fit`, and 8,494 `volume_bucket`.

The augmentation moved 839 of 1,678 eligible counterfactual groups into train.
Eligibility excludes groups whose `base_object_pair_id` or `assembly_group_id`
appears in validation, random test, or object-pair heldout splits. The remaining
near-boundary counterfactual groups stay held out.

Interpret the splits as a conventional supervised split plus named challenge
suites:

- `train`: training data.
- `validation`: dev validation.
- `test_random`: primary in-distribution held-out test.
- `test_object_pair_heldout`: strict object-pair/assembly generalization
  challenge.
- `test_near_boundary`: boundary/counterfactual challenge suite. It is not the
  clean headline test split; it forbids counterfactual group leakage, but the
  current export can share base object-pair/assembly IDs with train.
- `test_generator_heldout`: reserved generator-family challenge split; currently
  empty in the 90K export.

See `docs/dataset_split_framing.md` before interpreting evaluation numbers.

## Local Scripts

- `scripts/text_sft_train.py`: generic Transformers/PEFT LoRA or QLoRA runner.
  Use this for dense 4-bit QLoRA models.
- `scripts/text_sft_train_unsloth.py`: Unsloth bf16 LoRA runner. Use this for
  the current Qwen3.5 text-only SFT path and for any future MoE bf16 LoRA
  experiments. Its default `--task-types` now includes all seven public text
  tasks, including `pairwise_interference` and
  `ranking_normalized_intersection`.
- `scripts/evaluate_text_model.py`: exact-answer generation evaluator for a
  base or adapter model. It reports exact-answer accuracy by split/task plus
  per-task label precision, recall, F1, and confusion counts.
- `scripts/text_sft_smoke.py`: tiny local/remote SFT smoke test.
- `scripts/text_grpo_smoke.py`: tiny RL/GRPO smoke test.
- `scripts/add_counterfactual_train_rows.py`: deterministic in-place
  augmentation helper for moving leakage-safe counterfactual groups from
  `test_near_boundary` to `train`. It creates
  `data/IntersectionQA-90K/pre_counterfactual_train_backup` before rewriting
  split files.

## Vast State And Hardware Choice

Use an A100 80GB-class instance for the time-budgeted Qwen3.5 SFT experiment.
The cheap 16GB RTX instance was useful only for smoke tests and was destroyed.

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

Current preferred model for the 4-hour experiment:

- `unsloth/Qwen3.5-4B`
- `max_seq_length=2048`
- bf16/16-bit LoRA
- Full language, attention, and MLP LoRA modules
- Answer-only loss. Do not train the model to reproduce the long CadQuery
  prompt text.
- Manual token packing via `--pack-tokenized`. TRL/Unsloth built-in packing is
  skipped for Qwen3.5 because it is processor-based.
- Optional stepped quality probes via `--quality-eval-steps`. These run
  generation on a small fixed eval sample and log exact accuracy plus
  precision/recall/F1 to `quality_metrics.jsonl`.
- One epoch over the current train split

Why:

- The 25-35B path was the wrong tradeoff for this experiment. It fit only with
  attention-only LoRA or had step times too slow for a useful 4-hour run.
- The 9B model is viable but too slow for the current time budget when training
  full language, attention, and MLP LoRA modules over real rows.
- The 4B model keeps the experiment focused on proper full-module supervised
  fine-tuning instead of a mostly capacity-bound canary.
- `unsloth/Qwen3.5-4B` full-module LoRA at 4096 context fit comfortably in
  VRAM but settled around 26-30 seconds per optimizer step, too slow for the
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
The output directory used for the active answer-only packed run is
`/root/outputs/sft_unsloth_qwen3p5_4b_intersectionqa_90k_2048_tpack_answer_b32`.

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
