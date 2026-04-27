# Overnight Task: Apr 27 to Apr 28, 2026

Owner for implementation: next agent. This file is the handoff, not the full
protocol source. Use the repo protocols below and update this file with an
execution log as work proceeds.

## Protocol Sources

- `.agents/skills/run-experiments/SKILL.md`: required workflow for this repo.
- `.agents/skills/run-experiments/references/experiment-workflow.md`: Vast,
  launch, monitoring, stop rules, artifact upload, and experiment records.
- `specs/research-experiment-spec.md`: paper-facing experiment matrix, budgets,
  reporting rules, and split hygiene.
- `configs/overnight_experiment_suite.yaml`: restartable suite manifest.
- `docs/experiments/apr-27-qwen3p5-4b-intersectionqa-answer-sft.md`: latest
  answer-SFT recovery result.
- `docs/experiments/apr-27-openrouter-native-reasoning-distill.md`: latest
  OpenRouter native-reasoning distill result.
- `docs/qwen3p5-4b-tuning.md`: reusable Qwen3.5 4B runtime and training notes.
- Qwen3.5 model card, checked Apr 27, 2026:
  <https://huggingface.co/Qwen/Qwen3.5-4B>.
- Transformers Qwen3.5 docs, checked Apr 27, 2026:
  <https://huggingface.co/docs/transformers/main/model_doc/qwen3_5>.

## Current Known State

- Dataset: `MRiabov/IntersectionQA-90K`, local mirror
  `data/IntersectionQA-90K`.
- Stopped remote instance: Vast `35682721`, `NVIDIA A100 80GB PCIe`, already
  used for the Apr 27 SFT recovery. Prefer resuming that instance if still
  available and cost is acceptable.
- Answer-SFT baseline: `answer_sft_full` was stopped after step 400 because a
  fixed 64-row validation probe peaked at step 200 and regressed at steps 300
  and 400. Best checkpoint is `runs/answer_sft_full/checkpoint-best`, copied
  from `checkpoint-200`, and uploaded under:
  `hf://buckets/MRIabov/intersectionqa-qwen3p5-4b-grpo-artifacts/intersectionqa-overnight-2026-04-27/vast-35682721/runs/answer_sft_full`.
- Native-reasoning distill artifacts available locally:
  - `runs/openrouter_native_reasoning_distill_1000`: 1000 MVP rows, 659
    accepted correct traces. Task counts: `binary_interference=243`,
    `relation_classification=219`, `volume_bucket=197`.
  - `runs/openrouter_native_reasoning_distill_unexposed_1332`: 1332 rows, 600
    accepted correct traces. Task counts: `clearance_bucket=175`,
    `pairwise_interference=71`, `ranking_normalized_intersection=63`,
    `tolerance_fit=291`.
  - Combined available accepted traces: 1259 rows across seven task types.
- The accepted OpenRouter traces are long: mean accepted trace length is about
  18.8k to 19.1k characters. They are useful raw material, but they are not yet
  the 128-256 token reasoning targets requested for the main short-trace SFT
  condition.

## Research Question

Test whether reasoning length itself is useful once the model has already been
SFT'd on reasoning traces.

Compare two conditions:

1. **Short-trace reasoning SFT**: shorten accepted native-reasoning traces to
   128-256 useful reasoning tokens, preserve the canonical final answer, run a
   canary, then train/evaluate if healthy.
2. **Long-trace reasoning SFT**: train directly on the existing long native
   reasoning traces. This condition is not documented elsewhere yet; document
   it as an exploratory experiment, including throughput, memory pressure,
   truncation, and whether quality improves over the short-trace condition.

Then, if supervised canaries are healthy and there is still time/budget, run
SFT+GRPO from the better reasoning-SFT adapter. Do not start a main GRPO run
until the canary has non-degenerate reward, controlled invalid-output rate,
sane completion lengths, and held-out quality that is not obviously regressing.

## Qwen3.5 Formatting Decision

Use Qwen-native chat formatting. Do not hand-format training examples as raw
`<think>...</think><answer>...</answer>` strings without applying the model chat
template.

Current guidance from the official Qwen3.5 model card:

- Qwen3.5 thinks by default and emits a thinking block before the final answer.
- To disable thinking through OpenAI-compatible serving, pass
  `chat_template_kwargs: {"enable_thinking": false}` in the extra body.
- For this experiment, keep thinking enabled for reasoning SFT/GRPO unless a
  specific diagnostic intentionally disables it.
- Use message order `user` then `assistant` for SFT examples, and let
  `tokenizer.apply_chat_template(..., add_generation_prompt=True/False)` render
  the model-specific sequence. The current Unsloth SFT runner already follows
  this pattern for rows with `target_text`.

Implementation note: Qwen3.5 is processor-backed. For text-only training and
evaluation, use the underlying text tokenizer path already described in
`docs/qwen3p5-4b-tuning.md`.

## Work Plan

### 0. Local Preflight

Before touching GPU:

```bash
rtk uv run python -m compileall -q intersectionqa scripts
rtk uv run pytest -q tests/test_reasoning_traces.py tests/test_text_sft_train_unsloth.py tests/test_rewards.py tests/test_experiments.py
rtk uv run python -m scripts.experiments.run_experiment_suite configs/overnight_experiment_suite.yaml --list
```

If implementation changes the manifest or training-data scripts, add focused
tests for those changes before renting or resuming GPU.

### 1. Build The Reasoning SFT Inputs

Create two explicit datasets from the accepted distill artifacts. The long-trace
condition already exists as accepted `target_text` rows; it only needs to be
merged/materialized into a normal dataset directory with a report, not
regenerated.

Short-trace target:

- Input files:
  - `runs/openrouter_native_reasoning_distill_1000/accepted_reasoning_sft.jsonl`
  - `runs/openrouter_native_reasoning_distill_unexposed_1332/accepted_reasoning_sft.jsonl`
- Output suggestion: `data/reasoning_sft_short_native_distill/`.
- Each output row must keep the public row fields, `canonical_answer`, and
  `target_text`.
- Target format:
  `<think>128-256 tokens of concise geometry reasoning</think><answer>ANSWER</answer>`.
- The summary should explicitly reason about the closest relevant surviving
  shape or feature in the CadQuery design when that is inferable from the
  prompt. The useful pattern is: identify candidate parts, identify the closest
  relevant object/feature after transforms, then derive the relation or bucket.
- Preserve an audit field in `supervision`, for example:
  `target_text_source=openrouter_native_reasoning_shortened`,
  `source_trace_run`, `source_trace_row_id`, `shortener_model`, and
  `shortening_policy`.

Long-trace target:

- Output suggestion: `data/reasoning_sft_long_native_distill/`.
- Materialize this by merging the existing accepted native-reasoning rows into
  a training dataset, for example `train.jsonl`, and writing a report with row
  counts, task distribution, trace-length statistics, and truncation risk.
- Do not regenerate long traces.
- Copy accepted rows as-is unless a minimal packaging change is required by the
  training runner. Make truncation explicit in the training report. Do not
  silently rely on context truncation.
- If using a context length below the full prompt plus trace length, this is a
  canary-only condition until truncation is measured and reported.

Manual proof audit:

- Before any full short-trace SFT, manually inspect at least 20 shortened
  proofs, with examples from every task type that has accepted rows.
- Mark each proof `valid`, `fallacious`, `too vague`, `answer_mismatch`, or
  `leaks_label_without_reasoning`.
- Stop or revise the shortener if more than 20% of inspected proofs are
  fallacious or answer-mismatched.
- Save audit notes under the run directory and summarize them in
  `docs/experiments/apr-27-openrouter-native-reasoning-distill.md` or a new
  dated experiment record.

### 2. Canary Both SFT Conditions

Run short, symmetric canaries before committing to a longer training run.

Short-trace SFT canary:

```bash
python -m scripts.training.text_sft_train_unsloth \
  --dataset-dir data/reasoning_sft_short_native_distill \
  --output-dir runs/reasoning_sft_short_canary \
  --pack-tokenized \
  --max-train-rows 256 \
  --max-eval-rows 64 \
  --max-steps 20 \
  --per-device-train-batch-size 8 \
  --gradient-accumulation-steps 1 \
  --eval-strategy no \
  --no-final-eval \
  --final-adapter-save-mode checkpoint \
  --resume \
  --quality-eval-steps 10
```

Long-trace SFT canary:

```bash
python -m scripts.training.text_sft_train_unsloth \
  --dataset-dir data/reasoning_sft_long_native_distill \
  --output-dir runs/reasoning_sft_long_canary \
  --pack-tokenized \
  --max-train-rows 128 \
  --max-eval-rows 64 \
  --max-steps 20 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --eval-strategy no \
  --no-final-eval \
  --final-adapter-save-mode checkpoint \
  --resume \
  --quality-eval-steps 10
```

The long-trace command is intentionally conservative. Increase batch/context
only after verifying VRAM, step time, and truncation.

Canary decision rule:

- Prefer the condition with better held-out exact accuracy, lower invalid rate,
  lower truncation, and acceptable GPU utilization.
- If short-trace quality is close to long-trace quality, prefer short traces for
  the overnight run because they are cheaper and closer to the research spec.
- If long-trace SFT is much slower or truncates heavily, record it as a negative
  length diagnostic and stop that branch.

### 3. Full Reasoning SFT Gate

After the canaries, run one medium/full reasoning-SFT condition from the better
branch. Suggested priority:

1. Short-trace full or medium run.
2. Long-trace medium run only if the canary is clearly useful and not dominated
   by truncation or throughput issues.

Evaluate the resulting adapter on at least:

- `validation`
- `test_random`
- `test_object_pair_heldout`
- `test_near_boundary`
- `test_generator_heldout` when populated

Use exact-answer accuracy, invalid-output rate, task-type breakdown, and
completion length/truncation metrics. Compare against
`answer_sft_full/checkpoint-best`, with the caveat that the answer-SFT run used
a short-answer objective and was early-stopped.

### 4. GRPO Gate

Only run GRPO after a reasoning-SFT adapter is usable.

Starting command shape:

```bash
python -m scripts.training.text_grpo_train_unsloth \
  --dataset-dir data/reasoning_sft_short_native_distill \
  --adapter-init-dir runs/reasoning_sft_short_full/adapter \
  --output-dir runs/grpo_short_reasoning_canary \
  --max-train-rows 256 \
  --max-eval-rows 64 \
  --max-steps 50 \
  --eval-strategy no
```

Adjust paths if the long-trace adapter wins. Keep the first GRPO run to 50-100
steps. Do not continue to 1500-2000 steps unless:

- reward mean/std is non-degenerate;
- invalid-output rate is controlled;
- exact held-out quality is flat or improving;
- no common-label collapse is visible;
- GPU utilization and memory utilization are reasonable for the price.

## Remote Execution Notes

If using the existing Vast instance:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i <ssh_key> -p <ssh_port> root@<ssh_host>
cd /root/IntersectionQA
git status --short
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

For a fresh or stale instance, bootstrap with the repo script from the workflow
reference. On a bootstrapped Vast image, use image Python directly, not `rtk`.

Run long jobs in `tmux`:

```bash
tmux new -d -s iqa 'cd /root/IntersectionQA && python -m scripts.training.text_sft_train_unsloth ... > overnight.log 2>&1'
```

Monitor:

```bash
tail -n 160 overnight.log
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
df -h /root
python -m scripts.experiments.monitor_experiment_run <run_dir>
```

## Stop Rules

Hard stop:

- local preflight fails and the failure is not understood;
- dataset validation or split/leakage assumptions are unclear;
- training loss becomes NaN;
- repeated OOM after reducing batch/context settings;
- disk is close to full;
- expected artifacts cannot be written;
- live Vast price is unreasonable for the remaining task;
- job approaches the morning handoff without artifact preservation.

Soft stop or revise:

- shortened proofs fail manual audit;
- invalid-output rate remains high;
- reward improves while exact held-out quality falls;
- training/eval quality saturates and does not materially progress for 300
  optimizer iterations. Stop the run at the next safe checkpoint, move to the
  next planned phase if useful, or turn off/destroy the GPU instance and cease
  execution if all planned tasks are complete;
- completions collapse to a common answer such as `no`, `0`, or `disjoint`;
- reasoning is filler or repetitive;
- long traces are mostly truncated or too slow to compare fairly.

## Artifact Contract

Preserve for every canary or full run:

- exact command and resolved config;
- `git status --short` and dirty diff note;
- training/eval metrics JSONL;
- predictions and sampled completions;
- prompt/completion length and truncation stats;
- adapter/checkpoints, plus `checkpoint-best` or `adapter-best` when applicable;
- manual proof audit notes for shortened traces;
- upload target and checksum or local fallback tarball.

Preferred upload prefix:

```text
hf://buckets/MRIabov/intersectionqa-qwen3p5-4b-grpo-artifacts/intersectionqa-overnight-2026-04-27/vast-35682721
```

Before destroying any rented instance, verify that artifacts are uploaded or
mirrored locally.

## Morning Handoff Target

I will come back around 05:45 Dublin time. By then, leave:

- this file updated with the execution log below;
- a dated experiment record under `docs/experiments/`;
- current tmux/session status if any job is still running;
- exact next command to resume or continue;
- artifact paths and upload status;
- a short researcher summary: which condition looked better, what failed, and
  whether GRPO should proceed.

## Execution Log

Use this section as the live scratchpad during the night.

### To Do

- [ ] Run local preflight.
- [ ] Build short-trace reasoning SFT dataset.
- [ ] Materialize long-trace reasoning SFT dataset from existing accepted rows.
- [ ] Manually audit shortened proofs.
- [ ] Run short-trace SFT canary.
- [ ] Run long-trace SFT canary.
- [ ] Decide which SFT branch to continue.
- [ ] Evaluate the selected reasoning-SFT adapter.
- [ ] Run GRPO canary only if SFT is healthy.
- [ ] Upload or mirror artifacts.
- [ ] Update dated experiment record.

### Done

- [x] Existing answer-SFT best checkpoint identified from Apr 27 record.
- [x] Existing OpenRouter native-reasoning distill artifacts identified.
- [x] Qwen3.5 official input/template guidance checked on Apr 27, 2026.

### Researcher Summary

Fill this in before morning handoff.
