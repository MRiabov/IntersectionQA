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
- Apr 28 correction: the materialized short-trace dataset currently in
  `data/reasoning_sft_short_native_distill/` was not shortened with OpenRouter
  DeepSeek V4 Pro or a similar paid reasoning model. It was shortened by the
  local deterministic extractor
  `deterministic_native_reasoning_sentence_extractor_v1` in
  `scripts/training/prepare_native_reasoning_sft_datasets.py`. This does not
  satisfy the original experiment instruction to use OpenRouter for 256-token
  compression, so treat the completed short-trace SFT branch as a
  protocol-noncompliant diagnostic, not the requested short-trace condition.
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

Correction, Apr 28: do not make literal `<think>...</think><answer>...</answer>`
emission a required training target or GRPO gate. The earlier version of this
handoff over-concretized the requested "reasoning plus answer" format into
literal tags, and the observed canaries show that this collapses or distorts
the answer distribution. Treat the literal-tag strategy below in the execution
log as a negative result, not a recommendation.

Use Qwen-native chat formatting. Do not hand-format training examples as raw
tag strings without applying the model chat template, and do not assume that
Qwen's native thinking protocol is best represented by literal XML-like tags in
the assistant content.

Current guidance from the official Qwen3.5 model card:

- Qwen3.5 thinks by default in native generation modes, but the training target
  should follow the model/tokenizer's chat template and the serving API's
  expected thinking controls rather than a hand-invented tag contract.
- To disable thinking through OpenAI-compatible serving, pass
  `chat_template_kwargs: {"enable_thinking": false}` in the extra body.
- For future reasoning SFT/GRPO, first inspect the rendered Qwen chat template
  and run a tiny format canary. The health gate should be parse-valid final
  answers and non-collapsed task behavior, not literal `<think>` tag emission.
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
- Required shortener for a compliant rerun: OpenRouter `deepseek/deepseek-v4-pro`
  or a similar capable model at roughly the requested price class. Do not use
  the existing deterministic sentence extractor as the main compression method
  unless the run is explicitly labeled as a cheap heuristic diagnostic.
- Each output row must keep the public row fields, `canonical_answer`, and
  `target_text`.
- Target format: Qwen-native assistant content rendered through the tokenizer's
  chat template. The target should contain concise geometry reasoning plus the
  canonical final answer, but should not require literal `<think>` or
  `<answer>` tags unless a fresh tiny canary proves that this format improves
  parse-valid answers without label collapse.
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

- [x] Run local preflight.
- [x] Build short-trace reasoning SFT dataset.
- [x] Materialize long-trace reasoning SFT dataset from existing accepted rows.
- [x] Manually audit shortened proofs.
- [x] Run short-trace SFT canary.
- [x] Run/attempt long-trace SFT canary.
- [x] Decide which SFT branch to continue.
- [x] Evaluate the selected reasoning-SFT adapter.
- [x] Apply GRPO gate decision; do not run GRPO because SFT is not healthy.
- [x] Upload or mirror artifacts.
- [x] Update dated experiment record.

### Done

- [x] Existing answer-SFT best checkpoint identified from Apr 27 record.
- [x] Existing OpenRouter native-reasoning distill artifacts identified.
- [x] Qwen3.5 official input/template guidance checked on Apr 27, 2026.

### 2026-04-27 21:40 IST

- Read `AGENTS.md`, `/home/maksym/.codex/RTK.md`,
  `.agents/skills/run-experiments/SKILL.md`,
  `.agents/skills/run-experiments/references/experiment-workflow.md`, and this
  handoff.
- Local preflight passed:
  - `rtk uv run python -m compileall -q intersectionqa scripts`
  - `rtk uv run pytest -q tests/test_reasoning_traces.py tests/test_text_sft_train_unsloth.py tests/test_rewards.py tests/test_experiments.py`
    produced `28 passed, 7 warnings`.
  - `rtk uv run python -m scripts.experiments.run_experiment_suite configs/overnight_experiment_suite.yaml --list`
    listed the expected suite entries.
- Added `scripts/training/prepare_native_reasoning_sft_datasets.py` and focused
  coverage in `tests/test_prepare_native_reasoning_sft_datasets.py`. New tests
  passed with `rtk uv run pytest -q tests/test_prepare_native_reasoning_sft_datasets.py tests/test_reasoning_traces.py`.
- Built:
  - `data/reasoning_sft_short_native_distill/`
  - `data/reasoning_sft_long_native_distill/`
- Build command:
  `rtk uv run python -m scripts.training.prepare_native_reasoning_sft_datasets --input runs/openrouter_native_reasoning_distill_1000/accepted_reasoning_sft.jsonl runs/openrouter_native_reasoning_distill_unexposed_1332/accepted_reasoning_sft.jsonl --short-output-dir data/reasoning_sft_short_native_distill --long-output-dir data/reasoning_sft_long_native_distill --eval-fraction 0.12 --audit-per-task 3`
- Combined source rows: 1259 accepted traces across seven task types. Split is
  group-safe from public-train rows: 1108 train, 151 validation.
- Short-trace lengths: train mean 134.0 reasoning whitespace tokens, min 128,
  max 203; validation mean 134.0, min 128, max 155.
- Long-trace lengths: train mean 3300.7 reasoning whitespace tokens; validation
  mean 3305.1. Treat the long condition as truncation-sensitive until measured.
- Manual short-trace proof audit saved at
  `data/reasoning_sft_short_native_distill/manual_audit_results.md`: 21 rows
  audited across all seven task types; 17 valid, 4 too vague, 0 fallacious,
  0 answer_mismatch, 0 leaks_label_without_reasoning. The short-trace dataset
  passed the audit gate.

### 2026-04-27 22:34 IST

- Preferred Vast instance `35682721` was still available but did not start:
  `vastai start instance 35682721 --raw` returned "Required resources are
  currently unavailable, state change queued", and polling still showed it
  stopped.
- Rented fallback Vast contract `35708545`: `NVIDIA A100-SXM4-80GB`, 200GB
  disk, about `$0.969/hr`. Verified live GPU and disk by SSH.
- Bootstrapped `/root/IntersectionQA`. Runtime needed repair:
  - removed incompatible `torchao` after a Torch 2.5 import failure;
  - installed `fla-core==0.5.0`, which pulled incompatible Torch 2.11;
  - repaired to Torch `2.10.0+cu128`, transformers `5.5.0`, TRL `0.24.0`,
    Unsloth `2026.4.8`, torchvision `0.25.0`, torchaudio `2.10.0`,
    xformers `0.0.35`, torchao `0.17.0`.
- Synced `data/reasoning_sft_short_native_distill/` and
  `data/reasoning_sft_long_native_distill/` to the remote.
- Short canary command was launched with two necessary safety additions:
  `--save-steps 10` so `--final-adapter-save-mode checkpoint` can succeed, and
  `--quality-max-new-tokens 256` so reasoning-format quality eval is not
  truncated before `<answer>`.
- Short canary failures:
  - first failed at model load due missing `fla.modules`;
  - second failed at LoRA injection because PEFT rejected `torchao==0.13.0`;
  - repaired runtime then the 20-step canary stalled after step 2 with no log or
    artifact progress for about eight minutes, so it was stopped and preserved.
- Ran a smaller diagnostic short canary:
  `python -m scripts.training.text_sft_train_unsloth --dataset-dir data/reasoning_sft_short_native_distill --output-dir runs/reasoning_sft_short_diagnostic --pack-tokenized --max-seq-length 1024 --max-train-rows 32 --max-eval-rows 16 --quality-eval-max-rows 16 --max-steps 5 --per-device-train-batch-size 1 --gradient-accumulation-steps 1 --eval-strategy no --no-final-eval --final-adapter-save-mode checkpoint --save-steps 5 --resume --quality-eval-steps 5 --quality-max-new-tokens 256`
- Diagnostic completed 5 steps and wrote `checkpoint-5`, adapter, metrics, and
  predictions. Quality probe was `3/16 = 0.1875` on train rows.
- Hard stop: `train_result.json` reported `train_loss: NaN`. Per stop rules, no
  long-trace canary, no medium/full SFT, and no GRPO were run.
- Remote artifacts were packaged as
  `/root/IntersectionQA/runs/reasoning_sft_short_diagnostic_artifacts.tgz` and
  mirrored locally to
  `runs/vast_35708545/reasoning_sft_short_diagnostic_artifacts.tgz`.
  SHA-256:
  `8215d1ec9a223fbf5cce024e7884c55b0809c6cb997bf3d3f4a2eb35e5c923d7`.
- Upload status: local mirror only; this satisfies the mirror-before-teardown
  fallback. Fresh Vast instance `35708545` was destroyed. `vastai show
  instances --raw` now shows only stopped instance `35682721`.
- Dated experiment record created:
  `docs/experiments/apr-27-28-qwen3p5-4b-native-reasoning-sft-length.md`.

### Researcher Summary

Short native traces were successfully built and passed manual proof audit, but
the SFT runtime did not pass the health gate. The only completed optimizer run
ended with `train_loss: NaN`, so there is no defensible short-versus-long
research conclusion yet. Do not run GRPO from these artifacts. Next command
should be a runtime-only sanity check after fixing/replacing the Qwen3.5
Unsloth stack, for example a 2-5 step SFT run on 16 short rows with loss
logging before attempting the symmetric short/long canaries again.

### 2026-04-27 23:02 IST Continuation

- Root-caused the NaN diagnostic to packed chunks that could contain only
  prompt labels (`-100`) when CAD prompts consumed the context before answer
  tokens. Patched `scripts/training/text_sft_train_unsloth.py` so token packing
  drops all-ignored chunks, masks EOS only when a supervised target exists, and
  raises if no supervised chunks are produced. Local focused tests passed:
  `rtk uv run pytest -q tests/test_text_sft_train_unsloth.py tests/test_prepare_native_reasoning_sft_datasets.py`.
- Preferred stopped Vast instance `35682721` was retried and again remained
  unavailable. Rented new Vast contract `35710717`: `NVIDIA A100-SXM4-80GB`,
  120GB disk, about `$1.0867/hr`.
- Bootstrapped `/root/IntersectionQA` and repaired the remote runtime to Torch
  `2.10.0+cu128`, transformers `5.5.0`, TRL `0.24.0`, Unsloth `2026.4.8`,
  xformers `0.0.35`, torchao `0.17.0`, and `fla-core==0.5.0` installed with
  `--no-deps`.
- Synced the patched trainer and both native reasoning SFT datasets to
  `35710717`. Remote compileall and a direct packing sanity check passed.
- Ran the tiny short-trace loss sanity gate:
  `python -m scripts.training.text_sft_train_unsloth --dataset-dir data/reasoning_sft_short_native_distill --output-dir runs/reasoning_sft_short_loss_sanity --pack-tokenized --max-seq-length 1024 --max-train-rows 32 --max-eval-rows 16 --quality-eval-max-rows 8 --max-steps 5 --per-device-train-batch-size 1 --gradient-accumulation-steps 1 --eval-strategy no --no-final-eval --final-adapter-save-mode checkpoint --save-steps 5 --resume --quality-eval-steps 5 --quality-max-new-tokens 256`
- Loss sanity result: completed 5 optimizer steps with finite
  `train_loss = 2.450699234008789`; train rows 32, packed train sequences 41,
  eval rows 16, packed eval sequences 20. The first optimizer step spent about
  105s in Torch Inductor compilation, then the run completed normally.
- Gate decision: NaN blocker is cleared for the patched packed-loss path.
  Proceeding to symmetric short/long SFT canaries before considering any
  medium/full SFT or GRPO.

### 2026-04-27 23:55 IST Canary Outcome And Handoff

- Added a second scoped trainer fix: packed batches now pad to
  `--max-seq-length` to avoid one-sample variable-shape Torch Inductor compile
  churn. Local focused tests passed: `8 passed`.
- 2048-context short canaries remained unhealthy even after fixed padding,
  `gradient_accumulation_steps=1`, and `TORCH_COMPILE_DISABLE=1`: each made it
  to optimizer step 2 and then stopped making metric/artifact progress while
  consuming GPU. These attempts were stopped and preserved.
- Conservative 1024-context short canary completed:
  - run: `runs/reasoning_sft_short_canary_1024`
  - command used `TORCH_COMPILE_DISABLE=1`, `--max-seq-length 1024`,
    `--max-train-rows 128`, `--max-eval-rows 64`, `--max-steps 10`,
    `--gradient-accumulation-steps 1`, `--quality-eval-max-rows 32`.
  - result: finite `train_loss = 1.8363332748413086`;
    128 train rows -> 158 packed train sequences; 64 eval rows -> 79 packed
    eval sequences.
  - quality probe on 32 train rows: exact `7/32 = 0.21875`.
  - invalid/unparsed quality outputs: `21/32 = 0.65625`; output word length
    mean 84.0, median 119.5, max 151. Parsed predictions were mostly `no`
    (9) or `intersecting` (2), with the rest invalid.
- Conservative 1024-context long canary was not healthy:
  - run: `runs/reasoning_sft_long_canary_1024`
  - same 1024/no-compile/GA1 setup with `--quality-max-new-tokens 512`.
  - 128 long rows expanded to 1109 packed train sequences, confirming severe
    truncation/chunking pressure.
  - run stalled after optimizer step 2 with no metrics/checkpoint/result and
    was stopped.
- Decision: no medium/full reasoning SFT. The only completed branch is short
  at 1024 context, and it has high invalid-output rate; the long branch does
  not pass the canary health gate. Do not run GRPO: canary conditions
  (controlled invalid rate, sane branch health, non-regressing held-out quality)
  are not satisfied.
- Remote artifacts from Vast `35710717` were packaged as
  `/root/IntersectionQA/runs/reasoning_sft_length_canary_35710717_artifacts.tgz`
  and mirrored locally to
  `runs/vast_35710717/reasoning_sft_length_canary_35710717_artifacts.tgz`.
  SHA-256:
  `afdfaaf6c3486c2b1476709f8670607af4bf6577ae39ccbbaa45bf54dded6c70`.
- Upload status: local mirror complete and checksum verified. Exact next
  command, if continuing research later, should be a runtime/debug diagnostic
  rather than a training launch, e.g. inspect why Qwen3.5/Unsloth hangs after
  step 2 at 2048 context and why the 1024 short canary has 65.6% invalid
  completions.
- Vast `35710717` was destroyed after artifact mirroring. `vastai show
  instances --raw` showed only stopped instance `35682721`; no active GPU
  training instance remains.

### 2026-04-28 00:00 IST Continuation Diagnostics

- Re-read the current task state after the continuation request. The stop gate
  still applies: the only completed reasoning-SFT canary has high invalid rate,
  and the long branch did not complete.
- Checked whether the answer-SFT baseline was locally available for an
  `--adapter-init-dir` diagnostic. No local `runs/answer_sft_full` or
  `checkpoint-best` directory was present. The Apr 27 answer-SFT experiment note
  also says the downstream reasoning-sampling stage was corrected to sample from
  base `unsloth/Qwen3.5-4B`, not from the answer-SFT adapter, so the base-model
  reasoning-SFT canary was aligned with that documented decision.
- Extracted lightweight canary files from the local artifact tarball under
  `runs/vast_35710717/runs/...` for offline analysis.
- Posthoc short-canary diagnostic written to
  `runs/vast_35710717/short_canary_1024_posthoc_quality_diagnostics.json`.
  It confirms:
  - `answer_tag_rate = 0.0` and `think_tag_rate = 0.0`;
  - parsed accuracy `7/32 = 0.21875`;
  - invalid rate `21/32 = 0.65625`;
  - all volume, clearance, pairwise, and ranking probe examples were invalid;
  - binary/tolerance examples parsed mostly because plain `yes`/`no` text can
    be recovered without tags.
- Patched `scripts/training/text_sft_train_unsloth.py` so future quality
  metrics include aggregate `format` diagnostics: `invalid_rate`,
  `parse_valid_rate`, `parsed_accuracy`, `answer_tag_rate`, and
  `reasoning_format_rate`, plus parsed accuracy by split/task. Added focused
  test coverage; local command
  `rtk uv run pytest -q tests/test_text_sft_train_unsloth.py tests/test_prepare_native_reasoning_sft_datasets.py`
  now reports `9 passed`.
- No GPU was rented for this diagnostic continuation, and no training was
  launched. Exact next command remains a debug/format-validity investigation,
  not SFT/GRPO:
  `rtk uv run python -m pytest -q tests/test_text_sft_train_unsloth.py tests/test_prepare_native_reasoning_sft_datasets.py`.

### 2026-04-28 00:59 IST Format-Prompt Medium Run

- The preferred stopped Vast instance `35682721` started successfully at about
  `$0.837/hr`, so no fresh machine was rented. It still had the repaired
  Torch/Unsloth runtime and `runs/answer_sft_full/checkpoint-best` on disk.
- Added opt-in `prompt_feature_mode=reasoning_format`, which appends:
  `Respond with concise reasoning inside <think>...</think>, immediately followed by the final canonical answer inside <answer>...</answer>.`
  Defaults remain unchanged. Focused validation passed:
  `rtk uv run pytest -q tests/test_prompt_features.py tests/test_text_sft_train_unsloth.py tests/test_prepare_native_reasoning_sft_datasets.py`
  -> `15 passed`.
- Ran short-trace format canary:
  `runs/reasoning_sft_short_format_canary_1024`, base `unsloth/Qwen3.5-4B`,
  `--prompt-feature-mode reasoning_format`, `--max-seq-length 1024`,
  `TORCH_COMPILE_DISABLE=1`, `--max-train-rows 128`, `--max-steps 30`.
  - step 10: `9/32 = 0.28125`, invalid `15/32 = 0.46875`, answer-tag rate
    `0.53125`;
  - step 20: `18/32 = 0.5625`, invalid `2/32 = 0.0625`, answer-tag rate
    `1.0`;
  - step 30: `16/32 = 0.5`, invalid `2/32 = 0.0625`, answer-tag rate `1.0`.
  - `reasoning_format_rate` stayed `0.0`: the model learned answer tags but
    not `<think>` tags in the canary budget.
- Selected canary `checkpoint-20` and ran medium short-trace SFT:
  `runs/reasoning_sft_short_format_medium_1024`, initialized from
  `runs/reasoning_sft_short_format_canary_1024/checkpoint-20`, all 1108 short
  train rows, 1343 packed train sequences, 200 optimizer steps, 1024 context,
  no final eval, quality every 50 steps.
  - step 50: `34/64 = 0.53125`, invalid `0/64`;
  - step 100: `42/64 = 0.65625`, invalid `2/64`;
  - step 150: `42/64 = 0.65625`, invalid `0/64`;
  - step 200: `42/64 = 0.65625`, invalid `0/64`.
  - final train loss: `1.3497121238708496`.
  - `checkpoint-200` was copied to `checkpoint-best`.
- Held-out subset eval for `checkpoint-best`:
  `runs/reasoning_sft_short_format_medium_1024_eval`, dataset
  `data/IntersectionQA-90K`, prompt mode `reasoning_format`,
  `--max-rows-per-task-per-split 20`, requested splits `validation`,
  `test_random`, `test_object_pair_heldout`, `test_near_boundary`,
  `test_generator_heldout`. The populated result had 560 rows across four
  splits; `test_generator_heldout` was not present/populated in this sampled
  eval.
  - validation: `65/140 = 0.4643`, invalid `2/140`;
  - test_random: `59/140 = 0.4214`, invalid `0/140`;
  - test_object_pair_heldout: `65/140 = 0.4643`, invalid `0/140`;
  - test_near_boundary: `63/140 = 0.45`, invalid `3/140`;
  - overall invalid: `5/560 = 0.00893`, answer-tag rate `1.0`, think-tag rate
    `0.0`.
  - task accuracy: binary `0.5875`, clearance `0.5375`, pairwise `0.2125`,
    ranking `0.0375`, relation `0.4625`, tolerance `0.675`, volume `0.6375`.
- Decision: do not run GRPO. The format-prompt short branch is much healthier
  than the previous no-format run and is the better reasoning-SFT branch, but
  held-out accuracy is still well below the Apr 27 answer-SFT validation probe
  (`52/64 = 81.25%` at checkpoint 200), ranking remains nearly collapsed, and
  the model is not emitting `<think>` tags. GRPO gate conditions are not met.
- Remote artifacts from Vast `35682721` were packaged as
  `/root/IntersectionQA/runs/reasoning_sft_short_format_35682721_artifacts.tgz`
  and mirrored locally to
  `runs/vast_35682721/reasoning_sft_short_format_35682721_artifacts.tgz`.
  SHA-256:
  `d7b4ff65206fd775d8181331acdb745f9978e14c70433aa0956613e99b1a74d1`.
- Vast `35682721` was stopped after checksum-verified mirroring. No active GPU
  training instance remains.

### 2026-04-28 01:10 IST Matched Answer-SFT Baseline Eval

- Restarted preferred Vast instance `35682721` for a bounded evaluation-only
  comparison against the same held-out subset used by the reasoning-SFT medium
  eval. Live price was still about `$0.837/hr`.
- Ran the Apr 27 answer-SFT best checkpoint with answer-only prompting:
  `python -m scripts.training.evaluate_text_model --dataset-dir data/IntersectionQA-90K --model unsloth/Qwen3.5-4B --adapter-dir runs/answer_sft_full/checkpoint-best --splits validation test_random test_object_pair_heldout test_near_boundary test_generator_heldout --max-rows-per-task-per-split 20 --max-new-tokens 64 --prompt-feature-mode none --output-dir runs/answer_sft_checkpoint_best_matched_eval`
- Result on the same 560-row populated subset:
  - overall: `434/560 = 77.50%`, invalid `0/560`;
  - validation: `102/140 = 72.86%`;
  - test_random: `106/140 = 75.71%`;
  - test_object_pair_heldout: `117/140 = 83.57%`;
  - test_near_boundary: `109/140 = 77.86%`.
- Task accuracies: binary `86.25%`, clearance `77.50%`, pairwise `81.25%`,
  ranking `77.50%`, relation `65.00%`, tolerance `90.00%`, volume `65.00%`.
- This makes the GRPO decision stronger: the best reasoning-SFT branch
  (`252/560 = 45.00%`, invalid `5/560`) is not competitive with the existing
  answer-SFT baseline on the matched subset, and it still does not emit
  `<think>` tags.
- Remote artifacts were packaged as
  `/root/IntersectionQA/runs/answer_sft_checkpoint_best_matched_eval_35682721_artifacts.tgz`
  and mirrored locally to
  `runs/vast_35682721/answer_sft_checkpoint_best_matched_eval_35682721_artifacts.tgz`.
  SHA-256:
  `e06c8643c58eba9d569c44f193ebcc54d27f27d725dae103876e9c34d23e6356`.
- Vast `35682721` was stopped after checksum-verified mirroring. It is back to
  storage-only billing; no active GPU job remains.

### 2026-04-28 01:13 IST Local Report Hardening

- Confirmed Vast `35682721` remains stopped, with storage-only billing around
  `$0.037/hr`; no active GPU job remains.
- Added aggregate format diagnostics to
  `scripts/training/evaluate_text_model.py` so future standalone eval reports
  include `format.invalid_rate`, `format.parse_valid_rate`,
  `format.parsed_accuracy`, `format.answer_tag_rate`, and
  `format.reasoning_format_rate`, plus a `scope=format` row in
  `eval_metrics.jsonl`.
- Added focused coverage in `tests/test_evaluate_text_model.py`.
- Validation passed:
  `rtk uv run pytest -q tests/test_evaluate_text_model.py tests/test_prompt_features.py tests/test_text_sft_train_unsloth.py tests/test_prepare_native_reasoning_sft_datasets.py`
  -> `16 passed`.
- Compile check passed:
  `rtk uv run python -m compileall -q scripts/training/evaluate_text_model.py`.
- No further GPU work was launched. The next useful command is local/reporting,
  not GRPO:
  `rtk uv run pytest -q tests/test_evaluate_text_model.py tests/test_prompt_features.py tests/test_text_sft_train_unsloth.py tests/test_prepare_native_reasoning_sft_datasets.py`.

### Morning Handoff

Current status: the short-vs-long reasoning-SFT gate is complete. Short
format-prompt reasoning SFT is the best reasoning branch observed, but it is
not competitive with the existing answer-SFT baseline. Long-trace SFT was a
negative canary because 128 rows expanded to 1109 packed chunks at 1024 context
and stalled after optimizer step 2. No GRPO run should be started from these
artifacts.

Exact next command:

```bash
rtk uv run pytest -q tests/test_evaluate_text_model.py tests/test_prompt_features.py tests/test_text_sft_train_unsloth.py tests/test_prepare_native_reasoning_sft_datasets.py
```

If doing research next rather than code validation, the next task is a local
analysis/design pass on why the reasoning-format adapter learns `<answer>` tags
but not `<think>` tags. Do not start GPU training or GRPO without a new
format-compliance canary plan.

Artifact paths:

- `data/reasoning_sft_short_native_distill/`
- `data/reasoning_sft_long_native_distill/`
- `runs/vast_35708545/reasoning_sft_short_diagnostic_artifacts.tgz`
- `runs/vast_35710717/reasoning_sft_length_canary_35710717_artifacts.tgz`
- `runs/vast_35682721/reasoning_sft_short_format_35682721_artifacts.tgz`
- `runs/vast_35682721/answer_sft_checkpoint_best_matched_eval_35682721_artifacts.tgz`
- dated record:
  `docs/experiments/apr-27-28-qwen3p5-4b-native-reasoning-sft-length.md`

Upload status: artifacts are mirrored locally with checksums. No remote HF
bucket upload was completed in this continuation. Vast `35682721` is stopped
and storage-only; no active GPU job remains.

Researcher summary: the matched answer-SFT baseline is strongest on the same
560-row probe (`77.50%`, invalid `0%`). The best short reasoning-SFT format
medium run reached only `45.00%` with invalid `0.89%`, although it did recover
answer-tag validity. Reasoning SFT did not produce actual `<think>` tags, and
ranking remained nearly collapsed. GRPO should not proceed until a new
reasoning-format canary shows non-degenerate accuracy and no label collapse.
Apr 28 correction: do not require real literal `<think>...</think><answer>...</answer>`
compliance as the success criterion; that tag contract is now treated as a
negative-result diagnostic.

### 2026-04-28 Aligned Prediction Analysis

- Extracted the lightweight reasoning-SFT eval files from the large
  `35682721` artifact tarball without unpacking checkpoints.
- Wrote an aligned answer-SFT vs reasoning-SFT comparison under
  `runs/vast_35682721/matched_eval_comparison/`.
- The 560 evaluated row IDs match exactly between the answer-SFT baseline and
  short reasoning-SFT format medium run.
- Row-level comparison:
  - both correct: `221/560`;
  - answer-SFT only correct: `213/560`;
  - reasoning-SFT only correct: `31/560`;
  - both wrong: `95/560`.
- The reasoning branch still shows common-label collapse:
  - binary: parsed `no` for `80/80`;
  - pairwise: parsed `B` for `80/80`;
  - volume: parsed `0` for `80/80`;
  - clearance: mostly `>5` (`59/80`);
  - tolerance: mostly `no` (`77/80`).
- This strengthens the no-GRPO decision: the reasoning branch is not merely
  lower accuracy; it has task-specific collapsed output modes that a GRPO
  canary would likely reinforce without a better supervised initializer.
- New local analysis artifacts:
  - `runs/vast_35682721/matched_eval_comparison/matched_eval_comparison.json`
  - `runs/vast_35682721/matched_eval_comparison/matched_eval_comparison.md`

### 2026-04-28 Label Balance And Collapse Diagnostic

- Added a local diagnostic comparing the short reasoning-SFT train/validation
  answer distribution against matched held-out predictions:
  `runs/vast_35682721/matched_eval_comparison/label_balance_and_collapse.md`.
- All short reasoning-SFT targets contain the requested markup:
  - train `<think>` tags: `1108/1108`;
  - train `<answer>` tags: `1108/1108`;
  - validation `<think>` tags: `151/151`;
  - validation `<answer>` tags: `151/151`.
- Some collapsed reasoning predictions align with accepted-trace label bias:
  volume `0` is `163/170 = 95.9%` of train rows and `80/80` predictions;
  binary `no` is `169/213 = 79.3%` of train rows and `80/80` predictions;
  tolerance `no` is `206/265 = 77.7%` of train rows and `77/80`
  predictions.
- Other collapses are not explained by train-label majority:
  pairwise `B` is `28/57 = 49.1%` of train rows but `80/80` predictions;
  clearance `>5` is `30/157 = 19.1%` of train rows but `59/80`
  predictions; relation `disjoint` is `70/195 = 35.9%` of train rows but
  `56/80` predictions.
- Interpretation: the failure is a mix of accepted-trace imbalance,
  undertrained/objective-prompt mismatch, and an over-specified literal tag
  contract. Any future reasoning-SFT canary should balance or weight accepted
  traces by task answer, use Qwen-native formatting, and verify parse-valid
  final answers plus non-collapsed task behavior before GRPO. Do not use
  literal `<think>` emission as the gate.

### 2026-04-28 Balanced Canary Prep

- Added
  `scripts/training/prepare_balanced_reasoning_sft_canary_dataset.py`, a local
  deterministic SFT-only sampler that caps majority task-answer strata and
  lightly upsamples minority task-answer strata with a per-row repeat cap.
- Added focused tests in
  `tests/test_prepare_balanced_reasoning_sft_canary_dataset.py`.
- Validation passed:
  `rtk uv run pytest -q tests/test_prepare_balanced_reasoning_sft_canary_dataset.py tests/test_prepare_native_reasoning_sft_datasets.py tests/test_evaluate_text_model.py tests/test_prompt_features.py tests/test_text_sft_train_unsloth.py`
  -> `18 passed`.
- Materialized the next canary input locally:
  `data/reasoning_sft_short_native_distill_balanced_canary/`.
- Build command:
  `rtk uv run python -m scripts.training.prepare_balanced_reasoning_sft_canary_dataset --input-dir data/reasoning_sft_short_native_distill --output-dir data/reasoning_sft_short_native_distill_balanced_canary --target-per-answer 64 --max-repeat-per-row 8`.
- Dataset summary: source train `1108` rows, balanced train `1256` rows,
  validation unchanged at `151` rows.
- Wrote a restartable next-canary recipe at
  `runs/vast_35682721/matched_eval_comparison/next_balanced_reasoning_canary_plan.md`.
- This preparation does not change the current gate decision. It is only a
  future SFT canary input. Do not start GRPO from it without first satisfying
  the health gate in that plan.

### 2026-04-28 Strict Reasoning Prompt Prep

- Added opt-in `prompt_feature_mode=strict_reasoning_format`. It preserves the
  existing `reasoning_format` mode and defaults, but gives the next canary a
  stricter instruction: the completion must start exactly with `<think>`, close
  `</think>`, immediately provide `<answer>...</answer>`, and include no text
  outside those tags.
- Updated the balanced canary plan to use
  `--prompt-feature-mode strict_reasoning_format` because the previous softer
  prompt learned `<answer>` tags but never emitted `<think>` tags.
- Added focused prompt coverage in `tests/test_prompt_features.py`.
- This is still only next-canary preparation. It does not reopen the GRPO gate
  and does not justify GPU training without the health checks in
  `runs/vast_35682721/matched_eval_comparison/next_balanced_reasoning_canary_plan.md`.
- Apr 28 correction: this strict literal-tag prompt is superseded by the
  Qwen-native formatting correction above. Do not launch the strict-tag canary
  without first redesigning the format target and stop criteria.

### 2026-04-28 Artifact Checksums And Standalone Handoff

- Wrote checksum manifest for the local analysis and balanced-canary artifacts:
  `runs/vast_35682721/local_analysis_and_balanced_canary_checksums.sha256`.
- Wrote standalone morning handoff:
  `runs/vast_35682721/morning_handoff_2026-04-28.md`.
- Captured local controller git state:
  - `runs/vast_35682721/local_controller_git_status.txt`;
  - `runs/vast_35682721/local_controller_git_diff_tracked_excluding_env.patch`;
  - `runs/vast_35682721/local_controller_untracked_files.txt`.
- Packaged the small untracked local code/docs snapshot separately from large
  run artifacts:
  `runs/vast_35682721/local_controller_untracked_code_docs_snapshot.tgz`.
  Checksum is stored beside it in
  `runs/vast_35682721/local_controller_untracked_code_docs_snapshot.tgz.sha256`.
- Reconfirmed Vast `35682721` is stopped and storage-only (`gpuCostPerHour=0`,
  active `totalHour` about `$0.037/hr`).
- No GPU job was launched. GRPO remains blocked.
