# Apr 27-28 2026 Qwen3.5 4B Native Reasoning SFT Length Canary

## Purpose

Compare short native-reasoning traces against long native-reasoning traces for
IntersectionQA reasoning SFT. The intended gate was: build short and long SFT
datasets, audit short proofs, run symmetric SFT canaries, continue only if the
SFT gate is healthy, then consider GRPO.

Apr 28 correction: the literal `<think>...</think><answer>...</answer>` target
format used in this canary is now treated as a negative-result diagnostic, not
the correct Qwen3.5 formatting contract. It over-specified the output format,
contributed to answer collapse, and should not be used as the health gate for
future reasoning SFT/GRPO. Future attempts should use Qwen-native chat/template
formatting and gate on parse-valid final answers plus non-collapsed task
behavior.

Apr 28 correction: the short-trace compression in this run was also not
protocol-compliant with the original instruction to use OpenRouter
`deepseek/deepseek-v4-pro` or a similar paid model for 256-token shortening.
The dataset was created by a local deterministic sentence extractor,
`deterministic_native_reasoning_sentence_extractor_v1`, as recorded in
`data/reasoning_sft_short_native_distill/native_reasoning_sft_report.json`.
Therefore this experiment should be read as a cheap heuristic-shortening
diagnostic, not the requested OpenRouter-compressed short-trace condition.

## Dataset

- Source dataset: `data/IntersectionQA-90K`, from `MRiabov/IntersectionQA-90K`.
- Accepted native trace inputs:
  - `runs/openrouter_native_reasoning_distill_1000/accepted_reasoning_sft.jsonl`
  - `runs/openrouter_native_reasoning_distill_unexposed_1332/accepted_reasoning_sft.jsonl`
- Combined accepted rows: 1259 across seven task types.
- Materialized outputs:
  - `data/reasoning_sft_short_native_distill/`
  - `data/reasoning_sft_long_native_distill/`
- Group-safe train/validation split from public-train rows: 1108 train, 151
  validation.

Short trace stats: train mean 134.0 reasoning whitespace tokens, min 128, max
203; validation mean 134.0, min 128, max 155.

Shortener used: local deterministic sentence extraction from existing accepted
native-reasoning traces. No OpenRouter balance was used for this shortening
step.

Long trace stats: train mean 3300.7 reasoning whitespace tokens, validation
mean 3305.1. Treat this branch as truncation-sensitive until measured.

Manual audit of `data/reasoning_sft_short_native_distill/manual_audit_sample.jsonl`
covered 21 rows across all seven task types: 17 valid, 4 too vague, 0
fallacious, 0 answer-mismatched, 0 label-leak-only. The short dataset passed the
proof audit gate.

## Local Validation

Passed:

```bash
rtk uv run python -m compileall -q intersectionqa scripts
rtk uv run pytest -q tests/test_reasoning_traces.py tests/test_text_sft_train_unsloth.py tests/test_rewards.py tests/test_experiments.py
rtk uv run python -m scripts.experiments.run_experiment_suite configs/overnight_experiment_suite.yaml --list
rtk uv run pytest -q tests/test_prepare_native_reasoning_sft_datasets.py tests/test_reasoning_traces.py
```

## Remote Hardware

- Preferred stopped Vast instance `35682721` remained stopped after a start
  request; resources were unavailable.
- Rented fallback Vast contract `35708545`: `NVIDIA A100-SXM4-80GB`, 200GB
  disk, about `$0.969/hr`.
- The fallback instance was destroyed after local artifact mirroring.
- Remaining Vast state after teardown: only stopped instance `35682721`.

## Runtime Notes

Bootstrap required runtime repair:

- Initial import failed because `torchao` expected newer Torch symbols.
- Removed incompatible `torchao`, then installed missing `fla-core==0.5.0`.
- `fla-core` pulled incompatible Torch 2.11, so the environment was repaired to:
  Torch `2.10.0+cu128`, transformers `5.5.0`, TRL `0.24.0`, Unsloth
  `2026.4.8`, torchvision `0.25.0`, torchaudio `2.10.0`, xformers `0.0.35`,
  torchao `0.17.0`.
- `causal_conv1d` stayed disabled by Unsloth as a broken binary; Flash
  Attention 2 was unavailable and xformers was used.

## Runs

Short canary command was launched with explicit `--save-steps 10` and
`--quality-max-new-tokens 256` added to make checkpoint and reasoning-quality
artifacts possible. It failed twice during runtime repair, then stalled after
step 2 with no log/artifact progress for about eight minutes. The stalled run
was stopped and preserved.

Diagnostic short run:

```bash
python -m scripts.training.text_sft_train_unsloth \
  --dataset-dir data/reasoning_sft_short_native_distill \
  --output-dir runs/reasoning_sft_short_diagnostic \
  --pack-tokenized \
  --max-seq-length 1024 \
  --max-train-rows 32 \
  --max-eval-rows 16 \
  --quality-eval-max-rows 16 \
  --max-steps 5 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --eval-strategy no \
  --no-final-eval \
  --final-adapter-save-mode checkpoint \
  --save-steps 5 \
  --resume \
  --quality-eval-steps 5 \
  --quality-max-new-tokens 256
```

Diagnostic result:

- Completed 5 optimizer steps.
- Quality callback ran at step 5: train-probe accuracy `3/16 = 0.1875`.
- `train_result.json` reported `train_loss: NaN`.
- NaN loss is a hard stop. The long-trace SFT canary and GRPO were not run.

## Artifacts

Local artifacts:

- `data/reasoning_sft_short_native_distill/native_reasoning_sft_report.json`
- `data/reasoning_sft_long_native_distill/native_reasoning_sft_report.json`
- `data/reasoning_sft_short_native_distill/manual_audit_results.md`
- `runs/vast_35708545/reasoning_sft_short_diagnostic_artifacts.tgz`
- `runs/vast_35708545/reasoning_sft_short_diagnostic_artifacts.tgz.sha256`

Artifact checksum:

```text
8215d1ec9a223fbf5cce024e7884c55b0809c6cb997bf3d3f4a2eb35e5c923d7
```

The tarball contains the diagnostic run directory, adapter/checkpoint-5,
training and quality metrics, quality predictions, command text, remote
environment notes, git status/diff, and failed/stalled short-canary logs.

Upload status: local mirror only. The GPU instance was destroyed after checksum
verification to avoid further spend.

## Decision

Do not proceed to long-trace SFT or GRPO from this run. The short-trace dataset
itself passed audit, but the remote SFT runtime is not healthy because the only
completed optimizer run ended with NaN loss. Next work should first fix or
replace the Qwen3.5/Unsloth runtime and run a tiny 2-5 step loss sanity check
before any symmetric canary.

## Continuation After NaN Diagnosis

The NaN was traced to packed examples that could contain only ignored labels
when long CAD prompts filled the context before assistant answer tokens. The SFT
runner now drops all-ignored packed chunks, masks EOS only when the example has
supervised target tokens, and raises if packing produces no supervised chunks.

Local focused validation:

```bash
rtk uv run pytest -q tests/test_text_sft_train_unsloth.py tests/test_prepare_native_reasoning_sft_datasets.py
```

Result: `7 passed`.

New Vast contract `35710717` was rented after preferred instance `35682721`
again failed to start. Hardware: `NVIDIA A100-SXM4-80GB`, 120GB disk, about
`$1.0867/hr`. Runtime was repaired to Torch `2.10.0+cu128`, transformers
`5.5.0`, TRL `0.24.0`, Unsloth `2026.4.8`, xformers `0.0.35`, torchao
`0.17.0`, and `fla-core==0.5.0` with dependencies disabled.

Patched loss sanity command:

```bash
python -m scripts.training.text_sft_train_unsloth \
  --dataset-dir data/reasoning_sft_short_native_distill \
  --output-dir runs/reasoning_sft_short_loss_sanity \
  --pack-tokenized \
  --max-seq-length 1024 \
  --max-train-rows 32 \
  --max-eval-rows 16 \
  --quality-eval-max-rows 8 \
  --max-steps 5 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --eval-strategy no \
  --no-final-eval \
  --final-adapter-save-mode checkpoint \
  --save-steps 5 \
  --resume \
  --quality-eval-steps 5 \
  --quality-max-new-tokens 256
```

Result: completed 5 optimizer steps with finite `train_loss =
2.450699234008789`; 32 train rows became 41 packed train sequences, and 16 eval
rows became 20 packed eval sequences. This clears the NaN blocker for the
patched packed-loss path. Symmetric short/long canaries are the next gate.

## Canary Continuation

A second scoped trainer change fixed packed-batch padding to
`--max-seq-length`, avoiding one-sample variable-shape batches in the packed
path. Focused local validation then passed with `8 passed`.

The 2048-context short canary path was still unhealthy. Attempts with fixed
padding, `gradient_accumulation_steps=1`, and `TORCH_COMPILE_DISABLE=1` all
reached optimizer step 2 and then made no metric/checkpoint progress while
consuming GPU. These attempts are preserved in the artifact bundle:

- `runs/reasoning_sft_short_canary_fixed`
- `runs/reasoning_sft_short_canary_fixed_pad`
- `runs/reasoning_sft_short_canary_ga1`
- `runs/reasoning_sft_short_canary_no_compile`

Conservative 1024-context short canary:

```bash
TORCH_COMPILE_DISABLE=1 python -m scripts.training.text_sft_train_unsloth \
  --dataset-dir data/reasoning_sft_short_native_distill \
  --output-dir runs/reasoning_sft_short_canary_1024 \
  --pack-tokenized \
  --max-seq-length 1024 \
  --max-train-rows 128 \
  --max-eval-rows 64 \
  --quality-eval-max-rows 32 \
  --max-steps 10 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --eval-strategy no \
  --no-final-eval \
  --final-adapter-save-mode checkpoint \
  --save-steps 10 \
  --resume \
  --quality-eval-steps 10 \
  --quality-max-new-tokens 256
```

Result: completed 10 optimizer steps with finite `train_loss =
1.8363332748413086`; 128 train rows became 158 packed train sequences, and 64
eval rows became 79 packed eval sequences. Quality probe on 32 train rows was
`7/32 = 0.21875`, but `21/32 = 0.65625` completions were unparsable. Parsed
predictions were mostly `no` (9) or `intersecting` (2), with the rest invalid.
Completion word lengths were min 1, mean 84.0, median 119.5, max 151.

Conservative 1024-context long canary used the same setup with
`--quality-max-new-tokens 512`. It expanded 128 long rows to 1109 packed train
sequences, confirming severe truncation/chunking pressure, then stalled after
optimizer step 2 with no metrics/checkpoint/result and was stopped.

## Final Decision

Do not run medium/full SFT or GRPO from these artifacts. The only completed
branch is the short-trace 1024 canary, and it has a high invalid-output rate.
The long-trace branch did not pass the canary health gate. The next useful work
is runtime/debugging, not a longer training launch: investigate the step-2 hang
at 2048 context and improve answer-format validity before reconsidering full
SFT or GRPO.

Artifacts from Vast `35710717`:

- `runs/vast_35710717/reasoning_sft_length_canary_35710717_artifacts.tgz`
- `runs/vast_35710717/reasoning_sft_length_canary_35710717_artifacts.tgz.sha256`

SHA-256:

```text
afdfaaf6c3486c2b1476709f8670607af4bf6577ae39ccbbaa45bf54dded6c70
```

Vast `35710717` was destroyed after the artifact mirror and checksum
verification. The only remaining Vast instance shown afterward was stopped
instance `35682721`.

## Posthoc Format Diagnostic

After continuing locally without renting another GPU, the short-canary
prediction file was analyzed directly:

```text
runs/vast_35710717/runs/reasoning_sft_short_canary_1024/predictions/quality_step_10.jsonl
```

Diagnostic summary:

- parsed accuracy: `7/32 = 0.21875`
- invalid rate: `21/32 = 0.65625`
- `<answer>` tag rate: `0/32 = 0.0`
- `<think>` tag rate: `0/32 = 0.0`
- all volume, clearance, pairwise, and ranking probe examples were invalid
- binary/tolerance examples were parseable mostly because plain `yes`/`no`
  could be recovered without tags

The diagnostic JSON is saved at:

```text
runs/vast_35710717/short_canary_1024_posthoc_quality_diagnostics.json
```

The SFT runner was patched so future quality metrics report aggregate
`invalid_rate`, `parse_valid_rate`, `parsed_accuracy`, `answer_tag_rate`, and
`reasoning_format_rate`, plus parsed accuracy by split/task. Focused local
validation passed:

```bash
rtk uv run pytest -q tests/test_text_sft_train_unsloth.py tests/test_prepare_native_reasoning_sft_datasets.py
```

Result: `9 passed`.

## Format-Prompt Recovery Run

The stopped preferred Vast instance `35682721` was restarted successfully at
about `$0.837/hr`; no new machine was rented. It still contained the repaired
Torch/Unsloth environment and the Apr 27 answer-SFT artifacts.

The invalid-output diagnosis showed a prompt/target mismatch: the target rows
all contained `<think>...</think><answer>...</answer>`, but the original public
prompts did not explicitly ask for that response contract. A new opt-in
`prompt_feature_mode=reasoning_format` was added. It appends this instruction:

```text
Respond with concise reasoning inside <think>...</think>, immediately followed by the final canonical answer inside <answer>...</answer>.
```

Defaults remain unchanged. Focused validation passed:

```bash
rtk uv run pytest -q tests/test_prompt_features.py tests/test_text_sft_train_unsloth.py tests/test_prepare_native_reasoning_sft_datasets.py
```

Result: `15 passed`.

Short format canary:

- run: `runs/reasoning_sft_short_format_canary_1024`
- model: base `unsloth/Qwen3.5-4B`
- prompt mode: `reasoning_format`
- context: 1024
- `TORCH_COMPILE_DISABLE=1`
- 128 train rows
- 30 optimizer steps

Quality progression on the fixed 32-row probe:

| Step | Correct | Accuracy | Invalid | Answer-tag Rate |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 9 / 32 | 28.13% | 15 / 32 | 53.13% |
| 20 | 18 / 32 | 56.25% | 2 / 32 | 100% |
| 30 | 16 / 32 | 50.00% | 2 / 32 | 100% |

`reasoning_format_rate` remained `0.0`; the model learned final-answer tags
but not `<think>` tags in the canary budget. Checkpoint 20 was selected for the
medium run.

Medium short-trace SFT:

- run: `runs/reasoning_sft_short_format_medium_1024`
- initialized from `runs/reasoning_sft_short_format_canary_1024/checkpoint-20`
- all 1108 short-trace train rows
- 1343 packed train sequences
- 200 optimizer steps
- 1024 context
- final train loss: `1.3497121238708496`
- copied `checkpoint-200` to `checkpoint-best`

Quality progression on the fixed 64-row probe:

| Step | Correct | Accuracy | Invalid |
| ---: | ---: | ---: | ---: |
| 50 | 34 / 64 | 53.13% | 0 / 64 |
| 100 | 42 / 64 | 65.63% | 2 / 64 |
| 150 | 42 / 64 | 65.63% | 0 / 64 |
| 200 | 42 / 64 | 65.63% | 0 / 64 |

Held-out subset evaluation for `checkpoint-best`:

- run: `runs/reasoning_sft_short_format_medium_1024_eval`
- dataset: `data/IntersectionQA-90K`
- prompt mode: `reasoning_format`
- max rows per task per split: 20
- requested splits: `validation`, `test_random`, `test_object_pair_heldout`,
  `test_near_boundary`, `test_generator_heldout`
- populated/evaluated rows: 560 across four splits; `test_generator_heldout`
  was not present/populated in this sampled eval

Split results:

| Split | Correct | Accuracy | Invalid |
| --- | ---: | ---: | ---: |
| validation | 65 / 140 | 46.43% | 2 / 140 |
| test_random | 59 / 140 | 42.14% | 0 / 140 |
| test_object_pair_heldout | 65 / 140 | 46.43% | 0 / 140 |
| test_near_boundary | 63 / 140 | 45.00% | 3 / 140 |

Task results:

| Task | Accuracy |
| --- | ---: |
| binary_interference | 58.75% |
| clearance_bucket | 53.75% |
| pairwise_interference | 21.25% |
| ranking_normalized_intersection | 3.75% |
| relation_classification | 46.25% |
| tolerance_fit | 67.50% |
| volume_bucket | 63.75% |

Overall invalid rate was `5/560 = 0.89%`. Answer-tag rate was `100%`; think-tag
rate remained `0%`.

## Updated Decision

The format-prompt short branch is the best reasoning-SFT branch observed so far
and is much healthier than the no-format canary. It still should not proceed to
GRPO overnight: held-out accuracy is well below the Apr 27 answer-SFT validation
probe (`52/64 = 81.25%`), ranking remains nearly collapsed, and the model does
not emit `<think>` tags despite the target format. The next useful step is
prompt/reward/decoding work around actual reasoning-format compliance, not GRPO.

## Matched Answer-SFT Baseline Eval

After the reasoning-SFT medium run, the stopped preferred Vast instance
`35682721` was restarted for a bounded evaluation-only baseline comparison. The
instance was still the A100 PCIe 80GB contract at about `$0.837/hr`, and it was
stopped again after artifact mirroring.

Command:

```bash
python -m scripts.training.evaluate_text_model \
  --dataset-dir data/IntersectionQA-90K \
  --model unsloth/Qwen3.5-4B \
  --adapter-dir runs/answer_sft_full/checkpoint-best \
  --splits validation test_random test_object_pair_heldout test_near_boundary test_generator_heldout \
  --max-rows-per-task-per-split 20 \
  --max-new-tokens 64 \
  --prompt-feature-mode none \
  --output-dir runs/answer_sft_checkpoint_best_matched_eval
```

The populated eval subset again contained 560 rows across four splits. The
answer-SFT baseline was much stronger than the best reasoning-SFT branch:

| Model | Correct | Accuracy | Invalid |
| --- | ---: | ---: | ---: |
| answer-SFT `checkpoint-best` | 434 / 560 | 77.50% | 0 / 560 |
| short reasoning-SFT format medium | 252 / 560 | 45.00% | 5 / 560 |

Answer-SFT split results:

| Split | Correct | Accuracy |
| --- | ---: | ---: |
| validation | 102 / 140 | 72.86% |
| test_random | 106 / 140 | 75.71% |
| test_object_pair_heldout | 117 / 140 | 83.57% |
| test_near_boundary | 109 / 140 | 77.86% |

Answer-SFT task results:

| Task | Accuracy |
| --- | ---: |
| binary_interference | 86.25% |
| clearance_bucket | 77.50% |
| pairwise_interference | 81.25% |
| ranking_normalized_intersection | 77.50% |
| relation_classification | 65.00% |
| tolerance_fit | 90.00% |
| volume_bucket | 65.00% |

This matched comparison closes the main remaining caveat. The short
reasoning-SFT adapter is a useful format-validity recovery from the earlier
invalid-output canary, but it is not a better initializer than the existing
answer-SFT baseline for GRPO.

An aligned prediction analysis was then generated from the saved JSONL files:

- path: `runs/vast_35682721/matched_eval_comparison/matched_eval_comparison.md`
- both correct: `221/560`;
- answer-SFT only correct: `213/560`;
- reasoning-SFT only correct: `31/560`;
- both wrong: `95/560`.

The aligned analysis shows that the reasoning branch has task-specific
collapsed output modes: binary always parsed as `no`, pairwise always parsed as
`B`, volume always parsed as `0`, clearance mostly parsed as `>5`, and tolerance
mostly parsed as `no`. This is the most important qualitative reason not to
start GRPO from the reasoning-SFT adapter: a reward-only stage would likely
amplify those collapsed modes unless a new supervised initializer first fixes
format compliance and label diversity.

A follow-up label-balance diagnostic found that all short reasoning-SFT targets
do include both required tags (`<think>` and `<answer>`), so missing `<think>`
tags are not caused by absent target markup. Some collapsed outputs do align
with accepted-trace imbalance: volume `0` is 95.9% of short-trace train rows,
binary `no` is 79.3%, and tolerance `no` is 77.7%. Other collapsed outputs do
not: pairwise `B` is 49.1% of train rows but 100% of predictions, clearance
`>5` is 19.1% of train rows but 73.8% of predictions, and relation `disjoint`
is 35.9% of train rows but 70.0% of predictions. Future reasoning-SFT should
balance or weight accepted traces by task answer before another GPU canary.

A local balanced canary input was prepared for that future test:
`data/reasoning_sft_short_native_distill_balanced_canary/`. It uses the audited
short-trace targets, leaves validation unchanged, caps each train task-answer
stratum at 64 rows, and upsamples minority strata with a maximum per-row repeat
cap of 8. The resulting train set has 1256 rows. This is not a new result and
does not change the no-GRPO decision; it is a prepared SFT-only canary input.
The exact next-run recipe is saved at
`runs/vast_35682721/matched_eval_comparison/next_balanced_reasoning_canary_plan.md`.
That recipe originally used a new opt-in
`prompt_feature_mode=strict_reasoning_format` that required completions to
start exactly with `<think>`, then emit `</think><answer>...</answer>` with no
surrounding text. Apr 28 correction: do not run that strict literal-tag recipe
without redesigning the response contract around Qwen-native formatting.

After the matched baseline eval, the standalone evaluator was updated so future
reports directly include aggregate format diagnostics: invalid rate, parsed
accuracy, answer-tag rate, and reasoning-format rate. Focused validation passed:

```bash
rtk uv run pytest -q tests/test_evaluate_text_model.py tests/test_prompt_features.py tests/test_text_sft_train_unsloth.py tests/test_prepare_native_reasoning_sft_datasets.py
```

Result: `16 passed`.

New artifacts from Vast `35682721`:

- `runs/vast_35682721/reasoning_sft_short_format_35682721_artifacts.tgz`
- `runs/vast_35682721/reasoning_sft_short_format_35682721_artifacts.tgz.sha256`
- `runs/vast_35682721/answer_sft_checkpoint_best_matched_eval_35682721_artifacts.tgz`
- `runs/vast_35682721/answer_sft_checkpoint_best_matched_eval_35682721_artifacts.tgz.sha256`
- `runs/vast_35682721/matched_eval_comparison/matched_eval_comparison.json`
- `runs/vast_35682721/matched_eval_comparison/matched_eval_comparison.md`
- `runs/vast_35682721/matched_eval_comparison/label_balance_and_collapse.json`
- `runs/vast_35682721/matched_eval_comparison/label_balance_and_collapse.md`
- `runs/vast_35682721/matched_eval_comparison/next_balanced_reasoning_canary_plan.md`
- `runs/vast_35682721/local_analysis_and_balanced_canary_checksums.sha256`
- `runs/vast_35682721/morning_handoff_2026-04-28.md`
- `runs/vast_35682721/local_controller_git_status.txt`
- `runs/vast_35682721/local_controller_git_diff_tracked_excluding_env.patch`
- `runs/vast_35682721/local_controller_untracked_code_docs_snapshot.tgz`
- `runs/vast_35682721/local_controller_untracked_code_docs_snapshot.tgz.sha256`

SHA-256:

```text
d7b4ff65206fd775d8181331acdb745f9978e14c70433aa0956613e99b1a74d1
e06c8643c58eba9d569c44f193ebcc54d27f27d725dae103876e9c34d23e6356
```

Vast `35682721` was stopped after checksum-verified mirroring; no active GPU
training instance remains.
