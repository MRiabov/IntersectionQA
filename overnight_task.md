Task for this night, apr 25:
I want to fine-tune qwen-3.5-4b using reinforcement learning (GRPO) on both IntersectionQA and IntersectionEdit this night.

Warning: I'm sleeping, so you are on your own. I will send you automated reminders to continue, but you need to continue on your own anyway.

Important: the current sft pipeline with "only one answer" doesn't exactly fit for GRPO. GRPO needs reasoning tokens, so we need to define prompts an architecture for this to happen. I expect we need at least 2k max new tokens? In sft we had `batch_size = 32` I think, but I'm not sure how much will fit into H100/A100

We have a set of tasks today:
1. Implement the epic 15. Basically, make the dataset publishable. In fact, it would be great if you publish it on huggingface this night, as I have already done with IntersectionQA. Make splits for the IntersectionEdit.
    - We had a bug (bad logic) during intersectionQA implementation: we had imbalanced splits of datasets during training which led to poor SFT performance - because some class was hitting only with 10% chance, it fit very poorly to that class.
    - As such, ensure balanced datasets.
2. An important task: we don't want label contamination with SFT or RL. As such, split the train *internally* into RL and SFT. it's not split on dataset release level, rather we simply split it as consumers of the dataset. 
3. Do all the architecture for GRPO to happen. I think we already have it.
4. The most important part of the night: we want to test train the model on GRPO. So I want you to try and beat 
    - Note that "beat" may be fairly ill-defined here. Because in sft pipeline we had tall results with predicting with "max_output_tokens=16", which meant the pipeline was very cheap.
    - With that, that SFT pipeline would lose generality of the model. And to solve it, we either need to SFT on reasoning traces or we simply need to RL train the model. And I think the first step is RL unless you want to generate a few dozens of SFT reasoning traces, which probably will degrade it anyway. On the other hand, RL will be costly - we need much more compute to make RL work. So I'm split between it. But I will need to do RL in some other pipeline (and also, I am creating intersectionQA and intersectionEdit specifically to improve that pipeline; that pipeline is CAD-using-LLM pipeline, and we would need to RL that anyway), so I'm keen to get my hands dirty with custom RL (I never did it before.), although I acknowledge that SFT may solve the problem cheaper/faster.
5. Also, we didn't have a proper evaluation wired into the training loop(s). We trained blindly. Add evals every so often so that we can track it.
6. At the end of the night, whatever the train artifacts, upload them to hf bucket. https://huggingface.co/docs/huggingface_hub/guides/buckets.md. Also update the qwen3p5-4b-tuning experiment.
7. Make sure that the IntersectionEdit is well-covered in the document.
Oh yes. Try to spend no more than 5-7$ on the Vast.ai account. Which means - you either have to a) run 3-5 hours of H100 or 6 hours of A100 80gb. Decide on your own.

(I'm not quite sure if I should do any SFT going forward. It may prove results, and in fact it did in @docs/experiments/qwen-3p5-4b-tuning.md, however I don't know...)
Also, I suspect that we have dropped all unsloth training in SFT pipeline. Unsloth is 2x faster and results could be supreme!
Also, GRPO isn't the best, there is also Dr. GRPO, there is also GSPO... I'm not sure what these do, but I know they are improvements. Check out Unsloth docs, it implements both.

Also, commit your changes every once in a while.

Budget note: we've started with 9.57$ in vast ai balance in the evening.

### Other low-priority tasks:

- I've seen we had a lot of for loops in the app. Not exactly bad but maybe use `polars` to iterate over it? 
- if the tonights' run succeeds, mark more features in @epics-and-stories.yaml as "stable".

--- 

Warn: this laptop has no GPU, you'll need to rent on Vast.

## Execution log

<!--Add checklists here as To Do and Done so that you don't have to rescan what was done or not done... though you also have @epics-and-stories.yaml for that, but that's for broader features. -->

## Locked overnight course of action

Decision: prioritize a reliable, measurable GRPO pilot over a broad but
unverified full fine-tuning sweep. SFT is fallback/bootstrap only, not the main
overnight objective. Do not spend the night comparing every RL variant.

Budget decision: cap Vast.ai spend at roughly `$5-7`. Default to an A100 80GB
run for up to 6 hours because the previous A100 path is known to work and gives
more wall-clock room for dataset fixes, canaries, and evals. Use H100 only if
current Vast pricing makes a 3-5 hour run fit inside the cap and the canary
needs the speed. Check live Vast prices before renting; do not assume the old
contract price is still available.

Container decision: optionally build the training container locally and have the
Vast instance pull the prepared image instead of installing dependencies on paid
GPU time. Use this if a Dockerfile/image path is already close to working or can
be made ready in under ~30 local minutes; it can save roughly 10-15% of the Vast
budget. Do not let container work block the dataset/reward canary path.

Autonomy rule: these instructions are the default plan, not a straitjacket. If
there is a clearly better path, it may be tried, but only if the core constraints
remain intact:

- Total Vast.ai spend stays under roughly `$7`.
- No training uses validation/test rows or leaks labels across internal splits.
- Dataset/release validation is not skipped before publishing or serious
  training.
- At least one measurable GRPO/GSPO canary or pilot is preserved with logs.
- Every meaningful deviation is documented here and in the experiment doc with
  the reason, expected benefit, and stop/rollback condition.

Prefer small, reversible deviations. Do not start a large new research branch
unless the default path is blocked or the upside is obvious within the overnight
budget.

Primary success condition by morning:

- IntersectionEdit has a validated release-candidate dataset with balanced
  public splits and documented internal SFT/RL split hygiene.
- At least one Qwen3.5 4B GRPO/GSPO pilot has run from a clean dataset split,
  with periodic eval metrics and saved adapter/artifacts.
- The experiment document records exact commands, hardware, metrics, failures,
  and artifact locations.

Secondary success condition:

- If GRPO cannot train stably, leave a working canary plus a smaller SFT or
  GRPO-debug artifact, with the failure reason and next command written down.

Non-goals for tonight:

- Do not implement multi-object repair.
- Do not chase 9B/27B/35B unless 4B is blocked for a reason unrelated to model
  size.
- Do not publish a dataset/model artifact that fails validation or has unclear
  split leakage.

## Priority order

1. Stabilize the current repo state.
   - Run the focused tests that cover the changed files.
   - Fix obvious breakages in rewards, parsing, splits, prompts, schema,
     release build, and GRPO smoke only.
   - Commit once this baseline is green.

2. Decide whether to prebuild a GPU image.
   - If a local Docker build/push is straightforward, build/push it before
     renting the Vast instance.
   - If image work takes more than ~30 local minutes or hits CUDA/package
     churn, skip it and install on the instance.
   - Record the image tag or the reason for skipping it in the experiment doc.

3. Make IntersectionEdit publishable enough for the night.
   - Build a repair/edit release candidate from `configs/repair_smoke.yaml` or
     the largest safe config available locally.
   - Validate JSONL/parquet, dataset stats, split manifests, leakage audits,
     edit verifier reports, and dataset card content.
   - Check class/task balance per split. If imbalance is found, fix balancing
     before training.
   - Publish only if validation is clean and credentials are available. If not,
     leave a tarball plus exact upload command.

4. Prevent SFT/RL label contamination.
   - Treat public `train` as the only source for training-time internal splits.
   - Use group-safe internal split helpers, with `metadata.edit_split_group`
     preferred over generic split groups.
   - Materialize or log internal assignments for GRPO/SFT runs.
   - Never mix `validation`/`test_*` into optimizer updates.

5. Make GRPO production-runnable.
   - Replace answer-only instruction with reasoning-compatible chat format:
     system asks for concise reasoning in `<think>...</think>` and final answer
     in `<answer>...</answer>`.
   - Rewards should give format credit, canonical-answer correctness, and
     IntersectionEdit partial verifier-style credit where available.
   - Use `max_completion_length` around 512 for canaries, then 1024-2048 only
     if memory and step time are acceptable.
   - Start with `num_generations=4`; increase to 8 only after the canary is
     stable.
   - Prefer Unsloth GRPO/GSPO for the serious run. Plain TRL GRPO is acceptable
     for local smoke only.

6. Run training in staged gates.
   - Gate A: local/unit smoke on tiny rows, max 1-5 steps.
   - Gate B: GPU canary, 32-128 rows, 10-20 steps, frequent logging.
   - Gate C: pilot, at least 300 steps if reward is non-degenerate.
   - Gate D: overnight extension from the best pilot checkpoint if metrics are
     improving.
   - If rewards are flat because completions are invalid, fix prompt/reward
     formatting before extending the run.
   - Hard budget gate: destroy/stop the Vast instance when projected spend
     reaches `$7`, even if the pilot has not converged.

7. Evaluate during and after training.
   - Add or use periodic eval every 50-100 optimizer steps on internal eval.
   - Track exact answer accuracy, invalid-output rate, task-type accuracy,
     repair distance tolerance, candidate/ranking metrics, and reward mean.
   - Compare against the existing answer-only SFT baseline with the caveat that
     SFT used very short completions and may not be comparable on reasoning
     completions.

8. Persist artifacts.
   - Save adapters/checkpoints, metrics JSONL, train result JSON, eval outputs,
     release candidate reports, and command logs.
   - Upload final train artifacts to an HF bucket if credentials and bandwidth
     permit.
   - Update `docs/experiments/qwen3p5-4b-tuning.md` with the actual result,
     even if the result is a failed canary.

## Concrete command skeletons

Stabilize:

```bash
rtk uv run pytest -q tests/test_rewards.py tests/test_splits.py tests/test_metrics.py tests/test_prompts.py tests/test_config.py
rtk uv run python -m scripts.text_grpo_smoke --dataset-dir data/intersectionedit_repair_smoke --max-rows 8 --max-steps 1
```

Build IntersectionEdit release candidate:

```bash
rtk uv run python -m scripts.build_release_candidate \
  --config configs/repair_smoke.yaml \
  --output-dir data/intersectionedit_repair_rc
```

GPU canary target:

```bash
python scripts/text_grpo_train_unsloth.py \
  --dataset-dir /root/intersectionedit_repair_rc \
  --model unsloth/Qwen3.5-4B \
  --output-dir /root/outputs/grpo_qwen3p5_4b_intersectionedit_canary \
  --max-train-rows 128 \
  --max-eval-rows 64 \
  --max-steps 20 \
  --max-prompt-length 2048 \
  --max-completion-length 512 \
  --num-generations 4 \
  --eval-steps 10 \
  --save-steps 10
```

Pilot target if canary is healthy:

```bash
python scripts/text_grpo_train_unsloth.py \
  --dataset-dir /root/intersectionedit_repair_rc \
  --model unsloth/Qwen3.5-4B \
  --output-dir /root/outputs/grpo_qwen3p5_4b_intersectionqa_edit_pilot \
  --max-steps 300 \
  --max-prompt-length 2048 \
  --max-completion-length 1024 \
  --num-generations 4 \
  --eval-steps 50 \
  --save-steps 50 \
  --resume
```

Use GSPO for the serious run if the installed TRL/Unsloth stack exposes the
config field:

```python
importance_sampling_level = "sequence"
```

Use Dr. GRPO only as the second variant after a normal GRPO/GSPO canary exists:

```python
loss_type = "dr_grpo"
```

## Stop rules

- If release candidate validation fails: fix dataset/release first; do not
  train on a suspect dataset.
- If live Vast pricing cannot keep the run under `$7`: do not rent the GPU;
  leave the exact launch command and local smoke status instead.
- If local container prebuild is viable, do it before GPU rental; if it is not
  ready quickly, skip it rather than burning planning time.
- If using A100 80GB: reserve about 30-45 minutes for setup/upload/eval and
  cap training extension around 5 hours of actual GPU runtime.
- If using H100: cap training extension around 3-4 hours unless live price is
  low enough to stay below `$7`.
- If canary OOMs: reduce `max_completion_length` to 256, then
  `num_generations` to 2, then train rows/batch.
- If invalid-output rate stays high after 50-100 steps: fix prompt/template and
  reward formatting instead of extending training.
- If reward mean is flat and exact accuracy is zero by 300 steps: preserve logs,
  stop extension, and run the smaller SFT fallback only if time remains.
- If HF upload fails: keep local tarball/checksums and write the exact failed
  command plus error into the experiment doc.

## To Do

- [x] Run focused tests for changed reward/split/prompt/config code.
- [x] Fix any failing tests or obvious reward/parser bugs.
- [x] Commit the stabilized code baseline.
- [x] Decide whether to prebuild/push a training container locally.
- [x] Build and validate IntersectionEdit release candidate.
- [x] Inspect split/task/answer balance reports.
- [x] Add or confirm internal SFT/RL train/eval split usage in training code.
- [x] Add production GRPO/Unsloth runner if missing.
- [x] Run local GRPO smoke.
- [x] Launch GPU GRPO canary.
- [x] Launch 300-step pilot if canary is healthy. Stop decision: canary was not healthy enough to extend.
- [x] Run held-out/internal eval and compare with SFT baseline. Stop decision: GRPO canary is not comparable to the answer-only SFT baseline yet.
- [x] Upload validated dataset/artifacts if possible. Preserved local artifacts;
  HF bucket upload was not attempted because no bucket target was specified.
- [x] Update `docs/experiments/qwen3p5-4b-tuning.md`.
- [x] Commit final docs/code state.

## Done

- [x] Narrowed the overnight objective to validated IntersectionEdit data plus
  one measurable Qwen3.5 4B GRPO/GSPO pilot.
- [x] Deferred broad SFT/model-size/RL-algorithm comparisons unless the main
  path is blocked.
- [x] Added reasoning-compatible reward parsing for
  `<think>...</think><answer>...</answer>` completions while keeping canonical
  answer-only evaluation support.
- [x] Added `scripts/text_grpo_train_unsloth.py` with GSPO/Dr-GRPO flags,
  periodic generation-quality eval, JSONL metric logging, and adapter saving.
- [x] Local focused tests passed:
  `rtk uv run python -m compileall -q intersectionqa scripts && rtk uv run pytest -q tests/test_rewards.py tests/test_splits.py tests/test_metrics.py tests/test_prompts.py tests/test_config.py`.
- [x] Committed stabilized GRPO prep baseline as
  `Add IntersectionEdit GRPO training prep`.
- [x] Skipped local container prebuild because the repo does not have a ready
  CUDA training image and dependency churn would likely cost more time than the
  paid install path on a Vast A100.
- [x] Confirmed `configs/repair_smoke.yaml` was only a tiny smoke path:
  `geometry_limit: 100`. Exact `axis_aligned_repair` materialization on raw
  CADEvolve rows was too slow for the overnight budget, so the training pilot
  uses the fast mixed QA+Edit subset instead of the exact repair/candidate
  family.
- [x] Built `data/intersectionedit_grpo_pilot` from
  `configs/intersectionedit_grpo_pilot.yaml`: 500 geometries, 2,565 public task
  rows, leakage audit pass, and successful dataset validation.
- [x] Prepared mixed QA+Edit RL internal splits in
  `data/intersectionedit_grpo_pilot_inner_all` using only public `train` rows:
  1,627 `inner_train` rows, 200 `inner_eval` rows, 238 group-safe internal
  groups, and task coverage across binary/relation/volume/clearance/tolerance
  plus repair direction/translation and target clearance/contact/centroid
  movement.
- [x] Added `--scope all` to
  `scripts.prepare_intersectionedit_training_splits` so GRPO can train on mixed
  IntersectionQA + IntersectionEdit rows while preserving edit counterfactual
  groups and QA split groups.
- [x] Re-ran focused validation after the mixed-scope split change:
  `rtk uv run python -m compileall -q intersectionqa scripts && rtk uv run pytest -q tests/test_rewards.py tests/test_splits.py tests/test_metrics.py tests/test_prompts.py tests/test_config.py`
  passed with 43 tests.
- [x] Local GRPO execution smoke is intentionally deferred to the GPU canary:
  this laptop has no GPU and the local project environment does not install
  `torch`, `transformers`, `datasets`, `trl`, or `unsloth`.
- [x] Investigated the exact repair slowdown instead of treating it as
  inherent. The original path amplified CAD-kernel work by running full
  `measure_shape_pair`/Boolean checks inside every binary-search predicate.
  The repair materializer now caches placed shape/bbox/volume state, uses
  bbox/distance predicates during one-decimal label search, and reserves exact
  Boolean label derivation for final candidate metadata. A 10-geometry local
  CADEvolve probe with 4 positive-overlap rows dropped from not finishing after
  roughly 90 seconds to about 33.6 seconds for 4 exact repair rows.
- [x] Rented Vast contract `35599309`, an `NVIDIA A100-SXM4-80GB` instance at
  about `$1.10/hr`, and installed the Unsloth/TRL stack remotely. The working
  environment required `LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cu13/lib`
  after Unsloth pulled a Torch 2.11/CUDA 13 stack.
- [x] Ran the first 20-step GRPO canary attempt with
  `max_completion_length=512`, `num_generations=4`; it reached step 10 but the
  quality callback crashed because the Qwen processor tried image processing
  during text generation. Fixed by using the underlying text tokenizer in the
  quality callback.
- [x] Ran a faster canary from the same mixed split with
  `max_completion_length=192`, `num_generations=2`, 128 train rows, and 64 eval
  rows. Step 10 produced reward mean `0.08925`, eval reward `0.08164`, and a
  saved checkpoint at
  `/root/outputs/grpo_qwen3p5_4b_intersectionqa_edit_canary_fast/checkpoint-10`.
  Step time was still roughly 30-50 seconds, while the 32-row quality eval cost
  about 10 minutes.
- [x] Found a second evaluation bug: rewards accepted
  `<think>...</think><answer>...</answer>`, but `evaluate_predictions` parsed
  the raw generation string. Added shared answer-tag canonicalization so
  training quality metrics and offline evaluation use the same answer candidate
  as the reward path. A corrected 8-row resume quality probe at step 11 showed
  reward mean `0.2125`; tolerance-fit rows were `2/2` correct with zero invalid
  outputs, while clearance/relation/repair-translation rows remained invalid.
- [x] Found a GRPO performance regression: the trainer was still receiving the
  processor object instead of the underlying text tokenizer, causing processor
  warnings and avoidable multimodal preprocessing overhead. The production GRPO
  runner now passes the text tokenizer to `GRPOTrainer` and saves that tokenizer
  with the adapter.
- [x] Did not launch the 300-step pilot. The stop rule triggered because the
  canary still had high invalid-output rates outside tolerance-fit rows,
  clipped completions, and expensive quality evals. The next run should first
  tighten the prompt/format reward or bootstrap short reasoning traces, then
  rerun a smaller canary before spending on a long pilot.
- [x] Pulled canary artifacts into
  `data/training_artifacts/grpo_qwen3p5_4b_intersectionqa_edit_canary_fast/`,
  including train/quality JSONL logs, remote command logs, and the step-10 LoRA
  adapter checkpoint files. Destroyed Vast instance `35599309` after artifact
  retrieval.
- [x] Added small format-scaffold GRPO rewards for tagged answers, tightened
  the GRPO system prompt, reduced default quality-eval cost, and logged sample
  generations in `quality_metrics.jsonl`.
- [x] Fixed a cleaner-stack TRL/transformers compatibility issue by ensuring
  the Unsloth-wrapped Qwen3.5 model exposes `warnings_issued` before
  `GRPOTrainer` initialization.
- [x] Rented a second A100 contract `35601616` for the format canary. The
  initial clean Torch `2.5.1` stack produced zero trainable parameters; upgrading
  Unsloth installed Torch `2.10.0+cu128`, transformers `5.5.0`, TRL `0.23.0`,
  and trained `38,756,352` LoRA parameters.
- [x] Ran a 10-step format canary on 128 train rows and 32 eval rows. Step 10
  quality over 8 rows improved to reward mean `0.3491` with `0.0`
  invalid-output rate, but correct answers were still limited to tolerance-fit
  and one relation row.
- [x] Ran a bounded 50-step continuation from checkpoint 10. Step 25 train
  reward peaked at `0.4942`, but 16-row quality was `0.3319`; step 50 quality
  fell to `0.3116`. Tolerance-fit stayed `3/3` and relation stayed `1/2`, while
  binary interference, movement, bucket, volume, and repair rows remained wrong.
  Stop decision remains: do not launch a 300-step pilot from this checkpoint.
- [x] Pulled the format-pilot artifacts into
  `data/training_artifacts/grpo_qwen3p5_4b_intersectionqa_edit_format_pilot50/`,
  including logs, metrics JSONL files, `checkpoint-50`, final adapter, and the
  compressed remote artifact bundle.
- [x] Destroyed Vast instance `35601616`; `vastai show instances --raw` returned
  `[]`.
- [x] Fixed the next canary's data-starvation issue before renting more GPU:
  capped GRPO row loading now defaults to task-stratified sampling. For the
  existing 128-row cap this changes train coverage from only 7
  `repair_translation`, 5 `target_clearance_move`, and 3 `repair_direction`
  rows to roughly 12-13 rows per task family.
- [x] Updated GRPO quality logging to keep diverse low-reward samples instead
  of the first few eval rows, and raised the default sample log count to 16 so
  the next run exposes repair/movement/bucket failures directly.
- [x] Rented a third A100 contract `35603017` for the stratified capped-row
  canary. The working stack was Torch `2.10.0+cu128`, transformers `5.5.0`,
  TRL `0.24.0`, Unsloth `2026.4.8`, and the run trained `38,756,352` LoRA
  parameters on Qwen3.5 4B.
- [x] Ran the 20-step stratified canary with 128 train rows and 32 eval rows.
  The train cap covered every task family with 12-13 rows each and the eval cap
  covered every family with 3-4 rows each. Step 20 quality over 16 rows had
  reward mean `0.3911` with `0.0` invalid-output rate.
- [x] Ran a bounded 50-step continuation from `checkpoint-20`. Final internal
  eval reward was `0.3191` at step 50, down from `0.3475` at step 40, and the
  step-30 quality probe was only `0.4031`. The final quality callback did not
  log a step-50 record because the resumed trainer kept mismatched eval/save
  cadence, so the reliable final metric is the internal eval reward.
- [x] Patched the GRPO runner to always write a final quality record after
  `trainer.train()` when quality eval is enabled, unless the periodic callback
  already logged that same final step.
- [x] Preserved failure-focused samples from the stratified run. They show the
  same competence gap as before: repair direction predicts `-z` for `+x`/`+z`,
  repair translation predicts `+y 34.000000` for `+z 120.000100`, target
  clearance predicts `distance_mm=-199.5` for `distance_mm=0.5`, centroid move
  predicts negative distances for `distance_mm=5.0`, and volume bucket predicts
  `(0.01, 0.05]` for answer `0`.
- [x] Pulled the stratified-pilot artifacts into
  `data/training_artifacts/grpo_qwen3p5_4b_intersectionqa_edit_stratified_pilot50/`,
  including logs, metrics JSONL files, `train_result.json`, `checkpoint-50`,
  final adapter, and the compressed remote artifact bundle.
- [x] Destroyed Vast instance `35603017`; `vastai show instances --raw` returned
  `[]`.
- [x] Stop decision remains: do not launch the 300-step GRPO pilot from these
  checkpoints. Stratified sampling fixed data coverage and validity, but not
  repair/movement/bucket learning. The next serious attempt should change the
  learning signal before spending on a long RL run.
- [x] Added hierarchical task-then-answer stratified row caps for future GRPO
  canaries. The same 128-row cap now keeps task coverage while spreading
  binary, clearance, relation, tolerance, volume, repair-direction, repair-
  translation, and movement-distance answers across available labels/magnitudes.
  This also exposed a source-data limitation: the current fast pilot has no
  negative repair-direction labels, so that balance issue cannot be fixed by
  sampling alone.
- [x] Fixed the source-side repair-direction imbalance. CADEvolve bbox-guided
  gap placement no longer always puts `object_b` on the positive-x side; it now
  cycles deterministic pair placements across `+x`, `-x`, `+y`, `-y`, `+z`,
  and `-z`, stores the placement direction in metadata, and treats axis-specific
  translation variants as counterfactual groups. A 200-geometry repair-only
  probe produced all six repair directions.
- [x] Rechecked the broad fast pilot path and confirmed row materialization is
  still the slow stage when target movement tasks are included. This is local
  CPU time, not GPU spend, but it remains a practical blocker for quickly
  rebuilding larger mixed datasets.
- [x] Fixed the main local performance regressions found while rebuilding the
  balanced pilot. Repair/movement row materialization now caches placed
  shape/bbox/volume context, reuses stored geometry AABBs/volumes, and uses an
  AABB-disjoint certificate for centroid-distance non-intersection. A 120-record
  edit-row probe dropped from about `69s` to `34.8s`.
- [x] Fixed release-report replay overhead. Tool-assisted label-derived metrics
  now use stored exact labels, OBB baseline caches local object boxes by
  source-code hash, and conservative repair verification accepts moved-AABB
  disjointness as a proof before falling back to CadQuery.
- [x] Built a fresh balanced IntersectionEdit/QA release candidate at
  `data/intersectionedit_grpo_pilot_balanced`: 500 CADEvolve geometries, 2,821
  rows, leakage `pass`, exact stored-repair verification `338/338`, and
  repair-direction verifier success `169/169` with all six directions present.
- [x] Rebuilt internal RL splits from public `train` only at
  `data/intersectionedit_grpo_pilot_balanced_inner_all`, with group-safe
  task/answer balancing for low-cardinality labels. The split has 1,956
  `inner_train` rows and 217 `inner_eval` rows; inner eval now includes every
  repair direction, including `-y`.
- [x] Local GRPO smoke remains blocked on this laptop because the local
  environment has no `torch`; GPU smoke was used instead.
- [x] Rented A100 contract `35606811` for the balanced canary. Working stack:
  Torch `2.10.0+cu128`, transformers `5.5.0`, TRL `0.24.0`, Unsloth
  `2026.4.8`.
- [x] Ran a 20-step balanced-data GRPO canary on 128 train rows and 64 eval
  rows with task-then-answer stratified caps. Step 20 internal eval reward was
  `0.2646`, and final failure-focused quality reward was `0.3678`, down from
  `0.3791` at step 10. Repair direction, repair translation, centroid
  movement, target clearance, and target contact remained `0.0` exact in the
  quality sample, so the 300-step pilot is still stopped.
- [x] Pulled balanced-canary artifacts into
  `data/training_artifacts/grpo_qwen3p5_4b_intersectionqa_edit_balanced_canary20/`,
  including remote logs, metrics JSONL files, `train_result.json`, checkpoints,
  adapter files, and the compressed remote artifact bundle.
- [x] Destroyed Vast instance `35606811`; `vastai show instances --raw`
  returned `[]`.
