# Experiment Execution Runbook

This runbook describes how to run the paper experiment suite without turning GPU
time into a sequence of manual, one-off commands. The goal is to write the
experiment harness once, dry-run it cheaply, and then run staged GPU jobs with
clear monitoring and stop rules.

The experiment definitions live in `specs/research-experiment-spec.md`. This
document is operational guidance for executing them.

Related operational notes:

- `docs/text_finetune_runbook.md` tracks the current text-only dataset state,
  local training scripts, remote setup template, and SFT/GRPO staging commands.
- `docs/qwen3p5-4b-tuning.md` tracks reusable Qwen3.5 4B tuning guidance,
  runtime stack notes, GRPO guidance, reasoning-preservation guidance, and known
  model/runtime failure modes.
- `docs/experiments/` contains dated experiment records with exact commands,
  hardware, outcomes, and artifact locations.
- `docs/old_overnight_tasks.md/` contains archived task logs and should be used
  only for historical context.

This runbook does not replace those files. It describes how to orchestrate the
full paper experiment suite across those lower-level scripts and runbooks.

## Core Principle

Do not manually launch every experiment one by one on a rented GPU. Build a
restartable experiment harness, smoke-test it locally, then run the expensive
parts in staged GPU gates.

Manual monitoring is still valuable, especially for first GPU canaries and RL
runs, but monitoring should not be a substitute for reproducible scripts,
continuous metrics, checkpointing, and stop conditions.

## Recommended Architecture

Use three layers:

1. **Experiment manifest**
   A YAML or JSON file that lists the intended runs and their dependencies.
2. **Orchestration script**
   A script such as `scripts/experiments/run_experiment_suite.py` that executes the manifest
   in dependency order.
3. **Run-specific training/evaluation scripts**
   Existing scripts such as SFT, GRPO, zero-shot evaluation, baseline tables,
   and failure analysis.

The orchestration layer should call existing focused scripts rather than
duplicating their internals.

## Experiment Manifest

The manifest should describe every run in a structured way.

The manifest is an execution slice through the experiment matrix in
`specs/research-experiment-spec.md`; it is not expected to contain every P0/P1/P2
matrix cell at once. A default-budget manifest should cover the P0 cells needed
for the current paper claim, while P1/P2 cells should be added only after their
prerequisites and budget gates are satisfied.

Recommended fields:

- `name`: stable run identifier.
- `kind`: `dataset_report`, `baseline`, `zero_shot`, `sft`,
  `reasoning_sft`, `tool_assisted`, `tool_use_baseline`, `grpo`, `eval`, or
  `analysis`.
- `depends_on`: run names that must complete first.
- `dataset_dir`: dataset or internal split directory.
- `model`: base model or adapter model.
- `adapter_init_dir`: optional adapter initialization path.
- `train_splits`: train split names.
- `eval_splits`: eval split names.
- `task_types`: included tasks.
- `prompt_mode`: public prompt, reasoning prompt, `edit_geometry`, or
  `edit_geometry_with_candidates`.
- `decoding_mode`: `schema_constrained` or `unconstrained_diagnostic`.
- `row_caps`: train/eval/quality row caps.
- `training`: epochs, max steps, batch size, learning rate, context length,
  completion length, generation count.
- `budget`: max wall-clock time, max GPU hours, and early-stop thresholds.
- `output_dir`: durable run output directory.
- `artifacts`: expected files that mark the run complete.

Every manifest should include small canary runs and full runs separately. A full
run should depend on its canary.

## Orchestration Script Requirements

The orchestration script should:

- validate the dataset and split manifests before training;
- run leakage and answer-balance audits before optimizer updates;
- execute runs in dependency order;
- skip completed runs when all expected artifacts exist;
- write a per-run `run_manifest.json`;
- capture command stdout/stderr to log files;
- stream metrics to JSONL files;
- write a final `status.json` with `success`, `failed`, `skipped`, or
  `stopped`;
- fail fast on missing inputs, invalid output files, NaN losses, repeated OOMs,
  or impossible metrics;
- preserve partial artifacts on failure;
- copy the best checkpoint or adapter to `checkpoint-best` or `adapter-best`
  when the run records a best validation step;
- generate comparison tables and failure-analysis reports after evaluation.

The script should be restartable. Re-running the same manifest should continue
from completed or resumable runs instead of overwriting them blindly.

### Selection Controls

The orchestrator supports targeted local execution so long paper suites can be
run in stages on Vast or resumed by an overseeing agent:

- `--run NAME` runs one named manifest entry; repeat it to run a named set.
- `--with-dependencies` expands `--run` to include transitive dependencies.
- `--start-from NAME` runs the dependency-order suffix beginning at `NAME`.
- `--run-until NAME` runs the dependency-order prefix ending at `NAME`.
- `--start-from NAME --run-until OTHER` runs a closed dependency-order window.
- `--skip-dependencies` bypasses dependency checks for manual recovery only.
- `--list` and `--dry-run` show the selected plan without executing commands.

Unselected dependencies are considered satisfied when their run directory already
has expected artifacts or a `status.json` marked `success`/`skipped`. This is
what makes one-by-one execution predictable after earlier stages finish.

Manifest commands may reference run paths with placeholders. The most useful
forms are `{run_dir}` for the current run directory and `{run:NAME}` for another
run's output directory, for example `{run:reasoning_sft_full}/adapter`.

## Local Preparation Before Renting GPU

Before renting GPU time:

1. Run focused unit tests for touched code.
2. Run dataset validation and leakage audits.
3. Run a tiny CPU/local smoke suite where possible.
4. Export or package the exact dataset directories needed on GPU.
5. Confirm model/download/cache requirements.
6. Confirm available disk size for model caches, checkpoints, and artifacts.
7. Prepare upload/download paths for final artifacts.

Tiny smoke defaults:

- 8-32 rows;
- 1-5 optimizer steps;
- one train split and one eval split;
- one or two task types;
- smallest viable model if the full model cannot run locally.

The smoke suite should test the actual command path, output files, metrics
format, and resume behavior. It does not need meaningful quality.

## GPU Staging

Run expensive work in gates.

### Gate A: Eval And Dataset Reports

Run:

- dataset statistics;
- leakage audit;
- answer-balance audit;
- heuristic baselines;
- exact CAD-kernel oracle / scripted verifier upper bound, with tool failure
  counts and failure reasons;
- standard instruct-model-plus-tools baseline when the tool-use harness is
  available, or a bounded subset if full tool-use evaluation is too expensive;
- zero-shot request export or small zero-shot evaluation.

Stop if validation or leakage fails.

### Gate B: Supervised Canary

Run answer-only SFT or reasoning-SFT on a small capped subset.

Suggested cap:

- 128-512 train rows;
- 32-128 eval rows;
- 10-50 optimizer steps.

Continue only if:

- loss is finite;
- generation does not collapse;
- invalid output rate is acceptable;
- checkpoint and eval artifacts are written.

### Gate C: Full Supervised Runs

Run the main answer-only SFT, deterministic reasoning-SFT, and
rejection-sampled reasoning-SFT jobs. Prefer these over RL if the GPU budget is
tight.

Use rejection-sampled reasoning SFT before any serious 1,500-2,000 step GRPO
run. Accepted traces should be parse-valid, answer-correct, bounded in
reasoning length, and free of obvious filler or repetition. Preserve acceptance
rates and rejection reasons as artifacts.

Required after each run:

- evaluate all official splits;
- write prediction JSONL;
- build comparison tables;
- run failure-case analysis;
- select and preserve the best checkpoint.

### Gate D: Bounded RL Canary

Run GRPO/GSPO only after supervised baselines are working.

Use schema-constrained final-answer generation by default when available. Run
one small unconstrained rollout canary only to estimate parse-failure or
policy-collapse risk; do not duplicate every full RL experiment in unconstrained
mode.

Continue only if:

- reward is non-degenerate;
- exact quality metrics are not obviously regressing;
- invalid output rate is controlled;
- completion lengths are sane;
- sampled rollouts and reward components are logged.

### Gate E: Main RL Run

Run one bounded main RL job from the best supervised checkpoint only if Gate D is
healthy.

Stop or pause if:

- reward improves but held-out exact quality falls repeatedly;
- completions collapse to very short common answers;
- reasoning becomes filler;
- KL or equivalent policy-distance metric spikes;
- numeric edit MAE remains unchanged after the canary window;
- projected GPU spend exceeds budget.

The default-budget main RL condition should be SFT+GRPO/GSPO, initialized from
the best deterministic or rejection-sampled reasoning-SFT adapter. Base-model
GRPO is useful as an ablation or diagnostic, but should not replace SFT+GRPO for
the primary training claim.

## Default Training Lengths

Use these starting lengths for the Qwen3.5 4B stack. Adjust only when measured
throughput, quality metrics, or stop rules justify it.

| Run | Default length |
| --- | --- |
| Local smoke | 1-5 optimizer steps on 8-32 rows |
| Answer-only SFT canary | 20-50 optimizer steps on 128-512 train rows |
| Answer-only full SFT | 1 epoch over public `train`; approximately 1,365 steps for 43,664 rows at effective batch 32 |
| Reasoning-SFT canary | 20-50 optimizer steps |
| Deterministic reasoning-SFT | 500-1,500 optimizer steps, or 1 epoch over the trace set |
| Rejection-sampled reasoning-SFT | 500-1,500 optimizer steps, or 1 epoch over accepted traces |
| Data-scaling SFT | 1 epoch each at 1k, 5k, 15k, 30k, and full train |
| GRPO/GSPO canary | 50-100 optimizer steps |
| Main SFT+GRPO/GSPO | 1,500-2,000 optimizer steps after a healthy canary |

Historical H100 GRPO throughput was about 9 seconds per optimizer step under one
candidate-feature configuration. That makes 1,500-2,000 GRPO steps roughly
3.75-5 raw GPU hours before eval, checkpoint, and upload overhead. Record actual
step time for every run; do not assume the historical number applies after
changing prompt length, completion length, `num_generations`, vLLM/generation
batching, or feature exposure.

Default completion-length policy:

- use 256 completion tokens for efficient main closed-book evaluation and
  bounded GRPO;
- run a 256 versus 512 token diagnostic on a fixed validation subset for
  reasoning-SFT and GRPO checkpoints;
- use 1,024 completion tokens only on a small subset unless the 512-token
  diagnostic materially improves quality.

## Monitoring Guidance

Manual monitoring is most useful during:

- environment setup on a new GPU image;
- first 5-20 optimizer steps of each new training recipe;
- first quality-eval callback;
- first checkpoint/resume cycle;
- RL runs where reward can diverge from actual task accuracy.

Monitor:

- GPU memory and utilization;
- disk usage;
- examples/sec or tokens/sec;
- training loss;
- reward mean and reward standard deviation;
- exact quality accuracy;
- invalid output rate;
- completion length;
- truncation rate;
- task-wise quality metrics;
- representative sampled completions.

If asking an agent to monitor GPU execution, provide:

- SSH details or existing session context;
- expected run directory;
- budget limit;
- stop conditions;
- whether the agent may patch code and resume;
- artifact upload target.

The agent should be allowed to stop unhealthy runs rather than spending the full
budget just because a command is still running.

## Stop Rules

Use explicit stop rules before launching expensive jobs.

Hard stops:

- dataset validation fails;
- leakage audit fails;
- training loss becomes NaN;
- repeated OOM after reducing batch/completion settings;
- disk is close to full;
- artifacts cannot be written;
- run exceeds budget.

Soft stops:

- invalid output rate remains high after format fixes;
- RL reward improves while held-out exact accuracy falls;
- all task gains are concentrated in one easy task;
- repair or movement tasks remain at zero exact accuracy after a targeted
  canary;
- completions collapse to common labels such as `no`, `0`, or `disjoint`;
- reasoning length is achieved only through filler or repetition.

Soft stops should usually pause the run for inspection rather than destroy all
artifacts.

## Guided Generation Policy

For serious evaluation and training, use guided or constrained final-answer
generation when the backend supports it. Label these results as
`schema-constrained`.

Do not spend default-budget compute on full unconstrained duplicates of every
run. Instead:

- run unconstrained decoding on a fixed 256-1k row validation subset;
- run a short unconstrained RL rollout canary if using RL;
- escalate to full unconstrained evaluation only if the diagnostic shows a large
  discrepancy or the paper specifically needs that comparison.

This policy keeps compute focused on CAD competence rather than parse failures,
while still measuring whether schema guidance is hiding a formatting weakness.

## Run Artifact Contract

Use a self-contained directory per run. Prefer this layout for new orchestration
work:

```text
runs/<run_id>/
  run_manifest.json
  command.txt
  environment.json
  git_status.txt
  git_diff.patch
  train_metrics.jsonl
  eval_metrics.jsonl
  artifacts.json
  predictions/
  checkpoints/
```

For GRPO/RL runs, also write:

```text
runs/<run_id>/
  rollout_samples.jsonl
  reward_components.jsonl
  checkpoint-best/
```

SFT runs should also preserve the best checkpoint or adapter when it differs
from the final one. Use `checkpoint-best/`, `adapter-best/`, or an
`artifacts.json` entry that points to the selected directory.

Evaluation-only runs may omit training files, but should still write
`run_manifest.json`, `command.txt`, `environment.json`, `git_status.txt`,
`git_diff.patch`, `artifacts.json`, predictions, and eval metrics.

### Required Files

`run_manifest.json` is the highest-value file. It should capture:

- `run_id`;
- timestamp;
- model and adapter identifiers;
- dataset path;
- dataset manifest or hash when available;
- train and eval splits;
- task filters;
- prompt mode;
- decoding mode;
- random seeds;
- train/eval/quality row caps;
- hyperparameters;
- output paths;
- checkpoint-selection rule.

`command.txt` should store the exact command line used to launch the run.

`environment.json` should store:

- Python version;
- OS;
- CUDA version;
- GPU name and count;
- `torch`;
- `transformers`;
- `trl`;
- `unsloth`;
- `peft`;
- `bitsandbytes`;
- `vllm` if used;
- relevant environment variables.

`git_status.txt` should store `git status --short` or equivalent. Many runs
happen from dirty worktrees.

`git_diff.patch` should store the current diff for GPU runs. It may be large,
but it is worth preserving for expensive experiments because results often come
from code that is later edited.

`train_metrics.jsonl` should contain one JSON object per logging step. Include
loss, reward, KL, learning rate, gradient norm, prompt length, completion
length, and truncation status when applicable.

`eval_metrics.jsonl` should contain validation/test metrics by split and task.
If existing code writes `quality_metrics.jsonl`, either keep that file and copy
or alias it into `eval_metrics.jsonl`, or record the mapping in
`artifacts.json`.

`predictions/` should contain raw prediction JSONL files for final evaluations
and quality evaluations. Metrics without predictions are difficult to audit and
weak for paper-table reproduction.

`artifacts.json` should index produced artifacts:

- checkpoints;
- adapters;
- best checkpoint or adapter;
- prediction files;
- metric files;
- report files;
- uploaded Hugging Face repo, bucket, or path;
- tarball paths and checksums.

`checkpoints/` should preserve normal trainer checkpoints or adapters. Do not
rely only on the final checkpoint.

`checkpoint-best/` or `adapter-best/` is very important for GRPO/RL and useful
for SFT. Previous runs showed final checkpoints can be worse than earlier
checkpoints.

### RL-Specific Files

`rollout_samples.jsonl` should store capped sampled completions from train/eval
batches. Do not log every rollout forever. A reasonable default is 16-64 sampled
completions per quality step, plus representative low-reward and high-reward
examples.

`reward_components.jsonl` should store per-completion reward components for
shaped rewards. This is needed to tell whether a model learned the task or
gamed one reward component.

Prompt length, completion length, and truncation status do not need separate
files. Include them in `train_metrics.jsonl`, `rollout_samples.jsonl`, or
`reward_components.jsonl`.

### Optional Later Files

Do not block the first implementation on these:

- `status.jsonl`: useful for orchestration, but logs and metrics are enough
  initially.
- `gpu_metrics.csv`: useful for throughput tuning, but occasional
  `nvidia-smi` snapshots in logs are acceptable until utilization debugging
  matters.
- full logging of every GRPO rollout: too expensive and noisy. Use capped
  samples.
- top-level `quality_predictions.jsonl`: unnecessary if `predictions/` contains
  files such as `quality_step_<step>.jsonl` or `final_eval_<split>.jsonl`.

### Implementation Order

Start with a small shared run utility that creates `runs/<run_id>/` and writes
the one-time snapshots:

- `run_manifest.json`;
- `command.txt`;
- `environment.json`;
- `git_status.txt`;
- `git_diff.patch`;
- `artifacts.json`.

Then update evaluators and trainers so final evaluations write raw predictions
under `predictions/`.

Finally, add the GRPO/RL-specific files:

- `rollout_samples.jsonl`;
- `reward_components.jsonl`.

That last step is the only nontrivial RL-specific plumbing, and it is worth it
because previous RL failures were hard to diagnose without rollout and
reward-component logs.

## Upload And Preservation

At the end of a GPU session:

1. Compress run artifacts.
2. Upload adapters, checkpoints, logs, metrics, and predictions to a Hugging
   Face Xet-backed bucket when credentials and bandwidth are available. HF Xet
   buckets are the preferred target for ML artifacts because they are faster and
   more convenient for this workflow than standard S3-style storage.
3. If HF Xet upload is unavailable, keep a local tarball plus checksums and
   record the exact failed or deferred upload command.
4. Record artifact URLs, bucket paths, and checksums in the experiment doc and
   in `artifacts.json`.
5. Download or mirror critical logs locally.
6. Destroy rented instances after verifying uploads.

Do not rely on a rented GPU disk as the only copy of results.

## Practical Default Plan

Default order for the paper suite:

1. Implement manifest and orchestration script.
2. Run local smoke tests.
3. Run dataset reports, leakage audits, and answer-balance audits.
4. Run heuristic baselines and the exact CAD-kernel/tool-assisted upper bound.
5. Run zero-shot evaluation or request export.
6. Run the standard instruct-model-plus-tools baseline when available.
7. Run answer-only SFT canary.
8. Run reasoning-SFT canary.
9. Run full answer-only SFT.
10. Run deterministic reasoning-SFT.
11. Run rejection-sampled reasoning-SFT if a 1,500-2,000 step GRPO run is
    planned.
12. Run supervised data-scaling jobs for the best recipe.
13. Evaluate all held-out splits.
14. Run counterfactual and failure analysis.
15. Run one bounded SFT+GRPO canary.
16. Run one main SFT+GRPO job only if the canary is healthy.
17. Upload artifacts and update experiment docs.

The high-level rule is simple: automate the path, monitor the expensive gates,
and spend full GPU budget only on runs that have passed cheap health checks.
