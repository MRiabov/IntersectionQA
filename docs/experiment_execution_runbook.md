# Experiment Execution Runbook

This runbook describes how to run the paper experiment suite without turning GPU
time into a sequence of manual, one-off commands. The goal is to write the
experiment harness once, dry-run it cheaply, and then run staged GPU jobs with
clear monitoring and stop rules.

The experiment definitions live in `specs/research-experiment-spec.md`. This
document is operational guidance for executing them.

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
   A script such as `scripts/run_experiment_suite.py` that executes the manifest
   in dependency order.
3. **Run-specific training/evaluation scripts**
   Existing scripts such as SFT, GRPO, zero-shot evaluation, baseline tables,
   and failure analysis.

The orchestration layer should call existing focused scripts rather than
duplicating their internals.

## Experiment Manifest

The manifest should describe every run in a structured way.

Recommended fields:

- `name`: stable run identifier.
- `kind`: `dataset_report`, `baseline`, `zero_shot`, `sft`,
  `reasoning_sft`, `grpo`, `eval`, or `analysis`.
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
- heuristic/tool baselines;
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

Run the main answer-only SFT and reasoning-SFT jobs. Prefer these over RL if the
GPU budget is tight.

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

## Artifact Layout

Each run should write to a stable directory:

```text
outputs/<suite>/<run_name>/
  run_manifest.json
  status.json
  command.log
  environment.json
  train_metrics.jsonl
  quality_metrics.jsonl
  predictions/
  reports/
  checkpoints/
  adapter/
  adapter-best/ or checkpoint-best/
```

Evaluation-only runs may omit training files, but should still write
`run_manifest.json`, `status.json`, predictions, and reports.

## Upload And Preservation

At the end of a GPU session:

1. Compress run artifacts.
2. Upload adapters, checkpoints, logs, metrics, and predictions to durable
   storage.
3. Record artifact URLs and checksums in the experiment doc.
4. Download or mirror critical logs locally.
5. Destroy rented instances after verifying uploads.

Do not rely on a rented GPU disk as the only copy of results.

## Practical Default Plan

Default order for the paper suite:

1. Implement manifest and orchestration script.
2. Run local smoke tests.
3. Run dataset reports and baselines.
4. Run zero-shot evaluation or request export.
5. Run answer-only SFT canary.
6. Run reasoning-SFT canary.
7. Run full supervised jobs.
8. Evaluate all held-out splits.
9. Run counterfactual and failure analysis.
10. Run one bounded RL canary.
11. Run one main RL job only if the canary is healthy.
12. Upload artifacts and update experiment docs.

The high-level rule is simple: automate the path, monitor the expensive gates,
and spend full GPU budget only on runs that have passed cheap health checks.

