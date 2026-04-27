---
name: run-experiments
description: Run IntersectionQA and IntersectionEdit experiments in this repository, including local preflight checks, dataset/config selection, experiment-suite orchestration, Vast.ai GPU instance selection, remote bootstrap, SFT/GRPO launch and monitoring, stop rules, artifact preservation, and experiment-record updates. Use when Codex is asked to prepare, launch, resume, debug, monitor, or document training/evaluation experiments for this repo.
---

# Run Experiments

## Core Rule

Run experiments through the repo's restartable scripts and manifests, not one-off
manual training commands, unless you are doing a bounded diagnostic. Keep local
commands behind `rtk`; on a Vast PyTorch image, use the image Python directly
after running `scripts/devops/bootstrap_vast_instance.sh`.

## First Files To Read

Read only what is relevant to the requested run:

- `specs/research-experiment-spec.md` for paper experiment scope, budgets,
  required reporting, and split hygiene.
- `configs/overnight_experiment_suite.yaml` for the current full-suite manifest.
- `configs/orchestration_smoke.yaml` for a cheap local orchestrator smoke.
- `docs/experiments/` for the most recent dated result on the same model/task.
- `references/experiment-workflow.md` when the task involves Vast.ai, GPU
  rental, launch commands, monitoring, stop rules, or artifact preservation.

## Workflow

1. Identify the experiment family: dataset report, baseline, zero-shot, SFT,
   reasoning-SFT, GRPO/GSPO, evaluation, or analysis.
2. Prefer an existing manifest entry or config. Add or edit configs before
   writing new ad hoc shell commands.
3. Run cheap local checks before renting GPU:
   `rtk uv run python -m compileall -q intersectionqa scripts`,
   focused tests for touched code, and an orchestrator `--dry-run` or tiny smoke.
4. Use public `train` only for optimizer updates. Derive SFT/RL inner splits
   with existing group-safe helpers; never train on `validation` or `test_*`.
5. For Vast runs, filter offers first, sort by total hourly price, verify the
   live GPU/VRAM/price after creation, then bootstrap with the repo script.
6. Launch long jobs in `tmux` or `nohup`, with explicit run directories,
   metrics JSONL, quality eval cadence, checkpoint save cadence, and stop rules.
7. Monitor early optimizer steps, quality samples, invalid-output rate, disk,
   GPU memory/utilization, and budget. If GPU compute or memory utilization is
   below 75% during steady training, treat the run as underutilized; ideally aim
   for 90%+. Stop unhealthy, overpriced, or materially underutilized runs and
   tune the config before resuming from a checkpoint.
8. Preserve artifacts before teardown: logs, metrics, predictions, adapters,
   checkpoints, best-checkpoint selection, command, environment, git state,
   checksums, and upload paths.
9. Persist every meaningful experiment or coherent experiment set in
   `docs/experiments/`, even when the result is a failed canary or a negative
   result. Performance and utilization tuning are only one subsection of the
   record.
10. Update the relevant dated file with purpose, hypothesis, dataset/splits,
    config/manifest, exact commands, hardware, outcomes, failures, artifact
    locations, and follow-up decisions.

## Command Patterns

List or dry-run the orchestrator:

```bash
rtk uv run python -m scripts.experiments.run_experiment_suite \
  configs/overnight_experiment_suite.yaml --list

rtk uv run python -m scripts.experiments.run_experiment_suite \
  configs/overnight_experiment_suite.yaml --run grpo_canary --with-dependencies --dry-run
```

Run a selected suite locally or remotely:

```bash
rtk uv run python -m scripts.experiments.run_experiment_suite \
  configs/orchestration_smoke.yaml --with-dependencies
```

On a bootstrapped Vast instance, omit `rtk` and use image Python:

```bash
cd /root/IntersectionQA
python -m scripts.experiments.run_experiment_suite \
  configs/overnight_experiment_suite.yaml --run grpo_canary --with-dependencies
```

## Escalation Rules

Pause before spending GPU budget when local validation fails, split leakage is
unknown, dataset artifacts are stale, Vast pricing is outside budget, expected
artifacts are not configured, or stop conditions are ambiguous.

Do not write secrets, HF tokens, private SSH keys, or live credentials into repo
files. Historical instance IDs can stay in dated experiment records; new
credentials should stay out of docs and logs.
