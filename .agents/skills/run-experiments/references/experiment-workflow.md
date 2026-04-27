# IntersectionQA Experiment Workflow

Use this reference for GPU/Vast, orchestration, SFT, GRPO, monitoring, and
artifact-preservation tasks. Keep dated run observations in `docs/experiments/`;
keep reusable operational procedure here.

## Source Map

- Canonical experiment matrix and reporting: `specs/research-experiment-spec.md`.
- Orchestrator implementation: `scripts/experiments/run_experiment_suite.py`,
  `intersectionqa/experiment_runner.py`, and `intersectionqa/experiments.py`.
- Main manifest: `configs/overnight_experiment_suite.yaml`.
- Local orchestrator smoke: `configs/orchestration_smoke.yaml`.
- Current text dataset/training state: `docs/text_finetune_runbook.md`.
- Model/runtime notes and known Qwen failure modes:
  `docs/qwen3p5-4b-tuning.md`.
- Vast bootstrap: `scripts/devops/bootstrap_vast_instance.sh`.
- Dated outcomes and exact historical commands: `docs/experiments/`.

## Local Preflight

Run only the checks relevant to the code touched, but do not rent GPU before
basic local validation is clean.

```bash
rtk uv run python -m compileall -q intersectionqa scripts
rtk uv run pytest -q tests/test_rewards.py tests/test_metrics.py tests/test_training_sampling.py
rtk uv run python -m scripts.experiments.run_experiment_suite configs/orchestration_smoke.yaml --list
rtk uv run python -m scripts.experiments.run_experiment_suite configs/orchestration_smoke.yaml --dry-run
```

For dataset changes, also run the release/validation commands from `AGENTS.md`.
For training-data changes, audit leakage and answer balance before optimizer
updates:

```bash
rtk uv run python -m scripts.dataset.validate_dataset data/IntersectionQA-90K
rtk uv run python -m scripts.dataset.audit_answer_balance \
  --dataset-dir data/IntersectionQA-90K \
  --min-share 0.10 \
  --max-share 0.70 \
  --min-count 30
```

## Dataset And Split Rules

- Remote GPU jobs should download `MRiabov/IntersectionQA-90K` from Hugging
  Face into `data/IntersectionQA-90K`.
- Do not copy local `data/IntersectionQA-90K.tar.gz` to Vast for normal runs;
  it is slower and risks stale artifacts.
- Optimizer updates may use public `train` rows only.
- Internal SFT/RL splits must be group-safe. Use
  `scripts.training.prepare_intersectionedit_training_splits` for edit/repair
  work and `intersectionqa.splits.grouped.partition_internal_train_eval_rows`
  from package code when adding new split paths.
- Do not split counterfactual or edit-counterfactual groups across internal
  train/eval.

Download the public dataset outside the suite only when necessary:

```bash
python -m scripts.dataset.download_hf_dataset \
  --repo-id MRIabov/IntersectionQA-90K \
  --output-dir data/IntersectionQA-90K
```

## Manifest Execution

Prefer the manifest runner so runs are dependency-checked, restartable, and
artifact-indexed.

Useful controls:

- `--list` shows available runs.
- `--dry-run` prints the selected plan without executing it.
- `--run NAME` selects one run; repeat for multiple runs.
- `--with-dependencies` expands selected runs to their transitive dependencies.
- `--start-from NAME` and `--run-until NAME` select a dependency-order window.
- `--skip-dependencies` is for manual recovery only.

Examples:

```bash
rtk uv run python -m scripts.experiments.run_experiment_suite \
  configs/overnight_experiment_suite.yaml --run answer_sft_canary --with-dependencies --dry-run

rtk uv run python -m scripts.experiments.run_experiment_suite \
  configs/overnight_experiment_suite.yaml --run grpo_canary --with-dependencies
```

Manifest commands can reference `{run_dir}` for the current run directory and
`{run:NAME}` for another run's output directory.

## Vast.ai Instance Selection

For meaningful Qwen3.5 4B SFT/GRPO runs, use A100 80GB-class or H100-class
capacity. Small 16GB cards are only for smoke tests.

Selection policy:

1. Filter offers first: `gpu_name in [A100_SXM4,A100_PCIE]` for A100 runs,
   `num_gpus=1`, `gpu_ram>=70`, enough disk, direct SSH ports, CUDA-compatible
   image, and acceptable reliability.
2. Sort the filtered offers by `dph_total` or equivalent total hourly price
   ascending, not by raw GPU type or list order.
3. Prefer spot only when it is actually available near budget; otherwise use the
   cheapest matching on-demand offer.
4. After creation, immediately verify live GPU name, VRAM, disk, and total
   hourly price. Destroy and recreate before bootstrapping if an A100 is priced
   like an H100 or the GPU/VRAM is not what was requested.
5. Destroy failed, overpriced, or finished contracts after artifact upload is
   verified.

Typical teardown:

```bash
printf 'y\n' | rtk uv run vastai destroy instance <contract_id>
```

Keep SSH keys, tokens, account balance, and live credentials out of repo files.

## Vast Bootstrap

Use the PyTorch image's existing CUDA Python. Do not run `uv sync` and do not
create a second virtualenv unless the image is known broken.

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i <ssh_key> -p <ssh_port> root@<ssh_host>

curl -fsSL \
  https://raw.githubusercontent.com/MRiabov/IntersectionQA/main/scripts/devops/bootstrap_vast_instance.sh \
  -o /root/bootstrap_vast_instance.sh
chmod +x /root/bootstrap_vast_instance.sh
BRANCH=main /root/bootstrap_vast_instance.sh
```

The bootstrap script clones or updates the repo, installs project dependencies
into image Python, installs Unsloth, and prints Python/Torch/CUDA/GPU state.

Confirm CUDA after bootstrap:

```bash
cd /root/IntersectionQA
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

Stable historical stack for Qwen3.5 4B runs:

- Torch `2.10.0+cu128`
- transformers `5.5.0`
- TRL `0.24.0`
- Unsloth `2026.4.8`
- bf16 LoRA on A100/H100

If a fresh image has zero trainable LoRA parameters or processor-backed Qwen
warnings leak into text-only generation, compare against the latest dated
experiment record before continuing.

## Launch And Monitor

Run long jobs in `tmux` or `nohup`:

```bash
tmux new -d -s iqa 'cd /root/IntersectionQA && python -m scripts.experiments.run_experiment_suite configs/overnight_experiment_suite.yaml --run grpo_canary --with-dependencies > overnight.log 2>&1'
```

Monitor:

```bash
tail -n 160 <log_path>
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
df -h /root
python -m scripts.experiments.monitor_experiment_run <run_dir>
```

Watch the first 5-20 optimizer steps and first quality-eval callback for every
new recipe. Track loss, reward mean/std, exact quality, invalid-output rate,
completion length, truncation, GPU utilization, disk, and checkpoint writes.

Treat steady-state GPU compute or memory utilization below 75% as
underutilization; the target is 90%+ when the model, batch shape, generation
path, and memory budget allow it. Do not let an expensive run continue for hours
with low utilization unless there is a deliberate bottleneck being measured.

When utilization is low, tune the config before continuing. Candidate knobs:

- increase `--per-device-train-batch-size` until close to the VRAM limit;
- adjust `--gradient-accumulation-steps` to preserve effective batch size;
- increase `--generation-batch-size` or `--quality-generation-batch-size`;
- reduce overly frequent generation-heavy eval callbacks;
- use `--eval-strategy no` when the quality callback is the evaluation path;
- shorten prompts/completions only if token length, not batching, is the
  bottleneck;
- consider vLLM only as a separate canary after confirming wheel/CUDA
  compatibility and reserving conservative colocated memory.

If the run has not progressed far past the latest checkpoint, stop it, change
the config, and resume from that latest checkpoint. If it has progressed far
enough that discarding the post-checkpoint work would be material, wait until the
next checkpoint is written, then stop and resume from that checkpoint. Record the
underutilization observation, the changed knobs, and the resume checkpoint in
the experiment record.

## Experiment Records

Persist every meaningful experiment, or coherent set of closely related
experiments, in a dated file under `docs/experiments/`. Do this for failed
canaries, negative results, interrupted runs, dataset/report-only runs, and
successful training runs. The record is the durable research log; performance
tuning is only one secondary part of it.

Each experiment record should include:

- purpose, hypothesis, and decision being tested;
- dataset source, dataset revision or path, split names, internal split method,
  row caps, and leakage/balance status;
- model, adapter initialization, prompt mode, decoding mode, reward components,
  and feature-exposure mode;
- exact command or manifest entry, plus any config diff from the previous run;
- hardware, Vast contract class when applicable, runtime stack, disk size, and
  approximate cost;
- checkpoints used for resume and best-checkpoint selection rule;
- monitoring notes, including GPU compute/memory utilization, step time, OOMs,
  disk pressure, invalid-output rate, and stop-rule triggers;
- metrics and qualitative outcomes by split/task where available;
- failures and negative findings, including what not to repeat;
- artifact paths, upload locations, checksums, and local mirrors;
- follow-up decision: stop, rerun with changed config, promote config to
  manifest default, or run the next gate.

When tuning finds a materially better config, persist the exact config and the
measured reason it is better. If the setting should become a default, update the
manifest/config file and link the experiment record that justifies the change.

## SFT Defaults

Default model for current budget runs: `unsloth/Qwen3.5-4B`.

Answer-only SFT:

- Keep `--assistant-only-loss` enabled where supported.
- Use `--pack-tokenized` for the Qwen3.5 processor-backed path.
- Use `2048` context for budget runs unless measured throughput justifies more.
- Run one epoch over public `train` for the full IntersectionQA SFT baseline.

Reasoning SFT:

- Generate supervised completions as
  `<think>...geometry reasoning...</think><answer>...</answer>`.
- Use `target_text` datasets from `scripts.training.prepare_reasoning_sft_dataset`
  when present; keep public canonical `answer` unchanged.
- Prefer a medium trace set and validate generation quality before GRPO.

## GRPO/GSPO Defaults

GRPO/GSPO should start only after supervised baselines or a canary are healthy.

Starting canary shape:

- `128-512` train rows and `32-128` eval rows.
- `20-100` optimizer steps.
- `num_generations=4`; increase only after stability.
- `max_prompt_length=2048`.
- `max_completion_length=128-256` for canaries; use `512` only when the task
  needs it and throughput is acceptable.
- `--eval-strategy no` when using the quality callback as the only evaluation
  path.
- `--quality-eval-steps` at a deliberate cadence, usually `10-50` for canaries.
- `--quality-generation-batch-size 8` when VRAM permits.
- `--importance-sampling-level sequence` for GSPO-style sequence-level updates.
- `--loss-type dapo` for the currently working DAPO-style path.

Feature modes:

- `--prompt-feature-mode edit_geometry` exposes trusted training/eval geometry
  features for IntersectionEdit rows without changing public prompts.
- `--prompt-feature-mode edit_geometry_with_candidates` additionally exposes
  conservative signed-axis candidate moves for repair curriculum/debug canaries.
  Treat it as a diagnostic/curriculum mode, not the default public prompt.

Do not run a 1,500-2,000 step main RL job until a canary has non-degenerate
reward, controlled invalid-output rate, sane completion lengths, and held-out
quality that is not obviously regressing.

## Stop Rules

Hard stop:

- dataset validation or leakage audit fails;
- training loss becomes NaN;
- repeated OOM after reducing batch/completion settings;
- disk is close to full;
- expected artifacts cannot be written;
- live Vast price exceeds budget;
- run exceeds budget.

Soft stop or pause:

- invalid-output rate remains high after prompt/reward fixes;
- reward improves while held-out exact quality falls;
- all gains concentrate in one easy task;
- repair/movement exact accuracy stays at zero after targeted canaries;
- completions collapse to common labels such as `no`, `0`, or `disjoint`;
- reasoning length is filler or repetition.

## Artifact Contract

Every expensive run should preserve:

- `run_manifest.json`
- `command.txt`
- `environment.json`
- `git_status.txt`
- `git_diff.patch`
- `artifacts.json`
- stdout/stderr logs
- `train_metrics.jsonl`
- `eval_metrics.jsonl` or `quality_metrics.jsonl` with mapping in artifacts
- predictions under `predictions/`
- checkpoints or adapters
- `checkpoint-best/` or `adapter-best/` when best quality differs from final

For GRPO/RL, also preserve:

- `rollout_samples.jsonl`
- `reward_components.jsonl`
- capped sampled completions from train/eval batches
- prompt/completion lengths and truncation status
- group-relative ranks or selected winners when available

Do not rely only on `save_total_limit`; it can delete the best checkpoint.

## Upload And Teardown

Before destroying a rented instance:

1. Compress run artifacts or use the manifest upload target.
2. Upload adapters, checkpoints, logs, metrics, predictions, and tarball
   checksums to the configured HF bucket or fallback path.
3. Verify the uploaded artifact list or local mirror.
4. Record artifact URLs, bucket paths, checksums, live hardware, runtime stack,
   command, and outcome in the dated experiment file.
5. Destroy the Vast instance.

HF Xet-backed buckets are preferred for ML artifacts when credentials are
available. If upload fails, preserve a local tarball plus checksum and record
the failed or deferred upload command.

## Known Failure Modes

- 120GB disks are too small for multiple large Qwen caches. Clear failed model
  caches before switching large candidates.
- Generic PEFT QLoRA on 35B-class Qwen MoE OOMed during k-bit preparation on
  A100 80GB. Prefer Unsloth bf16 LoRA for MoE experiments.
- Qwen3.5 is processor-backed. For text-only training/evaluation, use the text
  tokenizer path in generation and trainer setup where the repo supports it.
- Plain single-field SFT without assistant-only masking trains mostly on prompt
  reconstruction.
- Random row caps can starve rare edit families. Use task/answer-stratified
  caps for GRPO canaries.
- Internal TRL eval plus quality callback can duplicate expensive generation.
  Pick one evaluation path per budget run.
