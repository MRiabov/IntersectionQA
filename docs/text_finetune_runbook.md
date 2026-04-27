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

The original 90K export had 90,000 final public rows. The current local
`data/IntersectionQA-90K` export has 58,155 rows after April 25, 2026
augmentation, split redistribution, and class-balancing passes:

- Near-boundary/counterfactual assembly groups were redistributed into the
  normal split mix instead of being routed wholesale to `test_near_boundary`.
  The deterministic target policy for boundary groups is 75% `train`, 10%
  `validation`, 5% `test_random`, 3% `test_object_pair_heldout`, and 7%
  `test_near_boundary`.
- `pairwise_interference` rows were rebuilt from exported relation rows and
  capped to balanced `A`, `B`, `both`, and `neither` counts within each split.
- Single-geometry public rows were downsampled by geometry group to target the
  v0.1 relation mix where all target classes exist: 40% `intersecting`, 30%
  `disjoint`, 15% `touching`, and 15% `near_miss`. Non-target relation classes
  such as `contained` are preserved, not forced into that target distribution.

Current local `data/IntersectionQA-90K` split counts:

- `train`: 43,664 rows, including 31,919 rows with
  `counterfactual_group_id`
- `validation`: 5,622 rows, including 4,232 rows with
  `counterfactual_group_id`
- `test_random`: 2,981 rows, including 2,271 rows with
  `counterfactual_group_id`
- `test_object_pair_heldout`: 2,080 rows, including 1,580 rows with
  `counterfactual_group_id`
- `test_near_boundary`: 3,808 rows, including 2,908 rows with
  `counterfactual_group_id`

Current train task counts:

- `binary_interference`
- `clearance_bucket`
- `pairwise_interference`
- `ranking_normalized_intersection`
- `relation_classification`
- `tolerance_fit`
- `volume_bucket`

Exact train counts are 7,535 `binary_interference`, 7,535
`clearance_bucket`, 4,288 `pairwise_interference`, 1,701
`ranking_normalized_intersection`, 7,535 `relation_classification`, 7,535
`tolerance_fit`, and 7,535 `volume_bucket`.

Current train `pairwise_interference` answer counts are 1,072 each for `A`,
`B`, `both`, and `neither`. Current `test_near_boundary` pairwise answer
counts are 87 each for `A`, `B`, `both`, and `neither`.

Current relation-classification counts after class balancing:

- `train`: 2,962 `intersecting`, 2,221 `disjoint`, 1,111 `touching`,
  1,110 `near_miss`, and 131 preserved `contained` rows.
- `validation`: 383 `intersecting`, 287 `disjoint`, 143 `near_miss`, 143
  `touching`, and 13 preserved `contained` rows.
- `test_random`: 207 `intersecting`, 155 `disjoint`, 78 `touching`, 77
  `near_miss`, and 7 preserved `contained` rows.
- `test_object_pair_heldout`: 137 `intersecting`, 103 `disjoint`, 52
  `touching`, 51 `near_miss`, and 12 preserved `contained` rows.
- `test_near_boundary`: 258 `intersecting`, 193 `disjoint`, 97 `touching`,
  96 `near_miss`, and 18 preserved `contained` rows.

The split redistribution is recorded in
`data/IntersectionQA-90K/split_redistribution_report.json`; the final
per-split balancing pass is recorded in
`data/IntersectionQA-90K/class_balance_report.json`.

Interpret the splits as a conventional supervised split plus named challenge
suites:

- `train`: training data.
- `validation`: dev validation.
- `test_random`: primary in-distribution held-out test.
- `test_object_pair_heldout`: strict object-pair/assembly generalization
  challenge.
- `test_near_boundary`: small boundary/counterfactual challenge suite, now
  about 6.5% of the local export after class balancing.
- `test_generator_heldout`: reserved generator-family challenge split; currently
  empty in the 90K export.

See `docs/dataset_split_framing.md` before interpreting evaluation numbers.

## Local Scripts

- `scripts/training/text_sft_train.py`: generic Transformers/PEFT LoRA or QLoRA runner.
  Use this for dense 4-bit QLoRA models.
- `scripts/training/text_sft_train_unsloth.py`: Unsloth bf16 LoRA runner. Use this for
  the current Qwen3.5 text-only SFT path and for any future MoE bf16 LoRA
  experiments. Its default `--task-types` now includes all seven public text
  tasks, including `pairwise_interference` and
  `ranking_normalized_intersection`.
- `scripts/training/evaluate_text_model.py`: exact-answer generation evaluator for a
  base or adapter model. It reports exact-answer accuracy by split/task plus
  per-task label precision, recall, F1, and confusion counts.
- `scripts/training/text_sft_smoke.py`: tiny local/remote SFT smoke test.
- `scripts/training/text_grpo_smoke.py`: tiny RL/GRPO smoke test.
- `scripts/training/prepare_intersectionedit_training_splits.py`: filters opt-in
  IntersectionEdit rows, applies `metadata.edit_counterfactual_group_id`-aware
  inner train/eval splitting, and writes `inner_train.jsonl`, `inner_eval.jsonl`,
  and `report.json` for SFT or GRPO staging.
- `scripts/dataset/add_counterfactual_train_rows.py`: legacy deterministic in-place
  augmentation helper for moving leakage-safe counterfactual groups from
  `test_near_boundary` to `train`. Prefer
  `scripts/dataset/redistribute_dataset_splits.py` for current exports. It creates
  `data/IntersectionQA-90K/pre_counterfactual_train_backup` before rewriting
  split files.
- `scripts/dataset/rebalance_pairwise_rows.py`: deterministic in-place export rewrite
  that removes old pairwise rows and rebuilds balanced pairwise comparisons
  from already exported relation-classification rows. It creates
  `data/IntersectionQA-90K/pre_pairwise_rebalance_backup` before rewriting
  split files.
- `scripts/dataset/redistribute_dataset_splits.py`: deterministic in-place export
  rewrite that applies the current group-safe split policy to already exported
  rows, refreshes JSONL, metadata, Parquet, dataset card files, writes
  `split_redistribution_report.json`, and optionally reruns class balancing.
  It creates `data/IntersectionQA-90K/pre_split_redistribution_backup` before
  rewriting split files.
- `scripts/dataset/balance_dataset_classes.py`: deterministic in-place export rewrite
  that applies the release class-balancing policy, refreshes JSONL, metadata,
  Parquet, dataset card files, and writes `class_balance_report.json`. It
  creates `data/IntersectionQA-90K/pre_class_balance_backup` before rewriting
  split files.
- `scripts/dataset/audit_answer_balance.py`: answer-distribution audit by split and
  task. Use it before publishing a dataset or starting a new training run:

```bash
rtk uv run python -m scripts.dataset.audit_answer_balance \
  --dataset-dir data/IntersectionQA-90K \
  --min-share 0.10 \
  --max-share 0.70 \
  --min-count 30
```

Current audit state: relation classes hit the v0.1 target in all non-empty
splits, and `pairwise_interference` is exactly balanced in all splits that
contain pairwise rows. Remaining distribution risks are sparse
geometry-derived bucket classes, especially `contained`, tiny clearance
buckets, nonzero normalized-volume buckets, and exact ranking permutations. Do
not blindly cap those to the rarest class without generating more targeted
source geometry first, because that would discard most of the useful dataset.
Future CADEvolve runs now include deterministic clearance-bucket and
overlap-magnitude candidate strategies to improve those scarce buckets before
the post-materialization capping pass.

## Experiments

Keep experiment-specific choices, rejected model paths, instance IDs, launch
commands, and run observations in `docs/experiments/` instead of expanding this
runbook.

- [Qwen3.5 4B tuning notes](qwen3p5-4b-tuning.md)
- [April 25, 2026 Qwen3.5 4B IntersectionQA SFT](experiments/apr-25-qwen3p5-4b-intersectionqa-sft.md)
- [April 25-26, 2026 Qwen3.5 4B IntersectionEdit GRPO](experiments/apr-25-26-qwen3p5-4b-intersectionedit-grpo.md)

## IntersectionEdit SFT/GRPO Staging

Build an opt-in edit dataset first:

```bash
rtk uv run python -m scripts.dataset.build_release_candidate \
  --config configs/repair_smoke.yaml \
  --output-dir data/intersectionedit_repair_smoke
```

Prepare group-safe inner splits for supervised fine-tuning:

```bash
rtk uv run python -m scripts.training.prepare_intersectionedit_training_splits \
  --dataset-dir data/intersectionedit_repair_smoke \
  --output-dir data/intersectionedit_repair_smoke_sft \
  --mode sft \
  --eval-fraction 0.10
```

Prepare the same rows for reward-learning experiments:

```bash
rtk uv run python -m scripts.training.prepare_intersectionedit_training_splits \
  --dataset-dir data/intersectionedit_repair_smoke \
  --output-dir data/intersectionedit_repair_smoke_grpo \
  --mode rl \
  --scope all \
  --eval-fraction 0.10
```

The split helper defaults to IntersectionEdit task families; pass `--scope all`
for mixed IntersectionQA + IntersectionEdit GRPO. It keeps edit counterfactual
groups intact and falls back to normal split groups for QA rows. `sft` mode
honors `edit_diagnostics.sft_include` when present; `rl` mode honors
`edit_diagnostics.rl_include`. The reward path is metadata-based through
`intersectionqa.evaluation.rewards`, covering exact
axis/distance repair, full-vector repair, edit-program repair, signed
clearance/contact/centroid movement, candidate selection, and candidate ranking.

Smoke GRPO on the prepared edit dataset:

```bash
rtk uv run python -m scripts.training.text_grpo_smoke \
  --dataset-dir data/intersectionedit_repair_smoke_grpo \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-rows 16 \
  --max-steps 1
```

## Remote Setup Template

Upload files:

```bash
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i <ssh_key> -P <ssh_port> \
  data/IntersectionQA-90K.tar.gz \
  scripts/training/text_sft_train.py \
  scripts/training/text_sft_train_unsloth.py \
  scripts/training/evaluate_text_model.py \
  scripts/training/text_sft_smoke.py \
  scripts/training/text_grpo_smoke.py \
  root@<ssh_host>:/root/
```

Prepare the remote box:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i <ssh_key> -p <ssh_port> root@<ssh_host>

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

Monitor remote training:

```bash
tail -n 160 <log_path>
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
df -h /root
```

## Known Failure Modes

- Qwen-specific model-selection, disk-cache, and hardware failures live in
  [Qwen3.5 4B tuning notes](qwen3p5-4b-tuning.md) and the dated experiment
  records. Keep this runbook focused on dataset and workflow procedure.
- Plain single-field SFT without `--assistant-only-loss` likely trains mostly on
  prompt reconstruction. Use the packed tokenized path, which masks labels
  before the assistant answer.
- Keep secrets out of logs and docs. The HF token was provided in chat but must
  not be written into repository files.
