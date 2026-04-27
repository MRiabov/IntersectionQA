# Experiment Execution Runbook

Reusable operational guidance for running IntersectionQA and IntersectionEdit
experiments has moved to the repo-specific Codex skill:

- `.agents/skills/run-experiments/SKILL.md`
- `.agents/skills/run-experiments/references/experiment-workflow.md`

Use that skill for local preflight checks, experiment-manifest execution,
Vast.ai instance selection, remote bootstrap, SFT/GRPO launch, monitoring, stop
rules, artifact preservation, and teardown.

Keep this document as an index only so operational details do not drift across
multiple runbooks.

## Related Sources

- `specs/research-experiment-spec.md`: paper experiment matrix, budgets,
  required metrics, and reporting rules.
- `configs/overnight_experiment_suite.yaml`: main restartable experiment
  manifest.
- `configs/orchestration_smoke.yaml`: cheap local orchestrator smoke manifest.
- `scripts/experiments/run_experiment_suite.py`: manifest runner CLI.
- `scripts/devops/bootstrap_vast_instance.sh`: Vast PyTorch image bootstrap.
- `docs/text_finetune_runbook.md`: current text dataset and training state.
- `docs/qwen3p5-4b-tuning.md`: reusable Qwen3.5 4B model/runtime notes.
- `docs/experiments/`: dated records with exact historical commands, hardware,
  outcomes, failures, and artifact locations.
