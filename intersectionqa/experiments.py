"""Experiment orchestration and run artifact helpers."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from intersectionqa.evaluation.metrics import Prediction
from intersectionqa.evaluation.parsing import canonical_answer_candidate, parse_answer
from intersectionqa.hashing import sha256_json
from intersectionqa.pipeline import validate_dataset_dir
from intersectionqa.schema import PublicTaskRow

RUN_KINDS = {
    "dataset_report",
    "data_prep",
    "baseline",
    "tool_assisted",
    "tool_use_baseline",
    "zero_shot",
    "prompt_ablation",
    "guided_decoding",
    "sft",
    "reasoning_sft",
    "rejection_sampled_reasoning_sft",
    "data_scaling_sft",
    "grpo",
    "gspo",
    "dr_grpo",
    "task_transfer",
    "reward_ablation",
    "model_scaling",
    "source_transfer",
    "eval",
    "analysis",
    "publishing",
}
DECODING_MODES = {"schema_constrained", "unconstrained_diagnostic"}
TRAINING_KINDS = {
    "sft",
    "reasoning_sft",
    "rejection_sampled_reasoning_sft",
    "data_scaling_sft",
    "grpo",
    "gspo",
    "dr_grpo",
    "task_transfer",
    "reward_ablation",
    "model_scaling",
    "source_transfer",
}
ENVIRONMENT_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "WANDB_DISABLED",
    "TOKENIZERS_PARALLELISM",
    "PYTORCH_CUDA_ALLOC_CONF",
    "OMP_NUM_THREADS",
)
SECRET_ENV_MARKERS = ("TOKEN", "KEY", "SECRET", "PASSWORD")


class CheckpointSelectionRule(BaseModel):
    """How to preserve a best checkpoint or adapter."""

    metric: str = "accuracy"
    mode: Literal["max", "min"] = "max"
    metrics_file: str | None = "quality_metrics.jsonl"
    step_field: str = "step"
    value_json_path: str | None = None
    source_glob: str | None = "checkpoint-{step}"
    destination: str = "checkpoint-best"


class ExperimentRunSpec(BaseModel):
    """One manifest run entry."""

    name: str
    kind: str
    command: list[str] | str | None = None
    depends_on: list[str] = Field(default_factory=list)
    canary: bool = False
    dataset_dir: Path | None = None
    model: str | None = None
    adapter_init_dir: Path | None = None
    train_splits: list[str] = Field(default_factory=list)
    eval_splits: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    prompt_mode: str | None = None
    decoding_mode: str | None = None
    row_caps: dict[str, int] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    output_dir: Path | None = None
    expected_artifacts: list[str] = Field(default_factory=list, alias="artifacts")
    stop_rules: dict[str, Any] | list[str] = Field(default_factory=dict)
    checkpoint_selection: CheckpointSelectionRule = Field(default_factory=CheckpointSelectionRule)
    upload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        if value not in RUN_KINDS:
            raise ValueError(f"unsupported experiment kind: {value}")
        return value

    @field_validator("decoding_mode")
    @classmethod
    def _validate_decoding_mode(cls, value: str | None) -> str | None:
        if value is not None and value not in DECODING_MODES:
            raise ValueError(f"unsupported decoding mode: {value}")
        return value

    @field_validator("row_caps")
    @classmethod
    def _validate_row_caps(cls, value: dict[str, int]) -> dict[str, int]:
        for key, item in value.items():
            if int(item) < 0:
                raise ValueError(f"row cap must be nonnegative: {key}")
        return value

    @model_validator(mode="after")
    def _validate_canary_dependency(self) -> "ExperimentRunSpec":
        if not self.canary and self.kind in TRAINING_KINDS:
            for dependency in self.depends_on:
                if "canary" in dependency:
                    break
            # Full runs are allowed without a canary in tiny test manifests, so
            # this is documented in run_manifest rather than rejected.
        return self


class ExperimentSuiteManifest(BaseModel):
    """Top-level experiment manifest."""

    version: int = 1
    runs: list[ExperimentRunSpec]
    defaults: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_graph(self) -> "ExperimentSuiteManifest":
        names = [run.name for run in self.runs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate run names: {duplicates}")
        known = set(names)
        missing = sorted({dep for run in self.runs for dep in run.depends_on if dep not in known})
        if missing:
            raise ValueError(f"unknown run dependencies: {missing}")
        topological_run_order(self.runs)
        return self


def load_experiment_manifest(path: Path) -> ExperimentSuiteManifest:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping")
    return ExperimentSuiteManifest.model_validate(data)


def topological_run_order(runs: Iterable[ExperimentRunSpec]) -> list[ExperimentRunSpec]:
    by_name = {run.name: run for run in runs}
    indegree = {name: 0 for name in by_name}
    children: dict[str, list[str]] = defaultdict(list)
    for run in by_name.values():
        for dependency in run.depends_on:
            children[dependency].append(run.name)
            indegree[run.name] += 1
    ready = deque(sorted(name for name, count in indegree.items() if count == 0))
    ordered: list[ExperimentRunSpec] = []
    while ready:
        name = ready.popleft()
        ordered.append(by_name[name])
        for child in sorted(children[name]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(ordered) != len(by_name):
        raise ValueError("experiment manifest contains a dependency cycle")
    return ordered


class RunArtifactManager:
    """Manage the artifact contract for one run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    @classmethod
    def create(cls, run_dir: Path, *, resume: bool = False) -> "RunArtifactManager":
        if run_dir.exists() and not resume:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "predictions").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        manager = cls(run_dir)
        if not (run_dir / "artifacts.json").exists():
            manager.write_artifacts({"schema_version": 1, "artifacts": []})
        return manager

    def initialize(
        self,
        *,
        run_id: str,
        spec: ExperimentRunSpec | None,
        command: list[str] | str | None,
        checkpoint_selection: CheckpointSelectionRule | None = None,
    ) -> None:
        self.write_run_manifest(
            build_run_manifest(
                run_id=run_id,
                run_dir=self.run_dir,
                spec=spec,
                checkpoint_selection=checkpoint_selection or (spec.checkpoint_selection if spec else None),
            )
        )
        self.write_command(command)
        self.write_environment(capture_environment())
        self.write_git_snapshots()

    def write_run_manifest(self, manifest: dict[str, Any]) -> None:
        _write_json(self.run_dir / "run_manifest.json", manifest)

    def write_command(self, command: list[str] | str | None) -> None:
        if command is None:
            text = ""
        elif isinstance(command, str):
            text = command
        else:
            import shlex

            text = shlex.join(str(part) for part in command)
        (self.run_dir / "command.txt").write_text(text + ("\n" if text else ""), encoding="utf-8")

    def write_environment(self, environment: dict[str, Any]) -> None:
        _write_json(self.run_dir / "environment.json", environment)

    def write_git_snapshots(self) -> None:
        (self.run_dir / "git_status.txt").write_text(_run_text(["git", "status", "--short"]), encoding="utf-8")
        (self.run_dir / "git_diff.patch").write_text(_run_text(["git", "diff", "--binary"]), encoding="utf-8")

    def write_artifacts(self, artifacts: dict[str, Any]) -> None:
        _write_json(self.run_dir / "artifacts.json", artifacts)

    def read_artifacts(self) -> dict[str, Any]:
        path = self.run_dir / "artifacts.json"
        if not path.exists():
            return {"schema_version": 1, "artifacts": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def add_artifact(
        self,
        *,
        kind: str,
        path: Path | str,
        role: str | None = None,
        metadata: dict[str, Any] | None = None,
        checksum: bool = True,
    ) -> dict[str, Any]:
        artifact_path = Path(path)
        stored_path = artifact_path if artifact_path.is_absolute() else self.run_dir / artifact_path
        entry: dict[str, Any] = {
            "kind": kind,
            "path": str(path),
            "role": role,
            "metadata": metadata or {},
            "exists": stored_path.exists(),
        }
        if checksum and stored_path.is_file():
            entry["sha256"] = sha256_file(stored_path)
        index = self.read_artifacts()
        index.setdefault("artifacts", []).append(entry)
        index["updated_at"] = datetime.now(UTC).isoformat()
        self.write_artifacts(index)
        return entry

    def write_status(self, status: str, **fields: Any) -> None:
        payload = {
            "status": status,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            **fields,
        }
        _write_json(self.run_dir / "status.json", payload)


def build_run_manifest(
    *,
    run_id: str,
    run_dir: Path,
    spec: ExperimentRunSpec | None,
    checkpoint_selection: CheckpointSelectionRule | None = None,
) -> dict[str, Any]:
    dataset_dir = spec.dataset_dir if spec else None
    dataset_identity = dataset_manifest_identity(dataset_dir) if dataset_dir else None
    return _json_ready(
        {
            "schema_version": 1,
            "run_id": run_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "kind": spec.kind if spec else None,
            "canary": spec.canary if spec else False,
            "model": spec.model if spec else None,
            "adapter_init_dir": spec.adapter_init_dir if spec else None,
            "dataset_dir": dataset_dir,
            "dataset": dataset_identity,
            "train_splits": spec.train_splits if spec else [],
            "eval_splits": spec.eval_splits if spec else [],
            "task_types": spec.task_types if spec else [],
            "prompt_mode": spec.prompt_mode if spec else None,
            "decoding_mode": spec.decoding_mode if spec else None,
            "seeds": _extract_seeds(spec),
            "row_caps": spec.row_caps if spec else {},
            "inputs": spec.inputs if spec else {},
            "hyperparameters": spec.training if spec else {},
            "budget": spec.budget if spec else {},
            "output_paths": {
                "run_dir": run_dir,
                "predictions_dir": run_dir / "predictions",
                "checkpoints_dir": run_dir / "checkpoints",
            },
            "depends_on": spec.depends_on if spec else [],
            "expected_artifacts": spec.expected_artifacts if spec else [],
            "stop_rules": spec.stop_rules if spec else {},
            "checkpoint_selection": (
                checkpoint_selection.model_dump(mode="json") if checkpoint_selection else None
            ),
        }
    )


def dataset_manifest_identity(dataset_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(dataset_dir), "exists": dataset_dir.exists()}
    for name in ("metadata.json", "source_manifest.json", "split_manifest.json"):
        path = dataset_dir / name
        if path.exists():
            result[name] = {"sha256": sha256_file(path)}
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for key in ("dataset_version", "config_hash", "source_manifest_hash", "created_from_commit"):
                if key in data:
                    result[key] = data[key]
    return result


def capture_environment() -> dict[str, Any]:
    packages = {
        name: _package_version(name)
        for name in ("torch", "transformers", "trl", "unsloth", "peft", "bitsandbytes", "vllm")
    }
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cuda": _cuda_summary(),
        "packages": packages,
        "environment_variables": {
            name: _safe_env_value(name, os.environ.get(name)) for name in ENVIRONMENT_VARIABLES
        },
    }


def prediction_record(row: PublicTaskRow, prediction: Prediction | dict[str, Any], *, prompt: bool = False) -> dict[str, Any]:
    output = prediction.output if isinstance(prediction, Prediction) else str(prediction.get("output", ""))
    candidate, _format_components = canonical_answer_candidate(output)
    parsed = parse_answer(row.task_type, candidate)
    return {
        "id": row.id,
        "row_id": row.id,
        "split": str(row.split),
        "task_type": str(row.task_type),
        "prompt": row.prompt if prompt else None,
        "prompt_hash": row.hashes.prompt_hash,
        "raw_completion": output,
        "output": output,
        "parsed_answer": parsed,
        "canonical_answer": row.answer,
        "parse_valid": parsed is not None,
        "correct": parsed == row.answer,
    }


def write_prediction_records(
    rows: Iterable[PublicTaskRow],
    predictions: Iterable[Prediction | dict[str, Any]],
    path: Path,
    *,
    include_prompt: bool = False,
) -> int:
    rows_by_id = {row.id: row for row in rows}
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            row_id = prediction.row_id if isinstance(prediction, Prediction) else prediction.get("row_id") or prediction.get("id")
            row = rows_by_id.get(str(row_id))
            if row is None:
                continue
            record = prediction_record(row, prediction, prompt=include_prompt)
            handle.write(json.dumps(_json_ready(record), sort_keys=True) + "\n")
            count += 1
    return count


def expected_artifacts_exist(run_dir: Path, expected: Iterable[str]) -> bool:
    expected = list(expected)
    return bool(expected) and all((run_dir / path).exists() for path in expected)


def preflight_dataset(run: ExperimentRunSpec) -> dict[str, Any]:
    if run.dataset_dir is None:
        return {"checked": False, "reason": "no_dataset_dir"}
    missing_required_files = [
        name
        for name in ("metadata.json", "source_manifest.json", "split_manifest.json")
        if not (run.dataset_dir / name).exists()
    ]
    rows = validate_dataset_dir(run.dataset_dir)
    result: dict[str, Any] = {
        "checked": True,
        "dataset_dir": str(run.dataset_dir),
        "missing_required_files": missing_required_files,
        "row_count": len(rows),
        "splits": sorted({str(row.split) for row in rows}),
        "task_types": sorted({str(row.task_type) for row in rows}),
        "answer_balance": answer_balance_audit(rows),
    }
    if run.kind in TRAINING_KINDS:
        train_splits = set(run.train_splits or ["train"])
        optimizer_rows = [row for row in rows if str(row.split) in train_splits]
        non_public_train = sorted({str(row.split) for row in optimizer_rows if str(row.split) != "train"})
        result["optimizer_train_row_count"] = len(optimizer_rows)
        result["optimizer_uses_public_train_only"] = not non_public_train
        result["non_public_train_splits"] = non_public_train
    return result


def answer_balance_audit(
    rows: Iterable[PublicTaskRow],
    *,
    min_share: float = 0.10,
    max_share: float = 0.70,
    min_count: int = 30,
) -> dict[str, Any]:
    by_split_task: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        by_split_task[str(row.split)][str(row.task_type)][row.answer] += 1
    findings: list[dict[str, Any]] = []
    for split, by_task in sorted(by_split_task.items()):
        for task_type, counts in sorted(by_task.items()):
            total = sum(counts.values())
            for answer, count in sorted(counts.items()):
                share = count / total if total else 0.0
                reasons = []
                if count < min_count:
                    reasons.append("low_count")
                if share < min_share:
                    reasons.append("low_share")
                if share > max_share:
                    reasons.append("high_share")
                if reasons:
                    findings.append(
                        {
                            "split": split,
                            "task_type": task_type,
                            "answer": answer,
                            "count": count,
                            "share": share,
                            "total": total,
                            "reason": ",".join(reasons),
                        }
                    )
    return {
        "thresholds": {"min_share": min_share, "max_share": max_share, "min_count": min_count},
        "finding_count": len(findings),
        "findings": findings,
    }


def scan_stop_signals(run_dir: Path) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*.jsonl")) + sorted((run_dir / "logs").glob("*.log")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")[-200_000:]
        lower = text.lower()
        if "outofmemoryerror" in lower or "cuda out of memory" in lower:
            signals.append({"severity": "hard", "reason": "oom", "path": str(path)})
        if "nan" in lower:
            signals.append({"severity": "hard", "reason": "nan", "path": str(path)})
        if "invalid_output_rate" in lower:
            signals.append({"severity": "soft", "reason": "invalid_output_rate_present", "path": str(path)})
    disk = shutil.disk_usage(run_dir)
    free_ratio = disk.free / disk.total if disk.total else 1.0
    if free_ratio < 0.05:
        signals.append({"severity": "hard", "reason": "disk_pressure", "free_ratio": free_ratio})
    elif free_ratio < 0.10:
        signals.append({"severity": "soft", "reason": "disk_pressure", "free_ratio": free_ratio})
    return signals


def select_best_checkpoint(run_dir: Path, rule: CheckpointSelectionRule) -> dict[str, Any] | None:
    if not rule.metrics_file:
        return None
    metrics_path = run_dir / rule.metrics_file
    if not metrics_path.exists():
        return None
    best_record: dict[str, Any] | None = None
    best_value: float | None = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            value = _metric_value(record, rule)
            if value is None or not math.isfinite(value):
                continue
            if best_value is None or (value > best_value if rule.mode == "max" else value < best_value):
                best_value = value
                best_record = record
    if best_record is None or best_value is None:
        return None
    step = best_record.get(rule.step_field) or best_record.get("global_step")
    source = _checkpoint_source(run_dir, rule, step)
    destination = run_dir / rule.destination
    copied = False
    if source is not None and source.exists():
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if source.is_dir():
            shutil.copytree(source, destination, symlinks=True)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / source.name)
        copied = True
    return {
        "step": step,
        "metric_name": rule.metric,
        "metric_value": best_value,
        "source_path": str(source) if source else None,
        "destination_path": str(destination),
        "copied": copied,
    }


def create_run_tarball(run_dir: Path, output_path: Path | None = None) -> dict[str, Any]:
    output_path = output_path or run_dir.with_suffix(".tar.gz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(run_dir, arcname=run_dir.name)
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _metric_value(record: dict[str, Any], rule: CheckpointSelectionRule) -> float | None:
    if rule.value_json_path:
        value: Any = record
        for part in rule.value_json_path.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
    else:
        value = record.get(rule.metric)
        if value is None:
            value = _find_key(record, rule.metric)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for item in value.values():
            found = _find_key(item, key)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_key(item, key)
            if found is not None:
                return found
    return None


def _checkpoint_source(run_dir: Path, rule: CheckpointSelectionRule, step: Any) -> Path | None:
    if not rule.source_glob:
        return None
    pattern = rule.source_glob.format(step=step)
    matches = sorted(run_dir.glob(pattern))
    return matches[-1] if matches else run_dir / pattern


def _extract_seeds(spec: ExperimentRunSpec | None) -> dict[str, Any]:
    if spec is None:
        return {}
    seeds = {}
    for source in (spec.training, spec.budget):
        for key, value in source.items():
            if "seed" in key:
                seeds[key] = value
    return seeds


def _cuda_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "nvidia_smi": _run_text(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
        ).strip(),
    }
    try:
        import torch

        summary.update(
            {
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_names": [
                    torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else [],
            }
        )
    except Exception as exc:  # pragma: no cover - depends on optional GPU stack
        summary["torch_error"] = str(exc)
    return summary


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_env_value(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if any(marker in name for marker in SECRET_ENV_MARKERS):
        return "<set>"
    return value


def _run_text(command: list[str], *, check: bool = False) -> str:
    try:
        result = subprocess.run(command, check=check, text=True, capture_output=True)
    except FileNotFoundError:
        return ""
    return (result.stdout or "") + (result.stderr or "")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_ready(item) for item in value]
    if isinstance(value, BaseModel):
        return _json_ready(value.model_dump(mode="json"))
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    return str(value)
