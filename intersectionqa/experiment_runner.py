"""Modular execution engine for experiment manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Protocol

from intersectionqa.experiments import (
    ExperimentRunSpec,
    ExperimentSuiteManifest,
    RunArtifactManager,
    expected_artifacts_exist,
    preflight_dataset,
    scan_stop_signals,
    select_best_checkpoint,
    topological_run_order,
)


@dataclass(frozen=True)
class ExperimentSelection:
    """Select a deterministic subset of manifest runs."""

    names: tuple[str, ...] = ()
    start_from: str | None = None
    run_until: str | None = None
    with_dependencies: bool = False
    skip_dependencies: bool = False


@dataclass(frozen=True)
class ExperimentRunnerOptions:
    """Runtime controls for the local orchestrator."""

    runs_dir: Path = Path("runs")
    resume: bool = False
    dry_run: bool = False
    fail_fast: bool = True
    selection: ExperimentSelection = field(default_factory=ExperimentSelection)
    stop_signal_scanner: Callable[[Path], list[dict[str, object]]] = scan_stop_signals
    bucket_syncer: Callable[[Path, str], dict[str, object]] | None = None


@dataclass(frozen=True)
class ExperimentContext:
    """Resolved paths and manifest context for one run."""

    manifest_path: Path | None
    manifest: ExperimentSuiteManifest
    run: ExperimentRunSpec
    run_dir: Path
    runs_dir: Path
    run_dirs: dict[str, Path]


@dataclass(frozen=True)
class ExperimentExecutionResult:
    """Result returned by a concrete experiment executor."""

    return_code: int = 0


class ExperimentExecutor(Protocol):
    """Protocol for command-backed or function-backed experiment implementations."""

    def execute(self, context: ExperimentContext) -> ExperimentExecutionResult:
        """Execute one experiment run."""


class ShellCommandExperimentExecutor:
    """Execute the manifest command for a run."""

    def execute(self, context: ExperimentContext) -> ExperimentExecutionResult:
        command = expand_command(context.run.command, context)
        if command is None:
            raise ValueError(
                f"{context.run.name}: no command configured and no function executor is registered"
            )
        logs_dir = context.run_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        stdout_path = logs_dir / "stdout.log"
        stderr_path = logs_dir / "stderr.log"
        shell = isinstance(command, str)
        env = _experiment_environment(context)
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            stdout.write(f"\n[started {datetime.now(UTC).isoformat()}]\n".encode())
            stdout.flush()
            process = subprocess.run(
                command,
                shell=shell,
                stdout=stdout,
                stderr=stderr,
                cwd=Path.cwd(),
                env=env,
            )
            stdout.write(
                f"\n[finished {datetime.now(UTC).isoformat()} return_code={process.returncode}]\n".encode()
            )
        return ExperimentExecutionResult(return_code=int(process.returncode))


class ExperimentExecutorRegistry:
    """Resolve run kinds to executors.

    The default executor runs shell commands. Registering a kind-specific
    executor lets future experiments run as Python protocols/functions while
    keeping the same artifact, dependency, and stop-rule contract.
    """

    def __init__(self, default_executor: ExperimentExecutor | None = None) -> None:
        self._default_executor = default_executor or ShellCommandExperimentExecutor()
        self._by_kind: dict[str, ExperimentExecutor] = {}

    def register(self, kind: str, executor: ExperimentExecutor) -> None:
        self._by_kind[kind] = executor

    def get(self, run: ExperimentRunSpec) -> ExperimentExecutor:
        return self._by_kind.get(run.kind, self._default_executor)


class ExperimentRunner:
    """Run a selected subset of a manifest predictably and restartably."""

    def __init__(
        self,
        manifest: ExperimentSuiteManifest,
        *,
        manifest_path: Path | None = None,
        options: ExperimentRunnerOptions | None = None,
        registry: ExperimentExecutorRegistry | None = None,
    ) -> None:
        self.manifest = manifest
        self.manifest_path = manifest_path
        self.options = options or ExperimentRunnerOptions()
        self.registry = registry or ExperimentExecutorRegistry()
        self._ordered = topological_run_order(manifest.runs)
        self._run_dirs = {
            run.name: run.output_dir or self.options.runs_dir / run.name for run in self._ordered
        }

    def selected_runs(self) -> list[ExperimentRunSpec]:
        return select_runs(self._ordered, self.options.selection)

    def run(self) -> list[dict[str, object]]:
        selected = self.selected_runs()
        selected_names = {run.name for run in selected}
        completed = {
            run.name
            for run in self._ordered
            if run.name not in selected_names and is_run_complete(self._run_dirs[run.name], run)
        }
        failed: set[str] = set()
        summary: list[dict[str, object]] = []

        for run in selected:
            run_dir = self._run_dirs[run.name]
            if not self.options.selection.skip_dependencies:
                dependency_failures = [dependency for dependency in run.depends_on if dependency in failed]
                if dependency_failures:
                    summary.append(
                        _status(
                            run,
                            run_dir,
                            "deferred",
                            reason="failed_dependency",
                            dependencies=dependency_failures,
                        )
                    )
                    failed.add(run.name)
                    if self.options.fail_fast:
                        break
                    continue
                missing_dependencies = [
                    dependency
                    for dependency in run.depends_on
                    if dependency not in completed and dependency not in selected_names
                ]
                missing_dependencies.extend(
                    dependency
                    for dependency in run.depends_on
                    if dependency in selected_names and dependency not in completed
                )
                if missing_dependencies:
                    summary.append(
                        _status(
                            run,
                            run_dir,
                            "deferred",
                            reason="missing_dependency",
                            dependencies=sorted(set(missing_dependencies)),
                        )
                    )
                    failed.add(run.name)
                    if self.options.fail_fast:
                        break
                    continue

            if expected_artifacts_exist(run_dir, run.expected_artifacts):
                manager = RunArtifactManager.create(run_dir, resume=True)
                manager.write_status("skipped", reason="expected_artifacts_exist")
                sync_run_to_hf_bucket(manager, self._context(run), self.options.bucket_syncer)
                summary.append(_status(run, run_dir, "skipped"))
                completed.add(run.name)
                continue

            if self.options.dry_run:
                context = self._context(run)
                summary.append(
                    _status(
                        run,
                        run_dir,
                        "planned",
                        command=expand_command(run.command, context),
                    )
                )
                completed.add(run.name)
                continue

            try:
                manager = RunArtifactManager.create(run_dir, resume=self.options.resume)
                context = self._context(run)
                manager.initialize(
                    run_id=run.name,
                    spec=run,
                    command=expand_command(run.command, context),
                    checkpoint_selection=run.checkpoint_selection,
                )
                preflight = preflight_dataset(run)
                (run_dir / "preflight.json").write_text(
                    json.dumps(preflight, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                manager.add_artifact(
                    kind="report",
                    path="preflight.json",
                    role="preflight",
                    metadata=preflight,
                    checksum=False,
                )

                result = self.registry.get(run).execute(context)
                index_common_artifacts(manager, run)
                stop_signals = self.options.stop_signal_scanner(run_dir)
                if stop_signals:
                    (run_dir / "stop_signals.json").write_text(
                        json.dumps(stop_signals, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                if result.return_code != 0:
                    manager.write_status("failed", return_code=result.return_code, stop_signals=stop_signals)
                    failed.add(run.name)
                    summary.append(_status(run, run_dir, "failed", return_code=result.return_code))
                    if self.options.fail_fast:
                        break
                    continue
                if any(signal.get("severity") == "hard" for signal in stop_signals):
                    manager.write_status("stopped", reason="hard_stop_signal", stop_signals=stop_signals)
                    failed.add(run.name)
                    summary.append(_status(run, run_dir, "stopped", reason="hard_stop_signal"))
                    if self.options.fail_fast:
                        break
                    continue
                if run.expected_artifacts and not expected_artifacts_exist(run_dir, run.expected_artifacts):
                    missing = [
                        artifact for artifact in run.expected_artifacts if not (run_dir / artifact).exists()
                    ]
                    manager.write_status("failed", reason="missing_expected_artifacts", missing_artifacts=missing)
                    failed.add(run.name)
                    summary.append(
                        _status(
                            run,
                            run_dir,
                            "failed",
                            reason="missing_expected_artifacts",
                            missing_artifacts=missing,
                        )
                    )
                    if self.options.fail_fast:
                        break
                    continue
                best = select_best_checkpoint(run_dir, run.checkpoint_selection)
                if best is not None:
                    manager.add_artifact(kind="checkpoint", path=best["destination_path"], role="best", metadata=best)
                sync_run_to_hf_bucket(manager, context, self.options.bucket_syncer)
                manager.write_status("success")
                summary.append(_status(run, run_dir, "success"))
                completed.add(run.name)
            except Exception as exc:
                RunArtifactManager.create(run_dir, resume=True).write_status("failed", error=str(exc))
                failed.add(run.name)
                summary.append(_status(run, run_dir, "failed", error=str(exc)))
                if self.options.fail_fast:
                    break

        if failed:
            summary.append({"status": "incomplete", "failed": sorted(failed)})
        return summary

    def _context(self, run: ExperimentRunSpec) -> ExperimentContext:
        return ExperimentContext(
            manifest_path=self.manifest_path,
            manifest=self.manifest,
            run=run,
            run_dir=self._run_dirs[run.name],
            runs_dir=self.options.runs_dir,
            run_dirs=self._run_dirs,
        )


def select_runs(
    ordered_runs: list[ExperimentRunSpec],
    selection: ExperimentSelection,
) -> list[ExperimentRunSpec]:
    """Return selected runs in topological order."""

    if not ordered_runs:
        return []
    names = [run.name for run in ordered_runs]
    name_set = set(names)
    for requested in [*selection.names, selection.start_from, selection.run_until]:
        if requested is not None and requested not in name_set:
            raise ValueError(f"unknown run name: {requested}")

    start_index = names.index(selection.start_from) if selection.start_from else 0
    end_index = names.index(selection.run_until) if selection.run_until else len(ordered_runs) - 1
    if start_index > end_index:
        raise ValueError("--start-from must not appear after --run-until in dependency order")

    window = ordered_runs[start_index : end_index + 1]
    if selection.names:
        requested = set(selection.names)
        window = [run for run in window if run.name in requested]
        if selection.with_dependencies:
            dependencies = _transitive_dependencies(ordered_runs, requested)
            allowed = requested | dependencies
            window = [run for run in ordered_runs[: end_index + 1] if run.name in allowed]
    return window


def is_run_complete(run_dir: Path, run: ExperimentRunSpec) -> bool:
    if expected_artifacts_exist(run_dir, run.expected_artifacts):
        return True
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return status.get("status") in {"success", "skipped"}


def expand_command(command: list[str] | str | None, context: ExperimentContext) -> list[str] | str | None:
    if command is None:
        return None
    if isinstance(command, str):
        return _expand_text(command, context)
    return [_expand_text(str(part), context) for part in command]


def sync_run_to_hf_bucket(
    manager: RunArtifactManager,
    context: ExperimentContext,
    bucket_syncer: Callable[[Path, str], dict[str, object]] | None = None,
) -> dict[str, object] | None:
    config = _upload_config(context)
    target = config.get("hf_bucket")
    if not target:
        return None
    target = _bucket_target(str(target), context)
    started = datetime.now(UTC).isoformat()
    record: dict[str, object] = {
        "status": "started",
        "started_at_utc": started,
        "hf_bucket": target,
    }
    (context.run_dir / "bucket_upload.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        syncer = bucket_syncer or _default_bucket_syncer
        result = syncer(context.run_dir, target)
        record.update(
            {
                "status": "uploaded",
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "result": result,
            }
        )
        (context.run_dir / "bucket_upload.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        syncer(context.run_dir, target)
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "error": str(exc),
            }
        )
        (context.run_dir / "bucket_upload.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manager.add_artifact(kind="upload", path="bucket_upload.json", role="hf_bucket", metadata=record, checksum=False)
        if config.get("required", True):
            raise
        return record

    manager.add_artifact(kind="upload", path="bucket_upload.json", role="hf_bucket", metadata=record, checksum=False)
    return record


def _upload_config(context: ExperimentContext) -> dict[str, Any]:
    defaults = context.manifest.defaults.get("upload", {})
    config: dict[str, Any] = dict(defaults) if isinstance(defaults, dict) else {}
    config.update(context.run.upload)
    return config


def _bucket_target(bucket: str, context: ExperimentContext) -> str:
    base = bucket.rstrip("/")
    if "{run" in base or "{manifest" in base:
        base = _expand_text(base, context)
    if "/runs/" in base or base.endswith(f"/{context.run.name}"):
        return base
    prefix = _upload_prefix(context)
    return f"{base}/{prefix}/runs/{context.run.name}"


def _upload_prefix(context: ExperimentContext) -> str:
    configured = _upload_config(context).get("prefix")
    if configured:
        return _expand_text(str(configured).strip("/"), context)
    if context.manifest_path is not None:
        stem = context.manifest_path.stem
    else:
        stem = "experiment_suite"
    return stem


def _default_bucket_syncer(source: Path, target: str) -> dict[str, object]:
    from huggingface_hub import sync_bucket

    result = sync_bucket(str(source), target)
    summary = getattr(result, "summary", None)
    if callable(summary):
        return dict(summary())
    return {"target": target}


def index_common_artifacts(manager: RunArtifactManager, run: ExperimentRunSpec) -> None:
    run_dir = manager.run_dir
    common = [
        ("metric", "train_metrics.jsonl", "train"),
        ("metric", "eval_metrics.jsonl", "eval"),
        ("metric", "quality_metrics.jsonl", "quality"),
        ("report", "train_result.json", "train_result"),
        ("log", "logs/stdout.log", "stdout"),
        ("log", "logs/stderr.log", "stderr"),
    ]
    for kind, relative, role in common:
        if (run_dir / relative).exists():
            manager.add_artifact(kind=kind, path=relative, role=role)
    predictions_dir = run_dir / "predictions"
    if predictions_dir.exists():
        for path in sorted(predictions_dir.glob("*.jsonl")):
            manager.add_artifact(kind="prediction", path=path.relative_to(run_dir), role="prediction")
    for name in ("adapter", "adapter-best", "checkpoint-best"):
        if (run_dir / name).exists():
            manager.add_artifact(kind="adapter" if "adapter" in name else "checkpoint", path=name, role=name)
    for checkpoint in sorted(run_dir.glob("checkpoint-*")):
        manager.add_artifact(kind="checkpoint", path=checkpoint.relative_to(run_dir), role="checkpoint")
    if (run_dir / "quality_metrics.jsonl").exists() and not (run_dir / "eval_metrics.jsonl").exists():
        shutil.copy2(run_dir / "quality_metrics.jsonl", run_dir / "eval_metrics.jsonl")
        manager.add_artifact(
            kind="metric",
            path="eval_metrics.jsonl",
            role="eval_alias",
            metadata={"source": "quality_metrics.jsonl"},
        )
    for expected in run.expected_artifacts:
        if (run_dir / expected).exists():
            manager.add_artifact(kind="expected", path=expected, role="expected")


_PLACEHOLDER_RE = re.compile(r"\{(run_dir|runs_dir|run_name|manifest_dir|manifest_path|run:([A-Za-z0-9_.-]+))\}")


def _expand_text(value: str, context: ExperimentContext) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        referenced_run = match.group(2)
        if token == "run_dir":
            return str(context.run_dir)
        if token == "runs_dir":
            return str(context.runs_dir)
        if token == "run_name":
            return context.run.name
        if token == "manifest_path":
            return str(context.manifest_path or "")
        if token == "manifest_dir":
            return str((context.manifest_path or Path(".")).parent)
        if referenced_run:
            path = context.run_dirs.get(referenced_run)
            if path is None:
                raise ValueError(f"{context.run.name}: unknown command placeholder run: {referenced_run}")
            return str(path)
        return match.group(0)

    return _PLACEHOLDER_RE.sub(replace, value)


def _experiment_environment(context: ExperimentContext) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "INTERSECTIONQA_RUN_NAME": context.run.name,
            "INTERSECTIONQA_RUN_DIR": str(context.run_dir),
            "INTERSECTIONQA_RUNS_DIR": str(context.runs_dir),
            "INTERSECTIONQA_MANIFEST_PATH": str(context.manifest_path or ""),
        }
    )
    for name, path in context.run_dirs.items():
        env[f"INTERSECTIONQA_RUN_DIR_{_env_key(name)}"] = str(path)
    return env


def _env_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()


def _transitive_dependencies(ordered_runs: list[ExperimentRunSpec], names: set[str]) -> set[str]:
    by_name = {run.name: run for run in ordered_runs}
    dependencies: set[str] = set()
    stack = list(names)
    while stack:
        name = stack.pop()
        run = by_name[name]
        for dependency in run.depends_on:
            if dependency not in dependencies:
                dependencies.add(dependency)
                stack.append(dependency)
    return dependencies


def _status(run: ExperimentRunSpec, run_dir: Path, status: str, **extra: object) -> dict[str, object]:
    return {"name": run.name, "kind": run.kind, "run_dir": str(run_dir), "status": status, **extra}
