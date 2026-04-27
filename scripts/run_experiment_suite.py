"""Run an experiment manifest with restartable per-run artifacts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

from intersectionqa.experiments import (
    ExperimentRunSpec,
    RunArtifactManager,
    expected_artifacts_exist,
    load_experiment_manifest,
    preflight_dataset,
    scan_stop_signals,
    select_best_checkpoint,
    topological_run_order,
)
from intersectionqa.logging import configure_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="append", help="Run only this manifest entry and its satisfied dependencies.")
    args = parser.parse_args(argv)

    configure_logging()
    manifest = load_experiment_manifest(args.manifest)
    selected_names = set(args.run or [])
    completed: set[str] = set()
    failed: set[str] = set()
    summary: list[dict[str, object]] = []

    for run in topological_run_order(manifest.runs):
        if selected_names and run.name not in selected_names:
            continue
        run_dir = run.output_dir or args.runs_dir / run.name
        dependency_failures = [dependency for dependency in run.depends_on if dependency in failed]
        if dependency_failures:
            summary.append(_status(run, run_dir, "deferred", reason="failed_dependency", dependencies=dependency_failures))
            failed.add(run.name)
            continue
        missing_dependencies = [dependency for dependency in run.depends_on if dependency not in completed]
        if missing_dependencies:
            summary.append(_status(run, run_dir, "deferred", reason="missing_dependency", dependencies=missing_dependencies))
            failed.add(run.name)
            continue
        if expected_artifacts_exist(run_dir, run.expected_artifacts):
            manager = RunArtifactManager.create(run_dir, resume=True)
            manager.write_status("skipped", reason="expected_artifacts_exist")
            summary.append(_status(run, run_dir, "skipped"))
            completed.add(run.name)
            continue
        if args.dry_run:
            summary.append(_status(run, run_dir, "planned", command=run.command))
            completed.add(run.name)
            continue

        try:
            manager = RunArtifactManager.create(run_dir, resume=args.resume)
            manager.initialize(
                run_id=run.name,
                spec=run,
                command=run.command,
                checkpoint_selection=run.checkpoint_selection,
            )
            preflight = preflight_dataset(run)
            manager.add_artifact(kind="report", path="preflight.json", role="preflight", metadata=preflight, checksum=False)
            (run_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if run.command:
                return_code = _execute(run, run_dir)
                _index_common_artifacts(manager, run)
                stop_signals = scan_stop_signals(run_dir)
                if stop_signals:
                    (run_dir / "stop_signals.json").write_text(
                        json.dumps(stop_signals, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                if return_code != 0:
                    manager.write_status("failed", return_code=return_code, stop_signals=stop_signals)
                    failed.add(run.name)
                    summary.append(_status(run, run_dir, "failed", return_code=return_code))
                    break
                if any(signal.get("severity") == "hard" for signal in stop_signals):
                    manager.write_status("stopped", reason="hard_stop_signal", stop_signals=stop_signals)
                    failed.add(run.name)
                    summary.append(_status(run, run_dir, "stopped", reason="hard_stop_signal"))
                    break
            best = select_best_checkpoint(run_dir, run.checkpoint_selection)
            if best is not None:
                manager.add_artifact(kind="checkpoint", path=best["destination_path"], role="best", metadata=best)
            manager.write_status("success")
            summary.append(_status(run, run_dir, "success"))
            completed.add(run.name)
        except Exception as exc:
            RunArtifactManager.create(run_dir, resume=True).write_status("failed", error=str(exc))
            failed.add(run.name)
            summary.append(_status(run, run_dir, "failed", error=str(exc)))
            break

    print(json.dumps({"runs": summary}, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


def _execute(run: ExperimentRunSpec, run_dir: Path) -> int:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"
    command = run.command
    if command is None:
        return 0
    shell = isinstance(command, str)
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        stdout.write(f"\n[started {datetime.now(UTC).isoformat()}]\n".encode())
        stdout.flush()
        process = subprocess.run(command, shell=shell, stdout=stdout, stderr=stderr, cwd=Path.cwd())
        stdout.write(f"\n[finished {datetime.now(UTC).isoformat()} return_code={process.returncode}]\n".encode())
    return int(process.returncode)


def _index_common_artifacts(manager: RunArtifactManager, run: ExperimentRunSpec) -> None:
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


def _status(run: ExperimentRunSpec, run_dir: Path, status: str, **extra: object) -> dict[str, object]:
    return {"name": run.name, "kind": run.kind, "run_dir": str(run_dir), "status": status, **extra}


if __name__ == "__main__":
    main()

