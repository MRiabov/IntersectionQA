"""Run an experiment manifest with restartable per-run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.experiment_runner import (
    ExperimentRunner,
    ExperimentRunnerOptions,
    ExperimentSelection,
)
from intersectionqa.experiments import load_experiment_manifest, scan_stop_signals  # re-exported for older tests
from intersectionqa.logging import configure_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", help="List the selected runs without executing them.")
    parser.add_argument("--run", action="append", help="Run only this manifest entry. Repeat for a set.")
    parser.add_argument("--start-from", help="Run the dependency-order suffix starting at this run.")
    parser.add_argument("--run-until", help="Run the dependency-order prefix ending at this run.")
    parser.add_argument(
        "--with-dependencies",
        action="store_true",
        help="When --run is used, include transitive dependencies in dependency order.",
    )
    parser.add_argument(
        "--skip-dependencies",
        action="store_true",
        help="Do not require dependency runs to be completed. Use only for manual recovery/debug.",
    )
    parser.add_argument("--no-fail-fast", action="store_true", help="Continue after a failed run when possible.")
    args = parser.parse_args(argv)

    configure_logging()
    manifest = load_experiment_manifest(args.manifest)
    options = ExperimentRunnerOptions(
        runs_dir=args.runs_dir,
        resume=args.resume,
        dry_run=args.dry_run or args.list,
        fail_fast=not args.no_fail_fast,
        stop_signal_scanner=scan_stop_signals,
        selection=ExperimentSelection(
            names=tuple(args.run or ()),
            start_from=args.start_from,
            run_until=args.run_until,
            with_dependencies=args.with_dependencies,
            skip_dependencies=args.skip_dependencies,
        ),
    )
    runner = ExperimentRunner(manifest, manifest_path=args.manifest, options=options)
    if args.list:
        print(json.dumps({"runs": [_planned_run(run, runner) for run in runner.selected_runs()]}, indent=2, sort_keys=True))
        return

    summary = runner.run()
    print(json.dumps({"runs": summary}, indent=2, sort_keys=True))
    if any(item.get("status") == "incomplete" for item in summary):
        raise SystemExit(1)


def _planned_run(run, runner: ExperimentRunner) -> dict[str, object]:  # noqa: ANN001
    run_dir = run.output_dir or runner.options.runs_dir / run.name
    return {
        "name": run.name,
        "kind": run.kind,
        "depends_on": run.depends_on,
        "run_dir": str(run_dir),
    }


if __name__ == "__main__":
    main()
