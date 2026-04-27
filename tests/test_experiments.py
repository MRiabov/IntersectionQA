import json
from pathlib import Path

import pytest

from intersectionqa.config import DatasetConfig
from intersectionqa.evaluation.metrics import Prediction
from intersectionqa.experiments import (
    CheckpointSelectionRule,
    ExperimentSuiteManifest,
    RunArtifactManager,
    load_experiment_manifest,
    prediction_record,
    select_best_checkpoint,
    topological_run_order,
)
from intersectionqa.experiment_runner import (
    ExperimentRunner,
    ExperimentRunnerOptions,
    ExperimentSelection,
    expand_command,
)
from intersectionqa.pipeline import build_smoke_rows
from scripts.experiments.run_experiment_suite import main as run_suite_main


def test_manifest_schema_validates_dependency_order(tmp_path):
    path = tmp_path / "suite.yaml"
    path.write_text(
        """
version: 1
runs:
  - name: canary
    kind: sft
    canary: true
    decoding_mode: schema_constrained
    artifacts: [train_result.json]
  - name: full
    kind: sft
    depends_on: [canary]
    artifacts: [adapter]
""",
        encoding="utf-8",
    )

    manifest = load_experiment_manifest(path)

    assert [run.name for run in topological_run_order(manifest.runs)] == ["canary", "full"]
    assert manifest.runs[0].expected_artifacts == ["train_result.json"]


def test_manifest_rejects_unknown_dependencies():
    with pytest.raises(ValueError, match="unknown run dependencies"):
        ExperimentSuiteManifest.model_validate(
            {"runs": [{"name": "full", "kind": "eval", "depends_on": ["missing"]}]}
        )


def test_run_artifact_manager_writes_snapshots_and_refuses_clobber(tmp_path):
    run_dir = tmp_path / "runs" / "unit"
    manager = RunArtifactManager.create(run_dir)
    manager.initialize(run_id="unit", spec=None, command=["python", "-m", "example"])

    assert json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["run_id"] == "unit"
    assert (run_dir / "command.txt").read_text(encoding="utf-8").strip() == "python -m example"
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "git_status.txt").exists()
    assert (run_dir / "git_diff.patch").exists()
    assert json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))["artifacts"] == []
    with pytest.raises(FileExistsError):
        RunArtifactManager.create(run_dir)


def test_prediction_record_contains_auditable_fields():
    rows, _ = build_smoke_rows(DatasetConfig())
    row = rows[0]

    record = prediction_record(row, Prediction(row_id=row.id, output=row.answer))

    assert record["id"] == row.id
    assert record["split"] == row.split
    assert record["task_type"] == row.task_type
    assert record["raw_completion"] == row.answer
    assert record["parsed_answer"] == row.answer
    assert record["canonical_answer"] == row.answer
    assert record["parse_valid"] is True
    assert record["correct"] is True
    assert record["prompt_hash"] == row.hashes.prompt_hash


def test_best_checkpoint_selection_copies_best_step(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoint-1").mkdir(parents=True)
    (run_dir / "checkpoint-1" / "weights.txt").write_text("one", encoding="utf-8")
    (run_dir / "checkpoint-2").mkdir()
    (run_dir / "checkpoint-2" / "weights.txt").write_text("two", encoding="utf-8")
    (run_dir / "quality_metrics.jsonl").write_text(
        '{"step":1,"accuracy":0.2}\n{"step":2,"accuracy":0.8}\n',
        encoding="utf-8",
    )

    result = select_best_checkpoint(run_dir, CheckpointSelectionRule(metric="accuracy"))

    assert result is not None
    assert result["step"] == 2
    assert result["metric_value"] == 0.8
    assert (run_dir / "checkpoint-best" / "weights.txt").read_text(encoding="utf-8") == "two"


def test_orchestrator_runs_and_skips_completed_manifest(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "smoke"
    manifest = tmp_path / "suite.yaml"
    script = (
        "from pathlib import Path; "
        f"root=Path({str(run_dir)!r}); "
        "(root/'predictions').mkdir(parents=True, exist_ok=True); "
        "(root/'predictions'/'final.jsonl').write_text('{\"id\":\"x\",\"output\":\"yes\"}\\n'); "
        "(root/'eval_metrics.jsonl').write_text('{\"step\":1,\"accuracy\":1.0}\\n')"
    )
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "runs": [
                    {
                        "name": "smoke",
                        "kind": "eval",
                        "command": ["python", "-c", script],
                        "output_dir": str(run_dir),
                        "artifacts": ["predictions/final.jsonl", "eval_metrics.jsonl"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    import scripts.experiments.run_experiment_suite as run_suite

    monkeypatch.setattr(run_suite, "scan_stop_signals", lambda _run_dir: [])
    run_suite_main([str(manifest), "--runs-dir", str(tmp_path / "runs")])
    run_suite_main([str(manifest), "--runs-dir", str(tmp_path / "runs")])

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "skipped"
    artifacts = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
    assert any(item["kind"] == "prediction" for item in artifacts["artifacts"])


def test_orchestrator_syncs_completed_runs_to_hf_bucket(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "smoke"
    manifest = ExperimentSuiteManifest.model_validate(
        {
            "version": 1,
            "defaults": {
                "upload": {
                    "hf_bucket": "hf://buckets/example/artifacts",
                    "prefix": "nightly",
                }
            },
            "runs": [
                {
                    "name": "smoke",
                    "kind": "eval",
                    "command": [
                        "python",
                        "-c",
                        f"from pathlib import Path; Path({str(run_dir)!r}, 'metric.jsonl').write_text('{{}}\\n')",
                    ],
                    "output_dir": str(run_dir),
                    "artifacts": ["metric.jsonl"],
                }
            ],
        }
    )
    calls = []

    monkeypatch.setattr("intersectionqa.experiment_runner.scan_stop_signals", lambda _run_dir: [])
    runner = ExperimentRunner(
        manifest,
        options=ExperimentRunnerOptions(
            runs_dir=tmp_path / "runs",
            stop_signal_scanner=lambda _run_dir: [],
            bucket_syncer=lambda source, target: calls.append((source, target)) or {"uploads": 1},
        ),
    )

    summary = runner.run()

    assert summary[0]["status"] == "success"
    assert calls == [
        (run_dir, "hf://buckets/example/artifacts/nightly/runs/smoke"),
        (run_dir, "hf://buckets/example/artifacts/nightly/runs/smoke"),
    ]
    upload = json.loads((run_dir / "bucket_upload.json").read_text(encoding="utf-8"))
    assert upload["status"] == "uploaded"
    artifacts = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
    assert any(item["role"] == "hf_bucket" for item in artifacts["artifacts"])


def test_orchestrator_selects_dependency_windows(tmp_path):
    manifest = ExperimentSuiteManifest.model_validate(
        {
            "runs": [
                {"name": "dataset", "kind": "dataset_report"},
                {"name": "sft", "kind": "sft", "depends_on": ["dataset"]},
                {"name": "eval", "kind": "eval", "depends_on": ["sft"]},
                {"name": "report", "kind": "analysis", "depends_on": ["eval"]},
            ]
        }
    )

    runner = ExperimentRunner(
        manifest,
        options=ExperimentRunnerOptions(
            runs_dir=tmp_path / "runs",
            dry_run=True,
            selection=ExperimentSelection(start_from="sft", run_until="eval"),
        ),
    )

    assert [run.name for run in runner.selected_runs()] == ["sft", "eval"]


def test_orchestrator_can_include_named_run_dependencies(tmp_path):
    manifest = ExperimentSuiteManifest.model_validate(
        {
            "runs": [
                {"name": "dataset", "kind": "dataset_report"},
                {"name": "sft", "kind": "sft", "depends_on": ["dataset"]},
                {"name": "eval", "kind": "eval", "depends_on": ["sft"]},
            ]
        }
    )

    runner = ExperimentRunner(
        manifest,
        options=ExperimentRunnerOptions(
            runs_dir=tmp_path / "runs",
            dry_run=True,
            selection=ExperimentSelection(names=("eval",), with_dependencies=True),
        ),
    )

    assert [run.name for run in runner.selected_runs()] == ["dataset", "sft", "eval"]


def test_command_expansion_can_reference_prior_run_outputs(tmp_path):
    manifest = ExperimentSuiteManifest.model_validate(
        {
            "runs": [
                {"name": "sft", "kind": "sft", "output_dir": str(tmp_path / "runs" / "sft")},
                {
                    "name": "eval",
                    "kind": "eval",
                    "command": ["python", "-m", "x", "--adapter", "{run:sft}/adapter", "--out", "{run_dir}"],
                },
            ]
        }
    )
    runner = ExperimentRunner(manifest, options=ExperimentRunnerOptions(runs_dir=tmp_path / "runs"))
    context = runner._context(manifest.runs[1])

    assert expand_command(manifest.runs[1].command, context) == [
        "python",
        "-m",
        "x",
        "--adapter",
        str(tmp_path / "runs" / "sft" / "adapter"),
        "--out",
        str(tmp_path / "runs" / "eval"),
    ]
