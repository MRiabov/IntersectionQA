import json
import subprocess
import sys


def test_build_release_candidate_writes_reports_and_parquet(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "smoke:",
                "  include_cadevolve_if_available: false",
                "  use_synthetic_fixtures: true",
                "  geometry_limit: 3",
                "  task_types:",
                "    - binary_interference",
                "    - repair_direction",
                "    - repair_translation",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "release_candidate"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.dataset.build_release_candidate",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output_dir / "parquet_manifest.json").exists()
    assert (output_dir / "parquet" / "train.parquet").exists()
    assert (output_dir / "reports" / "dataset_stats.json").exists()
    assert (output_dir / "reports" / "aabb_baseline.json").exists()
    assert (output_dir / "reports" / "obb_baseline.json").exists()
    assert (output_dir / "reports" / "tool_assisted_upper_bound.json").exists()
    assert (output_dir / "reports" / "edit_verifier.json").exists()
    assert (output_dir / "reports" / "repair_verifier.json").exists()
    assert (output_dir / "reports" / "failure_analysis.json").exists()
    report = json.loads((output_dir / "reports" / "release_candidate_report.json").read_text())
    metadata = json.loads((output_dir / "metadata.json").read_text())
    tool_report = json.loads((output_dir / "reports" / "tool_assisted_upper_bound.json").read_text())
    repair_report = json.loads((output_dir / "reports" / "repair_verifier.json").read_text())
    edit_report = json.loads((output_dir / "reports" / "edit_verifier.json").read_text())
    comparison_report = json.loads((output_dir / "reports" / "baseline_comparison.json").read_text())
    failure_report = json.loads((output_dir / "reports" / "failure_analysis.json").read_text())
    stats_report = json.loads((output_dir / "reports" / "dataset_stats.json").read_text())

    assert report["validated_rows"] > 0
    assert report["parquet_dir"] == str(output_dir / "parquet")
    assert report["repair_verifier"]["repair_success_rate"] == 1.0
    assert report["edit_verifier"]["repair_success_rate"] == 1.0
    assert "repair_direction" in metadata["task_types"]
    assert "repair_translation" in metadata["task_types"]
    assert metadata["counts"]["by_task"]["repair_direction"] > 0
    assert metadata["counts"]["by_task"]["repair_translation"] > 0
    assert tool_report["tool_failure_count"] == 0
    assert any(metric["task_type"] == "repair_direction" for metric in tool_report["metrics"])
    assert repair_report["report"]["row_count"] == metadata["counts"]["by_task"]["repair_direction"]
    assert repair_report["report"]["repair_success_rate"] == 1.0
    assert edit_report["report"]["row_count"] == (
        metadata["counts"]["by_task"]["repair_direction"]
        + metadata["counts"]["by_task"]["repair_translation"]
    )
    assert edit_report["report"]["repair_success_rate"] == 1.0
    assert failure_report["repair_prediction_verifier"]["repair_success_rate"] == 1.0
    assert stats_report["repair_direction"]["row_count"] == metadata["counts"]["by_task"]["repair_direction"]
    assert any(
        row["system"] == "tool_assisted_repair_verifier"
        and row["task_type"] == "repair_direction"
        and row["accuracy"] == 1.0
        for row in comparison_report
    )
