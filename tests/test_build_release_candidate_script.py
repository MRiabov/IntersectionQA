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
            "scripts.build_release_candidate",
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
    assert (output_dir / "reports" / "failure_analysis.json").exists()
    report = json.loads((output_dir / "reports" / "release_candidate_report.json").read_text())
    assert report["validated_rows"] > 0
    assert report["parquet_dir"] == str(output_dir / "parquet")
