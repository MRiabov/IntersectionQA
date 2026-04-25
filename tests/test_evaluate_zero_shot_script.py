import os
import subprocess
import sys

from intersectionqa.config import DatasetConfig, SmokeConfig
from intersectionqa.enums import TaskType
from intersectionqa.pipeline import write_smoke_dataset
from scripts.evaluate_zero_shot import _load_env_file


def test_load_env_file_preserves_existing_values(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_API_KEY=from_file\n"
        "export HF_TOKEN='hf_from_file'\n"
        'CUSTOM_VALUE="quoted value"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "from_shell")

    _load_env_file(path)

    assert os.environ["OPENAI_API_KEY"] == "from_shell"
    assert os.environ["HF_TOKEN"] == "hf_from_file"
    assert os.environ["CUSTOM_VALUE"] == "quoted value"


def test_evaluate_zero_shot_exports_few_shot_requests(tmp_path):
    dataset_dir = tmp_path / "dataset"
    write_smoke_dataset(
        DatasetConfig(
            output_dir=dataset_dir,
            smoke=SmokeConfig(use_synthetic_fixtures=True, include_cadevolve_if_available=False),
        )
    )
    requests_path = tmp_path / "fewshot_requests.jsonl"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_zero_shot",
            str(dataset_dir),
            "--model",
            "frontier-test",
            "--few-shot-count",
            "1",
            "--task-type",
            "binary_interference",
            "--limit",
            "1",
            "--requests-jsonl",
            str(requests_path),
            "--export-requests-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    text = requests_path.read_text(encoding="utf-8")
    assert "closed_book_few_shot_v01" in text
    assert "few_shot_example_ids" in text


def test_evaluate_zero_shot_exports_repair_direction_requests(tmp_path):
    dataset_dir = tmp_path / "dataset"
    write_smoke_dataset(
        DatasetConfig(
            output_dir=dataset_dir,
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                task_types=[TaskType.REPAIR_DIRECTION],
            ),
        )
    )
    requests_path = tmp_path / "repair_requests.jsonl"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_zero_shot",
            str(dataset_dir),
            "--model",
            "frontier-test",
            "--task-type",
            "repair_direction",
            "--limit",
            "1",
            "--requests-jsonl",
            str(requests_path),
            "--export-requests-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    text = requests_path.read_text(encoding="utf-8")
    assert "closed_book_zero_shot_v01" in text
    assert "repair_direction" in text
    assert "IntersectionQA/IntersectionEdit" in text
