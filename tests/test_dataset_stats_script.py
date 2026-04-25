import json
import subprocess
import sys

from intersectionqa.config import DatasetConfig, SmokeConfig
from intersectionqa.enums import TaskType
from intersectionqa.pipeline import validate_dataset_dir, write_smoke_dataset


def test_dataset_stats_script_reports_repair_direction_summary(tmp_path):
    dataset_dir = tmp_path / "dataset"
    write_smoke_dataset(
        DatasetConfig(
            output_dir=dataset_dir,
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                task_types=[TaskType.REPAIR_DIRECTION, TaskType.REPAIR_TRANSLATION],
            ),
        )
    )
    rows = validate_dataset_dir(dataset_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.dataset_stats",
            str(dataset_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stats = json.loads(completed.stdout)

    direction_rows = [row for row in rows if row.task_type == TaskType.REPAIR_DIRECTION]
    translation_rows = [row for row in rows if row.task_type == TaskType.REPAIR_TRANSLATION]
    assert stats["repair_direction"]["row_count"] == len(direction_rows)
    assert stats["repair_direction"]["by_policy"] == {
        "conservative_aabb_separating_translation_v01": len(direction_rows)
    }
    assert stats["repair_direction"]["selected_magnitude_mm"]["min"] is not None
    assert stats["repair_translation"]["row_count"] == len(translation_rows)
    assert stats["repair_translation"]["selected_magnitude_mm"]["min"] is not None
