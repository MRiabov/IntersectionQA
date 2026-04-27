import subprocess
import sys

from intersectionqa.config import DatasetConfig, SmokeConfig
from intersectionqa.enums import TaskType
from intersectionqa.export.dataset_card import build_dataset_card
from intersectionqa.pipeline import write_smoke_dataset


def test_smoke_export_writes_dataset_card(tmp_path):
    write_smoke_dataset(DatasetConfig(output_dir=tmp_path))

    path = tmp_path / "DATASET_CARD.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    for heading in [
        "## Motivation",
        "## Task Descriptions",
        "## Label Definitions",
        "## Data Sources",
        "## Generation Process",
        "## Splits",
        "## Known Limitations",
        "## Intended Uses",
        "## Out-of-Scope Uses",
        "## License",
    ]:
        assert heading in text


def test_dataset_card_cli_writes_custom_path(tmp_path):
    write_smoke_dataset(DatasetConfig(output_dir=tmp_path / "dataset"))
    output = tmp_path / "CARD.md"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.dataset.write_dataset_card",
            str(tmp_path / "dataset"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    assert build_dataset_card(tmp_path / "dataset") == output.read_text(encoding="utf-8")


def test_dataset_card_describes_opt_in_repair_direction(tmp_path):
    write_smoke_dataset(
        DatasetConfig(
            output_dir=tmp_path,
            smoke=SmokeConfig(
                include_cadevolve_if_available=False,
                task_types=[
                    TaskType.BINARY_INTERFERENCE,
                    TaskType.REPAIR_DIRECTION,
                    TaskType.REPAIR_TRANSLATION,
                ],
            ),
        )
    )

    text = build_dataset_card(tmp_path)

    assert "- intersectionedit" in text
    assert "- repair-direction" in text
    assert "- repair-translation" in text
    assert "`repair_direction` is an opt-in IntersectionEdit task" in text
    assert "`repair_translation` is an opt-in IntersectionEdit task" in text
    assert "conservative AABB-separating" in text
