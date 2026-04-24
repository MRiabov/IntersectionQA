import subprocess
import sys

from intersectionqa.config import DatasetConfig
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
            "scripts.write_dataset_card",
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
