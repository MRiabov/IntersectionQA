import json

from intersectionqa.config import DatasetConfig
from intersectionqa.pipeline import validate_dataset_dir, write_smoke_dataset
from scripts.dataset.export_row_artifacts import export_row_artifacts


def test_export_row_artifacts_writes_prompt_script_and_step_files(tmp_path):
    dataset_dir = tmp_path / "dataset"
    artifact_dir = tmp_path / "artifacts"
    write_smoke_dataset(DatasetConfig(output_dir=dataset_dir))
    row = validate_dataset_dir(dataset_dir)[0]

    manifest = export_row_artifacts(dataset_dir, row.id, artifact_dir)
    row_dir = artifact_dir / row.id

    assert manifest["row_id"] == row.id
    assert manifest["step_export_status"] in {"ok", "partial"}
    assert (row_dir / "prompt.txt").read_text(encoding="utf-8") == row.prompt
    assert (row_dir / "assembly.py").read_text(encoding="utf-8") == row.script
    assert (row_dir / "object_a.step").exists()
    assert (row_dir / "object_b.step").exists()
    assert (row_dir / "assembly.step").exists()
    assert json.loads((row_dir / "row.json").read_text(encoding="utf-8"))["id"] == row.id


def test_export_row_artifacts_can_skip_step_execution(tmp_path):
    dataset_dir = tmp_path / "dataset"
    artifact_dir = tmp_path / "artifacts"
    write_smoke_dataset(DatasetConfig(output_dir=dataset_dir))
    row = validate_dataset_dir(dataset_dir)[0]

    manifest = export_row_artifacts(dataset_dir, row.id, artifact_dir, write_step=False)

    assert manifest["step_export_status"] == "skipped"
    assert not (artifact_dir / row.id / "assembly.step").exists()
