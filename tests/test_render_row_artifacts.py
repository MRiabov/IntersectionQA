from pathlib import Path

from intersectionqa.config import DatasetConfig
from intersectionqa.pipeline import validate_dataset_dir, write_smoke_dataset
from scripts.dataset.render_row_artifacts import render_row_artifacts


def test_render_row_artifacts_writes_png_previews(tmp_path):
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "artifacts"
    write_smoke_dataset(DatasetConfig(output_dir=dataset_dir))
    rows = validate_dataset_dir(dataset_dir)
    row = next(item for item in rows if item.task_type == "binary_interference")

    manifest = render_row_artifacts(
        dataset_dir,
        row.id,
        output_dir,
        image_size=(320, 240),
        tessellation_tolerance=0.5,
    )

    assert manifest["renderer"] == "pyvista"
    for key in ["object_a", "object_b", "assembly", "contact_sheet", "render_manifest"]:
        assert Path(manifest["files"][key]).exists()
    assert Path(manifest["files"]["object_a"]).suffix == ".png"
