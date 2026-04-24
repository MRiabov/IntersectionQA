from pathlib import Path

from intersectionqa.config import DatasetConfig, SmokeConfig
from intersectionqa.pipeline import write_smoke_dataset
from scripts.audit_reproducibility import compare_dataset_dirs


def test_reproducibility_audit_passes_for_same_content_different_output_dirs(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    smoke = SmokeConfig(
        use_synthetic_fixtures=True,
        include_cadevolve_if_available=False,
        object_validation_cache_dir=tmp_path / "objects",
        geometry_label_cache_dir=tmp_path / "labels",
    )

    write_smoke_dataset(DatasetConfig(output_dir=left, smoke=smoke))
    write_smoke_dataset(DatasetConfig(output_dir=right, smoke=smoke))

    result = compare_dataset_dirs(left, right)

    assert result["status"] == "pass"
    assert result["mismatches"] == []


def test_reproducibility_audit_reports_file_mismatch(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    config = DatasetConfig(
        smoke=SmokeConfig(use_synthetic_fixtures=True, include_cadevolve_if_available=False)
    )
    write_smoke_dataset(config.model_copy(update={"output_dir": left}))
    write_smoke_dataset(config.model_copy(update={"output_dir": right}))
    with (right / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    result = compare_dataset_dirs(left, right)

    assert result["status"] == "fail"
    assert any(mismatch["artifact"] == "train.jsonl" for mismatch in result["mismatches"])
