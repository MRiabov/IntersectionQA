import pytest

from intersectionqa.config import DatasetConfig
from intersectionqa.geometry.cadquery_exec import cadquery_available
from intersectionqa.sources.synthetic import synthetic_source_object
from intersectionqa.sources.validation import validate_source_object, validate_source_objects_bounded


def test_synthetic_source_validation_uses_cadquery():
    config = DatasetConfig()
    source = synthetic_source_object("obj_test", "object_a", (2.0, 3.0, 4.0))
    validation = validate_source_object(
        source,
        config_hash=config.config_hash,
        validated_at_version="test",
    )
    assert validation.valid is True
    assert validation.volume == pytest.approx(24.0)
    assert validation.bbox is not None
    assert validation.failure_reason is None
    assert validation.cadquery_version is not None
    assert validation.ocp_version is not None


def test_invalid_synthetic_dimensions_report_volume_failure():
    config = DatasetConfig()
    source = synthetic_source_object("obj_test", "object_a", (0.0, 3.0, 4.0))
    validation = validate_source_object(
        source,
        config_hash=config.config_hash,
        validated_at_version="test",
    )
    assert validation.valid is False
    assert validation.failure_reason == "source_exec_error"


def test_cadquery_availability_check_returns_bool():
    assert isinstance(cadquery_available(), bool)


def test_isolated_validation_times_out_slow_source():
    config = DatasetConfig()
    source = synthetic_source_object("obj_slow", "object_a", (1.0, 1.0, 1.0))
    source.normalized_code = "\n".join(
        [
            "def object_a():",
            "    import time",
            "    time.sleep(2.0)",
            "    return cq.Workplane('XY').box(1.0, 1.0, 1.0)",
            "",
        ]
    )
    validation = validate_source_object(
        source,
        config_hash=config.config_hash,
        validated_at_version="test",
        timeout_seconds=0.1,
        isolated=True,
    )
    assert validation.valid is False
    assert validation.failure_reason == "timeout"


def test_bounded_source_validation_preserves_input_order():
    config = DatasetConfig()
    sources = [
        synthetic_source_object("obj_a", "object_a", (1.0, 1.0, 1.0)),
        synthetic_source_object("obj_b", "object_a", (2.0, 1.0, 1.0)),
        synthetic_source_object("obj_bad", "object_a", (0.0, 1.0, 1.0)),
    ]

    validations = validate_source_objects_bounded(
        sources,
        config_hash=config.config_hash,
        validated_at_version="test",
        timeout_seconds=15.0,
        worker_count=2,
    )

    assert [validation.object_id for validation in validations] == [
        "obj_a",
        "obj_b",
        "obj_bad",
    ]
    assert [validation.valid for validation in validations] == [True, True, False]


def test_bounded_source_validation_reports_each_completed_result():
    config = DatasetConfig()
    sources = [
        synthetic_source_object("obj_a", "object_a", (1.0, 1.0, 1.0)),
        synthetic_source_object("obj_b", "object_a", (2.0, 1.0, 1.0)),
        synthetic_source_object("obj_bad", "object_a", (0.0, 1.0, 1.0)),
    ]
    completed: list[tuple[int, str, str]] = []

    validations = validate_source_objects_bounded(
        sources,
        config_hash=config.config_hash,
        validated_at_version="test",
        timeout_seconds=15.0,
        worker_count=2,
        on_result=lambda index, source, record: completed.append(
            (index, source.object_id, record.object_id)
        ),
    )

    assert sorted(completed) == [
        (0, "obj_a", "obj_a"),
        (1, "obj_b", "obj_b"),
        (2, "obj_bad", "obj_bad"),
    ]
    assert [validation.object_id for validation in validations] == [
        "obj_a",
        "obj_b",
        "obj_bad",
    ]
