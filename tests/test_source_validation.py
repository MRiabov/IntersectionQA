from intersectionqa.config import DatasetConfig
from intersectionqa.geometry.cadquery_exec import cadquery_available
from intersectionqa.sources.synthetic import synthetic_source_object
from intersectionqa.sources.validation import validate_source_object


def test_synthetic_source_validation_is_analytic():
    config = DatasetConfig()
    source = synthetic_source_object("obj_test", "object_a", (2.0, 3.0, 4.0))
    validation = validate_source_object(
        source,
        config_hash=config.config_hash,
        validated_at_version="test",
    )
    assert validation.valid is True
    assert validation.volume == 24.0
    assert validation.bbox is not None
    assert validation.failure_reason is None


def test_invalid_synthetic_dimensions_report_volume_failure():
    config = DatasetConfig()
    source = synthetic_source_object("obj_test", "object_a", (0.0, 3.0, 4.0))
    validation = validate_source_object(
        source,
        config_hash=config.config_hash,
        validated_at_version="test",
    )
    assert validation.valid is False
    assert validation.failure_reason == "zero_or_negative_volume"


def test_cadquery_availability_check_returns_bool():
    assert isinstance(cadquery_available(), bool)
