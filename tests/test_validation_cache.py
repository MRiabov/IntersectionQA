from pathlib import Path

from intersectionqa.config import DatasetConfig, SmokeConfig
from intersectionqa.pipeline import build_smoke_rows
from intersectionqa.sources.synthetic import synthetic_source_object
from intersectionqa.sources.validation import validate_source_object
from intersectionqa.sources.validation_cache import ObjectValidationCache, object_validation_cache_key


def test_object_validation_cache_round_trips_record(tmp_path):
    config = DatasetConfig()
    source = synthetic_source_object("obj_cache", "object_a", (1.0, 2.0, 3.0))
    record = validate_source_object(
        source,
        config_hash=config.config_hash,
        validated_at_version="test",
        isolated=False,
    )
    key = object_validation_cache_key(
        source,
        timeout_seconds=5.0,
        validated_at_version="test",
    )
    cache = ObjectValidationCache(tmp_path)
    cache.set(key, record)

    loaded = cache.get(key)

    assert loaded is not None
    assert loaded.object_id == record.object_id
    assert loaded.valid is True
    assert loaded.volume == record.volume


def test_smoke_generation_reuses_object_validation_cache(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    config = DatasetConfig(
        smoke=SmokeConfig(
            use_synthetic_fixtures=True,
            include_cadevolve_if_available=False,
            object_validation_cache_dir=cache_dir,
        )
    )

    build_smoke_rows(config)
    first = capsys.readouterr().err
    build_smoke_rows(config)
    second = capsys.readouterr().err

    assert "cache_hits=0" in first
    assert "cache_hits=3" in second
    assert list(Path(cache_dir).glob("*/*.json"))
