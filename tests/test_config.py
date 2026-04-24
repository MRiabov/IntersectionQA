from pathlib import Path

from intersectionqa.config import DatasetConfig, SmokeConfig


def test_config_hash_ignores_local_output_and_cache_paths():
    base = DatasetConfig(
        output_dir=Path("/tmp/intersectionqa_a"),
        smoke=SmokeConfig(
            object_validation_cache_dir=Path("/tmp/cache_a/objects"),
            source_member_index_cache_dir=Path("/tmp/cache_a/indexes"),
            geometry_label_cache_dir=Path("/tmp/cache_a/labels"),
            object_validation_worker_count=2,
            use_source_member_index_cache=True,
            use_object_validation_cache=True,
            use_geometry_label_cache=True,
        ),
    )
    moved = DatasetConfig(
        output_dir=Path("/tmp/intersectionqa_b"),
        smoke=SmokeConfig(
            object_validation_cache_dir=Path("/tmp/cache_b/objects"),
            source_member_index_cache_dir=Path("/tmp/cache_b/indexes"),
            geometry_label_cache_dir=Path("/tmp/cache_b/labels"),
            object_validation_worker_count=8,
            use_source_member_index_cache=False,
            use_object_validation_cache=False,
            use_geometry_label_cache=False,
        ),
    )

    assert moved.config_hash == base.config_hash


def test_config_hash_tracks_generation_content_settings():
    base = DatasetConfig(smoke=SmokeConfig(geometry_limit=10))
    changed = DatasetConfig(smoke=SmokeConfig(geometry_limit=11))

    assert changed.config_hash != base.config_hash
