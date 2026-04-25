from pathlib import Path

from intersectionqa.config import DatasetConfig, SmokeConfig, load_config
from intersectionqa.enums import TaskType


def test_config_hash_ignores_local_output_and_cache_paths():
    base = DatasetConfig(
        cadevolve_archive=Path("/tmp/archive_a/cadevolve.tar"),
        output_dir=Path("/tmp/intersectionqa_a"),
        smoke=SmokeConfig(
            cadevolve_source_dir=Path("/tmp/source_a"),
            object_validation_cache_dir=Path("/tmp/cache_a/objects"),
            source_member_index_cache_dir=Path("/tmp/cache_a/indexes"),
            extracted_source_cache_dir=Path("/tmp/cache_a/sources"),
            cadevolve_source_cache_root=Path("/tmp/cache_a/sources/aa/hash"),
            geometry_label_cache_dir=Path("/tmp/cache_a/labels"),
            object_validation_worker_count=2,
            use_source_member_index_cache=True,
            use_extracted_source_cache=True,
            use_object_validation_cache=True,
            use_geometry_label_cache=True,
        ),
    )
    moved = DatasetConfig(
        cadevolve_archive=Path("/tmp/archive_b/cadevolve.tar"),
        output_dir=Path("/tmp/intersectionqa_b"),
        smoke=SmokeConfig(
            cadevolve_source_dir=Path("/tmp/source_b"),
            object_validation_cache_dir=Path("/tmp/cache_b/objects"),
            source_member_index_cache_dir=Path("/tmp/cache_b/indexes"),
            extracted_source_cache_dir=Path("/tmp/cache_b/sources"),
            cadevolve_source_cache_root=Path("/tmp/cache_b/sources/bb/hash"),
            geometry_label_cache_dir=Path("/tmp/cache_b/labels"),
            object_validation_worker_count=8,
            use_source_member_index_cache=False,
            use_extracted_source_cache=False,
            use_object_validation_cache=False,
            use_geometry_label_cache=False,
        ),
    )

    assert moved.config_hash == base.config_hash


def test_config_hash_tracks_generation_content_settings():
    base = DatasetConfig(smoke=SmokeConfig(geometry_limit=10))
    changed = DatasetConfig(smoke=SmokeConfig(geometry_limit=11))
    changed_balance = DatasetConfig(smoke=SmokeConfig(balance_geometry_relations=False))
    changed_pool = DatasetConfig(smoke=SmokeConfig(geometry_candidate_pool_multiplier=3))

    assert changed.config_hash != base.config_hash
    assert changed_balance.config_hash != base.config_hash
    assert changed_pool.config_hash != base.config_hash


def test_repair_smoke_config_is_opt_in_and_keeps_dataset_name():
    config = load_config(Path("configs/repair_smoke.yaml"))

    assert config.dataset_name == "IntersectionQA"
    assert config.dataset_version == "v0.1"
    assert config.smoke.task_types == [
        TaskType.BINARY_INTERFERENCE,
        TaskType.REPAIR_DIRECTION,
        TaskType.REPAIR_TRANSLATION,
    ]
    assert DatasetConfig().smoke.task_types == [
        TaskType.BINARY_INTERFERENCE,
        TaskType.RELATION_CLASSIFICATION,
        TaskType.VOLUME_BUCKET,
    ]
