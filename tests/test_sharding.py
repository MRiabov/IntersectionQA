import json

from intersectionqa.config import DatasetConfig, SmokeConfig
from intersectionqa.sharding import (
    build_source_shard_specs,
    config_for_source_shard,
    generate_source_shards,
    merge_validated_source_shards,
)


def test_source_shard_specs_are_deterministic(tmp_path):
    specs = build_source_shard_specs(tmp_path, shard_count=3, source_shard_size=10)

    assert [spec.shard_id for spec in specs] == ["shard_0000", "shard_0001", "shard_0002"]
    assert [spec.source_offset for spec in specs] == [0, 10, 20]
    assert [spec.source_limit for spec in specs] == [10, 10, 10]
    assert specs[1].output_dir == tmp_path / "shards" / "shard_0001"


def test_config_for_source_shard_sets_output_and_source_window(tmp_path):
    config = DatasetConfig()
    spec = build_source_shard_specs(tmp_path, shard_count=1, source_shard_size=7)[0]

    shard_config = config_for_source_shard(config, spec)

    assert shard_config.output_dir == tmp_path / "shards" / "shard_0000"
    assert shard_config.smoke.object_validation_offset == 0
    assert shard_config.smoke.object_validation_limit == 7


def test_generate_source_shards_resumes_existing_valid_shards(tmp_path):
    config = DatasetConfig(
        smoke=SmokeConfig(
            use_synthetic_fixtures=True,
            include_cadevolve_if_available=False,
            geometry_limit=2,
        )
    )

    first = generate_source_shards(config, tmp_path, shard_count=2, source_shard_size=3)
    second = generate_source_shards(config, tmp_path, shard_count=2, source_shard_size=3)

    assert first["all_valid"] is True
    assert [shard["status"] for shard in first["shards"]] == [
        "generated_valid",
        "generated_valid",
    ]
    assert [shard["status"] for shard in second["shards"]] == [
        "skipped_existing_valid",
        "skipped_existing_valid",
    ]
    manifest = json.loads((tmp_path / "shard_manifest.json").read_text())
    assert manifest["all_valid"] is True
    assert manifest["pending_shards"] == []


def test_merge_validated_source_shards_rewrites_colliding_ids(tmp_path):
    shard_root = tmp_path / "sharded"
    merged_dir = tmp_path / "merged"
    config = DatasetConfig(
        smoke=SmokeConfig(
            use_synthetic_fixtures=True,
            include_cadevolve_if_available=False,
            geometry_limit=2,
        )
    )
    generate_source_shards(config, shard_root, shard_count=2, source_shard_size=3)

    result = merge_validated_source_shards(shard_root, merged_dir, config=config)

    assert result["row_count"] == 12
    assert result["merged_shards"] == ["shard_0000", "shard_0001"]
    ids = []
    geometry_ids = []
    for path in merged_dir.glob("*.jsonl"):
        if path.name.endswith("manifest.jsonl"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            import json

            row = json.loads(line)
            ids.append(row["id"])
            geometry_ids.extend(row["geometry_ids"])
            assert row["base_object_pair_id"].startswith("shard_")
            assert row["assembly_group_id"].startswith("shard_")
            assert row["metadata"]["source_shard_id"] in {"shard_0000", "shard_0001"}
    assert len(ids) == len(set(ids))
    assert all(geometry_id.startswith("shard_") for geometry_id in geometry_ids)
    assert (merged_dir / "merge_manifest.json").exists()
