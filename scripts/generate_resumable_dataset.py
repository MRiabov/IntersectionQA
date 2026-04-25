"""Generate and merge a resumable multi-shard dataset build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.config import load_config
from intersectionqa.logging import configure_logging
from intersectionqa.sharding import (
    SourceShardSpec,
    generate_source_shards_from_specs,
    merge_validated_source_shards,
    seed_existing_dataset_shard,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merge-output-dir", type=Path, required=True)
    parser.add_argument("--seed-existing-dataset", type=Path, default=None)
    parser.add_argument("--seed-source-offset", type=int, default=0)
    parser.add_argument("--seed-source-limit", type=int, default=0)
    parser.add_argument("--additional-shard-count", type=int, required=True)
    parser.add_argument("--additional-source-offset", type=int, required=True)
    parser.add_argument("--additional-source-shard-size", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    specs = _build_specs(
        args.output_dir,
        seed_source_offset=args.seed_source_offset,
        seed_source_limit=args.seed_source_limit,
        additional_shard_count=args.additional_shard_count,
        additional_source_offset=args.additional_source_offset,
        additional_source_shard_size=args.additional_source_shard_size,
        include_seed=args.seed_existing_dataset is not None,
    )
    seed_result = None
    if args.seed_existing_dataset is not None:
        if args.seed_source_limit <= 0:
            raise ValueError("--seed-source-limit is required with --seed-existing-dataset")
        seed_result = seed_existing_dataset_shard(
            args.output_dir,
            specs[0],
            args.seed_existing_dataset,
            force=args.force,
        )

    shard_manifest = generate_source_shards_from_specs(
        config,
        args.output_dir,
        specs,
        force=False,
    )
    merge_manifest = merge_validated_source_shards(
        args.output_dir,
        args.merge_output_dir,
        config=config,
    )
    print(
        json.dumps(
            {
                "schema": "intersectionqa_resumable_dataset_job_v1",
                "seed": seed_result.model_dump(mode="json") if seed_result else None,
                "shards": shard_manifest,
                "merge": merge_manifest,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _build_specs(
    output_dir: Path,
    *,
    seed_source_offset: int,
    seed_source_limit: int,
    additional_shard_count: int,
    additional_source_offset: int,
    additional_source_shard_size: int,
    include_seed: bool,
) -> list[SourceShardSpec]:
    if additional_shard_count < 0:
        raise ValueError("--additional-shard-count must be non-negative")
    if additional_source_shard_size <= 0:
        raise ValueError("--additional-source-shard-size must be positive")
    specs: list[SourceShardSpec] = []
    next_index = 0
    if include_seed:
        specs.append(
            SourceShardSpec(
                shard_id="shard_0000",
                index=0,
                source_offset=seed_source_offset,
                source_limit=seed_source_limit,
                output_dir=output_dir / "shards" / "shard_0000",
            )
        )
        next_index = 1
    for index in range(additional_shard_count):
        shard_index = next_index + index
        specs.append(
            SourceShardSpec(
                shard_id=f"shard_{shard_index:04d}",
                index=shard_index,
                source_offset=additional_source_offset + index * additional_source_shard_size,
                source_limit=additional_source_shard_size,
                output_dir=output_dir / "shards" / f"shard_{shard_index:04d}",
            )
        )
    if not specs:
        raise ValueError("at least one shard is required")
    return specs


if __name__ == "__main__":
    main()
