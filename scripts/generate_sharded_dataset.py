"""Generate resumable source shards for the smoke dataset pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.config import load_config
from intersectionqa.logging import configure_logging
from intersectionqa.sharding import generate_source_shards, merge_validated_source_shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cadevolve-archive", type=Path, default=None)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--source-shard-size", type=int, required=True)
    parser.add_argument(
        "--merge-output-dir",
        type=Path,
        default=None,
        help="write a merged validated dataset after shard generation",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    if args.cadevolve_archive is not None:
        config.cadevolve_archive = args.cadevolve_archive
    manifest = generate_source_shards(
        config,
        args.output_dir,
        shard_count=args.shard_count,
        source_shard_size=args.source_shard_size,
        force=args.force,
    )
    if args.merge_output_dir is not None:
        manifest["merge"] = merge_validated_source_shards(
            args.output_dir,
            args.merge_output_dir,
            config=config,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
