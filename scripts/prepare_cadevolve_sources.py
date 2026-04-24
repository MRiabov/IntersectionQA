"""Preextract a bounded CADEvolve source subset into the local source cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

from intersectionqa.config import DatasetConfig, load_config
from intersectionqa.sources.cadevolve import CadevolveTarLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--cadevolve-archive", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config is not None else DatasetConfig()
    if args.cadevolve_archive is not None:
        config.cadevolve_archive = args.cadevolve_archive
    if args.cache_dir is not None:
        config.smoke.extracted_source_cache_dir = args.cache_dir

    started = monotonic()
    loader = CadevolveTarLoader(
        config.cadevolve_archive,
        config.config_hash,
        member_index_cache_dir=config.smoke.source_member_index_cache_dir
        if config.smoke.use_source_member_index_cache
        else None,
        extracted_source_cache_dir=config.smoke.extracted_source_cache_dir,
    )
    report = loader.prepare_extracted_sources(limit=args.limit, offset=args.offset)
    payload = {
        "cache_root": str(report.cache_root),
        "selected_count": report.selected_count,
        "newly_extracted_count": report.newly_extracted_count,
        "already_present_count": report.already_present_count,
        "selected_size_bytes": report.selected_size_bytes,
        "selected_size_mib": round(report.selected_size_bytes / 1024**2, 3),
        "complete": report.complete,
        "elapsed_seconds": round(monotonic() - started, 3),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
