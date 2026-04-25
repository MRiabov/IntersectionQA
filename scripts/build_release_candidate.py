"""Build a release-candidate dataset plus QA reports and Parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

from intersectionqa.config import load_config
from intersectionqa.evaluation.aabb import evaluate_aabb_binary
from intersectionqa.evaluation.comparison import (
    comparison_rows_from_aabb,
    comparison_rows_from_metrics,
    comparison_rows_to_markdown,
)
from intersectionqa.evaluation.obb import evaluate_obb_binary
from intersectionqa.evaluation.tool_assisted import run_tool_assisted_upper_bound
from intersectionqa.evaluation.failure_analysis import failure_case_analysis
from intersectionqa.evaluation.metrics import dataset_stats, manifest_stats
from intersectionqa.export.balance import balance_dataset_dir
from intersectionqa.export.jsonl import read_failure_manifest, read_object_validation_manifest
from intersectionqa.export.parquet import write_parquet_files
from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import validate_dataset_dir, write_smoke_dataset
from intersectionqa.sharding import generate_source_shards, merge_validated_source_shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cadevolve-archive",
        type=Path,
        default=None,
        help="DEPRECATED: bootstrap-only tar archive used to prepare an extracted source directory",
    )
    parser.add_argument("--cadevolve-source-dir", type=Path, default=None)
    parser.add_argument(
        "--cadevolve-source-cache-root",
        type=Path,
        default=None,
        help="DEPRECATED: alias for --cadevolve-source-dir",
    )
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--source-shard-size", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--balance-classes", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    configure_logging()
    started = monotonic()
    config = load_config(args.config)
    config.output_dir = args.output_dir
    if args.cadevolve_archive is not None:
        config.cadevolve_archive = args.cadevolve_archive
    if args.cadevolve_source_dir is not None:
        config.smoke.cadevolve_source_dir = args.cadevolve_source_dir
    if args.cadevolve_source_cache_root is not None:
        config.smoke.cadevolve_source_cache_root = args.cadevolve_source_cache_root

    build_report: dict[str, object]
    if args.shard_count is None:
        build_report = {"mode": "single", "smoke_report": write_smoke_dataset(config).model_dump(mode="json")}
        dataset_dir = args.output_dir
    else:
        if args.source_shard_size is None:
            raise ValueError("--source-shard-size is required with --shard-count")
        shard_root = args.output_dir / "source_shards"
        dataset_dir = args.output_dir / "dataset"
        shard_manifest = generate_source_shards(
            config,
            shard_root,
            shard_count=args.shard_count,
            source_shard_size=args.source_shard_size,
            force=args.force,
        )
        merge_manifest = merge_validated_source_shards(shard_root, dataset_dir, config=config)
        build_report = {"mode": "sharded", "shards": shard_manifest, "merge": merge_manifest}

    class_balance_report = None
    if args.balance_classes:
        class_balance_report = balance_dataset_dir(dataset_dir)
    rows = validate_dataset_dir(dataset_dir)
    parquet_counts = write_parquet_files(rows, dataset_dir / "parquet")
    _write_json({"files": parquet_counts, "compression": "zstd"}, dataset_dir / "parquet_manifest.json")
    reports_dir = args.output_dir / "reports" if args.shard_count is not None else dataset_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    object_validations = read_object_validation_manifest(dataset_dir / "object_validation_manifest.jsonl")
    failures = read_failure_manifest(dataset_dir / "failure_manifest.jsonl")

    stats = dataset_stats(rows)
    stats["manifests"] = manifest_stats(object_validations, failures)
    _write_json(stats, reports_dir / "dataset_stats.json")

    aabb = evaluate_aabb_binary(rows)
    _write_json(aabb.__dict__, reports_dir / "aabb_baseline.json")
    obb = evaluate_obb_binary(rows)
    _write_json(obb.__dict__, reports_dir / "obb_baseline.json")
    tool_assisted = run_tool_assisted_upper_bound(rows)
    _write_json(tool_assisted.report, reports_dir / "tool_assisted_upper_bound.json")

    failures_report = failure_case_analysis(rows, object_validations, failures)
    _write_json(failures_report, reports_dir / "failure_analysis.json")

    comparison_rows = comparison_rows_from_aabb(aabb)
    comparison_rows.extend(comparison_rows_from_aabb(obb, system="obb_overlap"))
    comparison_rows.extend(comparison_rows_from_metrics(tool_assisted.metrics, system="tool_assisted_upper_bound"))
    _write_json([row.as_dict() for row in comparison_rows], reports_dir / "baseline_comparison.json")
    (reports_dir / "baseline_comparison.md").write_text(
        comparison_rows_to_markdown(comparison_rows),
        encoding="utf-8",
    )

    release_report = {
        "dataset_dir": str(dataset_dir),
        "reports_dir": str(reports_dir),
        "elapsed_seconds": round(monotonic() - started, 3),
        "build": build_report,
        "class_balance": class_balance_report,
        "validated_rows": len(rows),
        "parquet_dir": str(dataset_dir / "parquet"),
    }
    _write_json(release_report, reports_dir / "release_candidate_report.json")
    print(json.dumps(release_report, indent=2, sort_keys=True))


def _write_json(value: object, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
