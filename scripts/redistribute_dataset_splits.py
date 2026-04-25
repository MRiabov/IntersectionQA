"""Reassign exported dataset rows to the current group-safe split policy."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from intersectionqa.export.balance import balance_dataset_dir
from intersectionqa.export.dataset_card import write_dataset_card
from intersectionqa.export.jsonl import (
    build_metadata,
    read_jsonl,
    read_metadata,
    validate_rows,
    write_metadata,
    write_split_files,
)
from intersectionqa.export.parquet import write_parquet_files
from intersectionqa.splits.grouped import (
    ALL_SPLITS,
    audit_group_leakage,
    reassign_public_row_splits,
    split_manifest,
    split_names_for_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/IntersectionQA-90K"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balance-classes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = redistribute_dataset_splits(
        args.dataset_dir,
        seed=args.seed,
        backup=args.backup,
        balance_classes=args.balance_classes,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def redistribute_dataset_splits(
    dataset_dir: Path,
    *,
    seed: int,
    backup: bool,
    balance_classes: bool,
    dry_run: bool,
) -> dict[str, object]:
    metadata = read_metadata(dataset_dir / "metadata.json")
    if metadata is None:
        raise ValueError(f"missing metadata.json in {dataset_dir}")

    rows = []
    for split in ALL_SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        if path.exists():
            rows.extend(read_jsonl(path))

    reassigned_rows, report = reassign_public_row_splits(rows, seed)
    validate_rows(reassigned_rows)
    audit = audit_group_leakage(reassigned_rows)
    if audit.status != "pass":
        raise ValueError(f"group leakage audit failed after split redistribution: {audit.violations}")

    report = {
        **report,
        "dataset_dir": str(dataset_dir),
        "old_row_count": len(rows),
        "new_row_count": len(reassigned_rows),
        "leakage_audit_status": audit.status,
        "class_balance_requested": balance_classes,
        "dry_run": dry_run,
    }
    if dry_run:
        return report

    if backup:
        backup_dir = dataset_dir / "pre_split_redistribution_backup"
        backup_dir.mkdir(exist_ok=True)
        for name in [
            *[f"{split}.jsonl" for split in split_names_for_rows(rows)],
            "metadata.json",
            "split_manifest.json",
            "parquet_manifest.json",
            "DATASET_CARD.md",
            "README.md",
            "class_balance_report.json",
        ]:
            path = dataset_dir / name
            if path.exists():
                shutil.copy2(path, backup_dir / name)

    write_dataset_files(dataset_dir, metadata, reassigned_rows)
    (dataset_dir / "split_redistribution_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if balance_classes:
        report["class_balance"] = balance_dataset_dir(dataset_dir, backup=backup)
    return report


def write_dataset_files(dataset_dir: Path, metadata, rows) -> None:
    split_summary = write_split_files(rows, dataset_dir)
    (dataset_dir / "split_manifest.json").write_text(
        split_manifest(rows).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    write_metadata(
        build_metadata(
            dataset_version=metadata.dataset_version,
            config_hash=metadata.config_hash,
            source_manifest_hash=metadata.source_manifest_hash,
            label_policy=metadata.label_policy,
            splits=split_summary,
            rows=rows,
            license=metadata.license,
        ),
        dataset_dir / "metadata.json",
    )
    parquet_counts = write_parquet_files(rows, dataset_dir / "parquet")
    (dataset_dir / "parquet_manifest.json").write_text(
        json.dumps({"files": parquet_counts, "compression": "zstd"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_dataset_card(dataset_dir)


if __name__ == "__main__":
    main()
