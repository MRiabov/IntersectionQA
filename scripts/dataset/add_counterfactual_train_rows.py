"""Move a deterministic subset of counterfactual groups into the train split."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

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
from intersectionqa.schema import PublicTaskRow
from intersectionqa.splits.grouped import (
    ALL_SPLITS,
    audit_group_leakage,
    split_manifest,
    split_names_for_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/IntersectionQA-90K"))
    parser.add_argument("--source-split", default="test_near_boundary")
    parser.add_argument("--target-split", default="train")
    parser.add_argument("--train-fraction", type=float, default=0.25)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0 < args.train_fraction <= 1:
        raise ValueError("--train-fraction must be in (0, 1]")

    dataset_dir = args.dataset_dir
    rows = [
        row
        for split in ALL_SPLITS
        if (dataset_dir / f"{split}.jsonl").exists()
        for row in read_jsonl(dataset_dir / f"{split}.jsonl")
    ]
    metadata = read_metadata(dataset_dir / "metadata.json")
    if metadata is None:
        raise ValueError(f"missing metadata.json in {dataset_dir}")

    source_rows = [
        row
        for row in rows
        if row.split == args.source_split and row.counterfactual_group_id is not None
    ]
    forbidden_splits = set(ALL_SPLITS) - {args.source_split, args.target_split}
    forbidden_base_object_pair_ids = {
        row.base_object_pair_id for row in rows if row.split in forbidden_splits
    }
    forbidden_assembly_group_ids = {
        row.assembly_group_id for row in rows if row.split in forbidden_splits
    }
    by_group: dict[str, list[PublicTaskRow]] = defaultdict(list)
    for row in source_rows:
        by_group[row.counterfactual_group_id or ""].append(row)
    eligible_groups = [
        group_id
        for group_id, group_rows in by_group.items()
        if not ({row.base_object_pair_id for row in group_rows} & forbidden_base_object_pair_ids)
        and not ({row.assembly_group_id for row in group_rows} & forbidden_assembly_group_ids)
    ]
    selected_groups = select_groups(sorted(eligible_groups), fraction=args.train_fraction, seed=args.seed)
    if args.max_groups is not None:
        selected_groups = selected_groups[: args.max_groups]
    selected_set = set(selected_groups)

    moved_rows: list[PublicTaskRow] = []
    updated_rows: list[PublicTaskRow] = []
    for row in rows:
        if row.split == args.source_split and row.counterfactual_group_id in selected_set:
            moved = row.model_copy(update={"split": args.target_split})
            moved_rows.append(moved)
            updated_rows.append(moved)
        else:
            updated_rows.append(row)

    validate_rows(updated_rows)
    audit = audit_group_leakage(updated_rows)
    if audit.status != "pass":
        raise ValueError(f"group leakage audit failed after reassignment: {audit.violations}")

    summary = {
        "dataset_dir": str(dataset_dir),
        "source_split": args.source_split,
        "target_split": args.target_split,
        "candidate_counterfactual_groups": len(by_group),
        "eligible_counterfactual_groups": len(eligible_groups),
        "moved_counterfactual_groups": len(selected_groups),
        "moved_rows": len(moved_rows),
        "moved_task_counts": dict(Counter(row.task_type for row in moved_rows)),
        "new_split_counts": dict(Counter(row.split for row in updated_rows)),
        "new_train_task_counts": dict(Counter(row.task_type for row in updated_rows if row.split == args.target_split)),
        "new_source_task_counts": dict(Counter(row.task_type for row in updated_rows if row.split == args.source_split)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return

    if args.backup:
        backup_dir = dataset_dir / "pre_counterfactual_train_backup"
        backup_dir.mkdir(exist_ok=True)
        for name in [
            *[f"{split}.jsonl" for split in split_names_for_rows(rows)],
            "metadata.json",
            "split_manifest.json",
            "parquet_manifest.json",
            "DATASET_CARD.md",
            "README.md",
        ]:
            path = dataset_dir / name
            if path.exists():
                shutil.copy2(path, backup_dir / name)

    split_summary = write_split_files(updated_rows, dataset_dir)
    (dataset_dir / "split_manifest.json").write_text(
        split_manifest(updated_rows).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    write_metadata(
        build_metadata(
            dataset_version=metadata.dataset_version,
            config_hash=metadata.config_hash,
            source_manifest_hash=metadata.source_manifest_hash,
            label_policy=metadata.label_policy,
            splits=split_summary,
            rows=updated_rows,
            license=metadata.license,
        ),
        dataset_dir / "metadata.json",
    )
    parquet_counts = write_parquet_files(updated_rows, dataset_dir / "parquet")
    (dataset_dir / "parquet_manifest.json").write_text(
        json.dumps({"files": parquet_counts, "compression": "zstd"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_dataset_card(dataset_dir)


def select_groups(groups: list[str], *, fraction: float, seed: int) -> list[str]:
    target_count = max(1, round(len(groups) * fraction))
    scored = sorted((stable_score(group, seed), group) for group in groups)
    return [group for _, group in scored[:target_count]]


def stable_score(value: str, seed: int) -> str:
    import hashlib

    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
