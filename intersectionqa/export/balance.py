"""Deterministic post-materialization class balancing for exported datasets."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from math import floor
from pathlib import Path
from typing import Iterable

from intersectionqa.enums import Relation, TaskType
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
from intersectionqa.hashing import sha256_json
from intersectionqa.schema import PublicTaskRow
from intersectionqa.splits.grouped import DEFAULT_SPLITS, audit_group_leakage, split_manifest

DEFAULT_RELATION_TARGETS = {
    Relation.INTERSECTING: 0.40,
    Relation.DISJOINT: 0.30,
    Relation.TOUCHING: 0.15,
    Relation.NEAR_MISS: 0.15,
}
DEFAULT_RELATION_BALANCE_SPLITS = set(DEFAULT_SPLITS)
PAIRWISE_ANSWERS = ("A", "B", "both", "neither")
SINGLE_GEOMETRY_TASKS = {
    TaskType.BINARY_INTERFERENCE,
    TaskType.RELATION_CLASSIFICATION,
    TaskType.VOLUME_BUCKET,
    TaskType.CLEARANCE_BUCKET,
    TaskType.REPAIR_DIRECTION,
    TaskType.TOLERANCE_FIT,
}


def balance_dataset_dir(
    dataset_dir: Path,
    *,
    backup: bool = True,
    dry_run: bool = False,
    relation_balance_splits: Iterable[str] = DEFAULT_RELATION_BALANCE_SPLITS,
    relation_targets: dict[Relation, float] = DEFAULT_RELATION_TARGETS,
    cap_pairwise: bool = True,
) -> dict[str, object]:
    rows = [row for split in DEFAULT_SPLITS for row in read_jsonl(dataset_dir / f"{split}.jsonl")]
    metadata = read_metadata(dataset_dir / "metadata.json")
    if metadata is None:
        raise ValueError(f"missing metadata.json in {dataset_dir}")

    balanced_rows, report = balance_rows(
        rows,
        relation_balance_splits=set(relation_balance_splits),
        relation_targets=relation_targets,
        cap_pairwise=cap_pairwise,
    )
    validate_rows(balanced_rows)
    audit = audit_group_leakage(balanced_rows)
    if audit.status != "pass":
        raise ValueError(f"group leakage audit failed after balancing: {audit.violations}")

    report = {
        **report,
        "dataset_dir": str(dataset_dir),
        "old_row_count": len(rows),
        "new_row_count": len(balanced_rows),
        "old_split_counts": dict(Counter(str(row.split) for row in rows)),
        "new_split_counts": dict(Counter(str(row.split) for row in balanced_rows)),
        "leakage_audit_status": audit.status,
        "dry_run": dry_run,
    }
    if dry_run:
        return report

    if backup:
        backup_dir = dataset_dir / "pre_class_balance_backup"
        backup_dir.mkdir(exist_ok=True)
        for name in [
            *[f"{split}.jsonl" for split in DEFAULT_SPLITS],
            "metadata.json",
            "split_manifest.json",
            "parquet_manifest.json",
            "DATASET_CARD.md",
            "README.md",
        ]:
            path = dataset_dir / name
            if path.exists():
                shutil.copy2(path, backup_dir / name)

    split_summary = write_split_files(balanced_rows, dataset_dir)
    (dataset_dir / "split_manifest.json").write_text(
        split_manifest(balanced_rows).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    write_metadata(
        build_metadata(
            dataset_version=metadata.dataset_version,
            config_hash=metadata.config_hash,
            source_manifest_hash=metadata.source_manifest_hash,
            label_policy=metadata.label_policy,
            splits=split_summary,
            rows=balanced_rows,
            license=metadata.license,
        ),
        dataset_dir / "metadata.json",
    )
    parquet_counts = write_parquet_files(balanced_rows, dataset_dir / "parquet")
    (dataset_dir / "parquet_manifest.json").write_text(
        json.dumps({"files": parquet_counts, "compression": "zstd"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_dataset_card(dataset_dir)
    (dataset_dir / "class_balance_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def balance_rows(
    rows: list[PublicTaskRow],
    *,
    relation_balance_splits: set[str] = DEFAULT_RELATION_BALANCE_SPLITS,
    relation_targets: dict[Relation, float] = DEFAULT_RELATION_TARGETS,
    cap_pairwise: bool = True,
) -> tuple[list[PublicTaskRow], dict[str, object]]:
    rows_by_split: dict[str, list[PublicTaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_split[str(row.split)].append(row)

    balanced: list[PublicTaskRow] = []
    split_reports: dict[str, object] = {}
    for split in DEFAULT_SPLITS:
        split_name = str(split)
        split_rows = rows_by_split.get(split_name, [])
        selected, split_report = balance_split_rows(
            split_rows,
            balance_relations=split_name in relation_balance_splits,
            relation_targets=relation_targets,
            cap_pairwise=cap_pairwise,
        )
        balanced.extend(selected)
        split_reports[split_name] = split_report

    return balanced, {
        "schema": "intersectionqa_class_balance_report_v1",
        "relation_targets": {relation.value: target for relation, target in relation_targets.items()},
        "relation_balance_splits": sorted(relation_balance_splits),
        "cap_pairwise": cap_pairwise,
        "splits": split_reports,
    }


def balance_split_rows(
    rows: list[PublicTaskRow],
    *,
    balance_relations: bool,
    relation_targets: dict[Relation, float],
    cap_pairwise: bool,
) -> tuple[list[PublicTaskRow], dict[str, object]]:
    single_rows: list[PublicTaskRow] = []
    pairwise_rows: list[PublicTaskRow] = []
    other_rows: list[PublicTaskRow] = []
    for row in rows:
        if row.task_type in SINGLE_GEOMETRY_TASKS and len(row.geometry_ids) == 1:
            single_rows.append(row)
        elif row.task_type == TaskType.PAIRWISE_INTERFERENCE:
            pairwise_rows.append(row)
        else:
            other_rows.append(row)

    kept_single = single_rows
    relation_report: dict[str, object] = {}
    if balance_relations and single_rows:
        kept_single, relation_report = balance_single_geometry_rows(single_rows, relation_targets)
    kept_pairwise = cap_pairwise_rows(pairwise_rows) if cap_pairwise else pairwise_rows

    selected = [*kept_single, *kept_pairwise, *other_rows]
    return selected, {
        "old_row_count": len(rows),
        "new_row_count": len(selected),
        "relation_balance": relation_report,
        "old_task_counts": dict(Counter(str(row.task_type) for row in rows)),
        "new_task_counts": dict(Counter(str(row.task_type) for row in selected)),
        "old_answer_counts": answer_counts(rows),
        "new_answer_counts": answer_counts(selected),
    }


def balance_single_geometry_rows(
    rows: list[PublicTaskRow],
    relation_targets: dict[Relation, float],
) -> tuple[list[PublicTaskRow], dict[str, object]]:
    by_geometry: dict[str, list[PublicTaskRow]] = defaultdict(list)
    for row in rows:
        by_geometry[row.geometry_ids[0]].append(row)

    by_relation: dict[Relation, list[tuple[str, list[PublicTaskRow]]]] = defaultdict(list)
    preserved: list[tuple[str, list[PublicTaskRow]]] = []
    for geometry_id, geometry_rows in sorted(by_geometry.items()):
        relation = Relation(geometry_rows[0].labels.relation)
        if relation in relation_targets:
            by_relation[relation].append((geometry_id, geometry_rows))
        else:
            preserved.append((geometry_id, geometry_rows))

    missing_targets = [
        relation.value for relation in relation_targets if not by_relation.get(relation)
    ]
    if missing_targets:
        kept_groups = [group for groups in by_relation.values() for group in groups]
        kept_groups.extend(preserved)
        return flatten_groups(kept_groups), {
            "status": "skipped_missing_target_classes",
            "missing_target_relations": missing_targets,
            "old_relation_counts": relation_group_counts(by_relation, preserved),
            "new_relation_counts": relation_group_counts(by_relation, preserved),
        }

    target_counts = proportional_cap_counts(
        {relation: len(by_relation[relation]) for relation in relation_targets},
        relation_targets,
    )
    kept: list[tuple[str, list[PublicTaskRow]]] = []
    for relation in relation_targets:
        relation_groups = sorted(by_relation[relation], key=lambda item: stable_group_score(item[1]))
        kept.extend(relation_groups[: target_counts[relation]])
    kept.extend(preserved)
    return flatten_groups(kept), {
        "status": "balanced",
        "old_relation_counts": relation_group_counts(by_relation, preserved),
        "new_relation_counts": dict(Counter(str(group[1][0].labels.relation) for group in kept)),
        "target_counts": {relation.value: count for relation, count in target_counts.items()},
        "preserved_non_target_relation_count": len(preserved),
    }


def cap_pairwise_rows(rows: list[PublicTaskRow]) -> list[PublicTaskRow]:
    by_answer: dict[str, list[PublicTaskRow]] = defaultdict(list)
    for row in rows:
        by_answer[row.answer].append(row)
    if any(not by_answer.get(answer) for answer in PAIRWISE_ANSWERS):
        return rows
    cap = min(len(by_answer[answer]) for answer in PAIRWISE_ANSWERS)
    kept: list[PublicTaskRow] = []
    for answer in PAIRWISE_ANSWERS:
        kept.extend(sorted(by_answer[answer], key=stable_row_score)[:cap])
    return sorted(kept, key=lambda row: row.id)


def proportional_cap_counts(
    available: dict[Relation, int],
    targets: dict[Relation, float],
) -> dict[Relation, int]:
    upper_total = sum(available.values())
    for total in range(upper_total, 0, -1):
        raw_counts = {relation: total * target for relation, target in targets.items()}
        counts = {relation: floor(raw_count) for relation, raw_count in raw_counts.items()}
        remaining = total - sum(counts.values())
        remainders = sorted(
            targets,
            key=lambda relation: (raw_counts[relation] - counts[relation], relation.value),
            reverse=True,
        )
        for relation in remainders[:remaining]:
            counts[relation] += 1
        if all(0 < counts[relation] <= available[relation] for relation in targets):
            return counts
    raise ValueError(f"cannot allocate positive proportional counts from {available}")


def flatten_groups(groups: Iterable[tuple[str, list[PublicTaskRow]]]) -> list[PublicTaskRow]:
    return [row for _, group_rows in groups for row in group_rows]


def relation_group_counts(
    by_relation: dict[Relation, list[tuple[str, list[PublicTaskRow]]]],
    preserved: list[tuple[str, list[PublicTaskRow]]],
) -> dict[str, int]:
    counts = {relation.value: len(groups) for relation, groups in by_relation.items()}
    for _, group_rows in preserved:
        relation = str(group_rows[0].labels.relation)
        counts[relation] = counts.get(relation, 0) + 1
    return dict(sorted(counts.items()))


def answer_counts(rows: list[PublicTaskRow]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[str(row.task_type)][row.answer] += 1
    return {task: dict(counter) for task, counter in sorted(counts.items())}


def stable_group_score(rows: list[PublicTaskRow]) -> str:
    representative = rows[0]
    return sha256_json(
        {
            "geometry_id": representative.geometry_ids[0],
            "split": representative.split,
            "relation": representative.labels.relation,
            "base_object_pair_id": representative.base_object_pair_id,
        }
    )


def stable_row_score(row: PublicTaskRow) -> str:
    return sha256_json(
        {
            "answer": row.answer,
            "counterfactual_group_id": row.counterfactual_group_id,
            "geometry_ids": row.geometry_ids,
            "variant_id": row.variant_id,
        }
    )
