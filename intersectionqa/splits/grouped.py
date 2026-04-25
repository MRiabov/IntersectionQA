"""Deterministic group-safe splits and leakage audits."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from itertools import combinations
from typing import Iterable

from intersectionqa.enums import AuditStatus, Split, TaskType
from intersectionqa.schema import (
    GeometryRecord,
    GroupHoldoutRule,
    LeakageAudit,
    LeakageViolation,
    PublicTaskRow,
    SplitLabelDistributions,
    SplitManifest,
    SplitManifestSummary,
)

DEFAULT_SPLITS = [
    "train",
    "validation",
    "test_random",
    "test_generator_heldout",
    "test_object_pair_heldout",
    "test_near_boundary",
]

BOUNDARY_SPLIT_BUCKETS = (
    (75, "train"),
    (85, "validation"),
    (90, "test_random"),
    (93, "test_object_pair_heldout"),
    (100, "test_near_boundary"),
)


def split_group(record: GeometryRecord | PublicTaskRow) -> str:
    return (
        record.assembly_group_id
        or record.base_object_pair_id
        or record.counterfactual_group_id
        or record.geometry_ids[0]  # type: ignore[attr-defined]
    )


def assign_geometry_splits(records: list[GeometryRecord], seed: int) -> dict[str, str]:
    groups: dict[str, list[GeometryRecord]] = defaultdict(list)
    for record in records:
        groups[split_group(record)].append(record)

    heldout_generators = _heldout_generators(groups, seed)
    assignments: dict[str, str] = {}
    for group_id, group_records in sorted(groups.items()):
        split = _group_split(group_id, group_records, seed, heldout_generators)
        for record in group_records:
            assignments[record.geometry_id] = split
    return assignments


def reassign_public_row_splits(
    rows: list[PublicTaskRow],
    seed: int,
) -> tuple[list[PublicTaskRow], dict[str, object]]:
    groups: dict[str, list[PublicTaskRow]] = defaultdict(list)
    for row in rows:
        groups[split_group(row)].append(row)

    reassigned: list[PublicTaskRow] = []
    group_reports: dict[str, dict[str, object]] = {}
    for group_id, group_rows in sorted(groups.items()):
        split = _public_group_split(group_id, group_rows, seed)
        for row in group_rows:
            metadata = dict(row.metadata)
            metadata["split_group"] = group_id
            reassigned.append(row.model_copy(update={"split": Split(split), "metadata": metadata}))
        group_reports[group_id] = {
            "old_splits": _counts(str(row.split) for row in group_rows),
            "new_split": split,
            "row_count": len(group_rows),
            "near_boundary_group": is_near_boundary_group(group_rows),
        }

    return sorted(reassigned, key=lambda row: row.id), {
        "schema": "intersectionqa_split_redistribution_report_v1",
        "seed": seed,
        "old_split_counts": _counts(str(row.split) for row in rows),
        "new_split_counts": _counts(str(row.split) for row in reassigned),
        "old_group_counts": _counts(str(group_rows[0].split) for group_rows in groups.values()),
        "new_group_counts": _counts(report["new_split"] for report in group_reports.values()),
        "boundary_split_buckets": [
            {"upper_bound": upper_bound, "split": split}
            for upper_bound, split in BOUNDARY_SPLIT_BUCKETS
        ],
        "groups": group_reports,
    }


def _public_group_split(group_id: str, rows: list[PublicTaskRow], seed: int) -> str:
    if any(row.split == Split.TEST_GENERATOR_HELDOUT for row in rows):
        return "test_generator_heldout"
    if is_near_boundary_group(rows):
        return _boundary_group_split(group_id, seed)
    bucket = _stable_bucket(group_id, seed, 100)
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "validation"
    if bucket < 90:
        return "test_object_pair_heldout"
    return "test_random"


def _heldout_generators(
    groups: dict[str, list[GeometryRecord]],
    seed: int,
) -> set[str]:
    generators = sorted(
        {
            record.metadata.get("generator_ids", [record.metadata.get("generator_id")])[0]
            for group_records in groups.values()
            for record in group_records
            if record.source != "synthetic"
            and (record.metadata.get("generator_ids", [record.metadata.get("generator_id")])[0])
        }
    )
    if len(generators) < 2:
        return set()
    selected = {generator for generator in generators if _stable_bucket(generator, seed, 100) >= 88}
    if not selected:
        selected = {generators[-1]}
    return selected


def _group_split(
    group_id: str,
    records: list[GeometryRecord],
    seed: int,
    heldout_generators: set[str],
) -> str:
    if any(record.source == "synthetic" for record in records):
        # Keep fixture smoke rows represented across non-heldout splits without splitting groups.
        bucket = _stable_bucket(group_id, seed, 3)
        return ["train", "validation", "test_random"][bucket]
    group_generators = {
        record.metadata.get("generator_ids", [record.metadata.get("generator_id")])[0]
        for record in records
        if record.metadata.get("generator_ids", [record.metadata.get("generator_id")])[0]
    }
    if group_generators & heldout_generators:
        return "test_generator_heldout"
    if is_near_boundary_group(records):
        return _boundary_group_split(group_id, seed)
    bucket = _stable_bucket(group_id, seed, 100)
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "validation"
    if bucket < 90:
        return "test_object_pair_heldout"
    return "test_random"


def is_near_boundary_group(records: Iterable[GeometryRecord | PublicTaskRow]) -> bool:
    for record in records:
        tags = set(record.difficulty_tags)
        if "near_boundary" in tags:
            return True
        if str(record.labels.relation) in {"touching", "near_miss"}:
            return True
        if getattr(record, "task_type", None) in {
            TaskType.PAIRWISE_INTERFERENCE,
            TaskType.RANKING_NORMALIZED_INTERSECTION,
        }:
            return True
    return False


def _boundary_group_split(group_id: str, seed: int) -> str:
    bucket = _stable_bucket(f"boundary:{group_id}", seed, 100)
    for upper_bound, split in BOUNDARY_SPLIT_BUCKETS:
        if bucket < upper_bound:
            return split
    raise AssertionError("unreachable boundary split bucket")


def _stable_bucket(value: str, seed: int, modulo: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def audit_group_leakage(
    rows: Iterable[PublicTaskRow],
    forbidden_pairs: list[tuple[str, str]] | None = None,
) -> LeakageAudit:
    forbidden_pairs = forbidden_pairs or list(combinations(DEFAULT_SPLITS, 2))
    fields = ["generator_id", "base_object_pair_id", "assembly_group_id", "counterfactual_group_id"]
    split_values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        for field in fields:
            value = getattr(row, field)
            if value is not None:
                split_values[row.split][field].add(value)

    violations: list[LeakageViolation] = []
    for left, right in forbidden_pairs:
        for field in _fields_for_pair(left, right):
            overlap = sorted(split_values[left][field] & split_values[right][field])
            if overlap:
                violations.append(
                    LeakageViolation(split_pair=[left, right], field=field, values=overlap)
                )
    return LeakageAudit(
        status=AuditStatus.PASS if not violations else AuditStatus.FAIL,
        checked_group_fields=fields,
        violation_count=len(violations),
        violations=violations,
    )


def _fields_for_pair(left: str, right: str) -> list[str]:
    fields = ["base_object_pair_id", "assembly_group_id", "counterfactual_group_id"]
    if "test_generator_heldout" in {left, right}:
        fields.append("generator_id")
    return fields


def split_manifest(rows: list[PublicTaskRow]) -> SplitManifest:
    splits: dict[str, SplitManifestSummary] = {}
    for split in DEFAULT_SPLITS:
        split_rows = [row for row in rows if row.split == split]
        splits[split] = SplitManifestSummary(
            row_count=len(split_rows),
            task_counts=_counts(row.task_type for row in split_rows),
            label_distributions=SplitLabelDistributions(
                relation=_counts(row.labels.relation for row in split_rows),
                binary_answer=_counts(
                    row.answer for row in split_rows if row.task_type == TaskType.BINARY_INTERFERENCE
                ),
                volume_bucket=_counts(
                    row.answer for row in split_rows if row.task_type == TaskType.VOLUME_BUCKET
                ),
            ),
            generator_ids=sorted({row.generator_id for row in split_rows if row.generator_id}),
            base_object_pair_ids=sorted({row.base_object_pair_id for row in split_rows}),
            assembly_group_ids=sorted({row.assembly_group_id for row in split_rows}),
            counterfactual_group_ids=sorted(
                {row.counterfactual_group_id for row in split_rows if row.counterfactual_group_id}
            ),
            group_holdout_rule_ids=["counterfactual_inseparable", "object_pair_holdout"],
        )
    audit = audit_group_leakage(rows)
    return SplitManifest(
        dataset_version="v0.1",
        split_names=DEFAULT_SPLITS,
        splits=splits,
        group_holdout_rules=[
            GroupHoldoutRule(
                rule_id="counterfactual_inseparable",
                description="Rows sharing counterfactual_group_id must not cross splits.",
                group_fields=["counterfactual_group_id"],
                forbidden_cross_split_pairs=[list(pair) for pair in combinations(DEFAULT_SPLITS, 2)],
                status=audit.status,
            ),
            GroupHoldoutRule(
                rule_id="object_pair_holdout",
                description="Rows sharing object-pair or assembly IDs must not cross splits.",
                group_fields=["base_object_pair_id", "assembly_group_id"],
                forbidden_cross_split_pairs=[list(pair) for pair in combinations(DEFAULT_SPLITS, 2)],
                status=audit.status,
            ),
        ],
        leakage_audit=audit,
    )


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return dict(sorted(counts.items()))
