"""Deterministic group-safe splits and leakage audits."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

from intersectionqa.enums import AuditStatus, TaskType
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


def split_group(record: GeometryRecord | PublicTaskRow) -> str:
    return (
        record.counterfactual_group_id
        or record.assembly_group_id
        or record.base_object_pair_id
        or record.geometry_ids[0]  # type: ignore[attr-defined]
    )


def assign_geometry_splits(records: list[GeometryRecord], seed: int) -> dict[str, str]:
    groups: dict[str, list[GeometryRecord]] = defaultdict(list)
    for record in records:
        groups[split_group(record)].append(record)

    assignments: dict[str, str] = {}
    for group_id, group_records in sorted(groups.items()):
        split = _group_split(group_id, group_records, seed)
        for record in group_records:
            assignments[record.geometry_id] = split
    return assignments


def _group_split(group_id: str, records: list[GeometryRecord], seed: int) -> str:
    tags = set().union(*(set(record.difficulty_tags) for record in records))
    relations = {record.labels.relation for record in records}
    if "near_boundary" in tags or relations & {"touching", "near_miss"}:
        return "test_near_boundary"
    if any(record.source == "synthetic" for record in records):
        # Keep fixture smoke rows represented across non-heldout splits without splitting groups.
        bucket = _stable_bucket(group_id, seed, 3)
        return ["train", "validation", "test_random"][bucket]
    bucket = _stable_bucket(group_id, seed, 100)
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "validation"
    if bucket < 88:
        return "test_generator_heldout"
    if bucket < 96:
        return "test_object_pair_heldout"
    return "test_random"


def _stable_bucket(value: str, seed: int, modulo: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def audit_group_leakage(
    rows: Iterable[PublicTaskRow],
    forbidden_pairs: list[tuple[str, str]] | None = None,
) -> LeakageAudit:
    forbidden_pairs = forbidden_pairs or [
        ("train", "validation"),
        ("train", "test_random"),
        ("train", "test_generator_heldout"),
        ("train", "test_object_pair_heldout"),
        ("train", "test_near_boundary"),
        ("validation", "test_object_pair_heldout"),
    ]
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
                forbidden_cross_split_pairs=[
                    ["train", "validation"],
                    ["train", "test_random"],
                    ["train", "test_near_boundary"],
                ],
                status=audit.status,
            ),
            GroupHoldoutRule(
                rule_id="object_pair_holdout",
                description="Rows sharing object-pair or assembly IDs must not cross object-pair tests.",
                group_fields=["base_object_pair_id", "assembly_group_id"],
                forbidden_cross_split_pairs=[
                    ["train", "test_object_pair_heldout"],
                    ["validation", "test_object_pair_heldout"],
                ],
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
