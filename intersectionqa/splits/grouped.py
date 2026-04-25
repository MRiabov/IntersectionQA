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

EXTENSION_SPLITS = [
    "test_topology_heldout",
    "test_operation_heldout",
]

ALL_SPLITS = [*DEFAULT_SPLITS, *EXTENSION_SPLITS]

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


def split_names_for_rows(rows: Iterable[PublicTaskRow]) -> list[str]:
    observed = {str(row.split) for row in rows}
    return [split for split in ALL_SPLITS if split in DEFAULT_SPLITS or split in observed]


def assign_geometry_splits(records: list[GeometryRecord], seed: int) -> dict[str, str]:
    groups: dict[str, list[GeometryRecord]] = defaultdict(list)
    for record in records:
        groups[split_group(record)].append(record)

    heldout_generators = _heldout_generators(groups, seed)
    heldout_topologies = _heldout_metadata_values(
        groups,
        seed,
        metadata_key="topology_tags",
        excluded_values={"unknown", "primitive"},
        preferred_values=[
            "ring",
            "bracket",
            "clamp",
            "hollow_box",
            "hollow",
            "shaft",
            "housing",
            "flange",
            "plate_with_holes",
        ],
    )
    heldout_operations = _heldout_metadata_values(
        groups,
        seed,
        metadata_key="cadquery_ops",
        excluded_values=set(),
        preferred_values=[
            "fillet",
            "chamfer",
            "shell",
            "loft",
            "revolve",
            "sweep",
            "cutThruAll",
        ],
    )
    assignments: dict[str, str] = {}
    for group_id, group_records in sorted(groups.items()):
        split = _group_split(
            group_id,
            group_records,
            seed,
            heldout_generators,
            heldout_topologies,
            heldout_operations,
        )
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
    heldout_topologies = _heldout_public_metadata_values(
        groups,
        seed,
        metadata_key="topology_tags",
        excluded_values={"unknown", "primitive"},
        preferred_values=[
            "ring",
            "bracket",
            "clamp",
            "hollow_box",
            "hollow",
            "shaft",
            "housing",
            "flange",
            "plate_with_holes",
        ],
    )
    heldout_operations = _heldout_public_metadata_values(
        groups,
        seed,
        metadata_key="cadquery_ops",
        excluded_values=set(),
        preferred_values=[
            "fillet",
            "chamfer",
            "shell",
            "loft",
            "revolve",
            "sweep",
            "cutThruAll",
        ],
    )
    group_reports: dict[str, dict[str, object]] = {}
    for group_id, group_rows in sorted(groups.items()):
        split = _public_group_split(
            group_id,
            group_rows,
            seed,
            heldout_topologies,
            heldout_operations,
        )
        for row in group_rows:
            metadata = dict(row.metadata)
            metadata["split_group"] = group_id
            reassigned.append(row.model_copy(update={"split": Split(split), "metadata": metadata}))
        group_reports[group_id] = {
            "old_splits": _counts(str(row.split) for row in group_rows),
            "new_split": split,
            "row_count": len(group_rows),
            "near_boundary_group": is_near_boundary_group(group_rows),
            "topology_tags": sorted(_group_public_metadata_values(group_rows, "topology_tags")),
            "cadquery_ops": sorted(_group_public_metadata_values(group_rows, "cadquery_ops")),
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
        "heldout_topology_tags": sorted(heldout_topologies),
        "heldout_cadquery_ops": sorted(heldout_operations),
        "groups": group_reports,
    }


def _public_group_split(
    group_id: str,
    rows: list[PublicTaskRow],
    seed: int,
    heldout_topologies: set[str],
    heldout_operations: set[str],
) -> str:
    if any(row.split == Split.TEST_GENERATOR_HELDOUT for row in rows):
        return "test_generator_heldout"
    if _group_public_metadata_values(rows, "topology_tags") & heldout_topologies:
        return "test_topology_heldout"
    if _group_public_metadata_values(rows, "cadquery_ops") & heldout_operations:
        return "test_operation_heldout"
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
    heldout_topologies: set[str],
    heldout_operations: set[str],
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
    if _group_metadata_values(records, "topology_tags") & heldout_topologies:
        return "test_topology_heldout"
    if _group_metadata_values(records, "cadquery_ops") & heldout_operations:
        return "test_operation_heldout"
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


def _heldout_metadata_values(
    groups: dict[str, list[GeometryRecord]],
    seed: int,
    *,
    metadata_key: str,
    excluded_values: set[str],
    preferred_values: list[str],
) -> set[str]:
    group_values = [
        _group_metadata_values(group_records, metadata_key)
        for group_records in groups.values()
        if not any(record.source == "synthetic" for record in group_records)
    ]
    return _select_heldout_values(
        group_values,
        seed,
        excluded_values=excluded_values,
        preferred_values=preferred_values,
        key_namespace=metadata_key,
    )


def _heldout_public_metadata_values(
    groups: dict[str, list[PublicTaskRow]],
    seed: int,
    *,
    metadata_key: str,
    excluded_values: set[str],
    preferred_values: list[str],
) -> set[str]:
    group_values = [
        _group_public_metadata_values(group_rows, metadata_key)
        for group_rows in groups.values()
        if not any(row.source == "synthetic" for row in group_rows)
    ]
    return _select_heldout_values(
        group_values,
        seed,
        excluded_values=excluded_values,
        preferred_values=preferred_values,
        key_namespace=metadata_key,
    )


def _select_heldout_values(
    group_values: list[set[str]],
    seed: int,
    *,
    excluded_values: set[str],
    preferred_values: list[str],
    key_namespace: str,
) -> set[str]:
    total_groups = len(group_values)
    if total_groups < 3:
        return set()
    counts: dict[str, int] = defaultdict(int)
    for values in group_values:
        for value in values:
            normalized = value.strip()
            if normalized and normalized not in excluded_values:
                counts[normalized] += 1
    if len(counts) < 2:
        return set()

    max_groups = max(1, total_groups // 5)
    eligible = {
        value: count
        for value, count in counts.items()
        if 0 < count <= max_groups and count < total_groups
    }
    if not eligible:
        return set()
    preferred_rank = {value: index for index, value in enumerate(preferred_values)}
    selected = min(
        eligible,
        key=lambda value: (
            preferred_rank.get(value, len(preferred_rank)),
            eligible[value],
            _stable_bucket(f"{key_namespace}:{value}", seed, 1_000_000),
            value,
        ),
    )
    return {selected}


def _group_metadata_values(records: Iterable[GeometryRecord], metadata_key: str) -> set[str]:
    values: set[str] = set()
    for record in records:
        raw = record.metadata.get(metadata_key, [])
        if isinstance(raw, list):
            values.update(str(item) for item in raw)
    return values


def _group_public_metadata_values(rows: Iterable[PublicTaskRow], metadata_key: str) -> set[str]:
    values: set[str] = set()
    for row in rows:
        raw = row.metadata.get(metadata_key, [])
        if isinstance(raw, list):
            values.update(str(item) for item in raw)
    return values


def audit_group_leakage(
    rows: Iterable[PublicTaskRow],
    forbidden_pairs: list[tuple[str, str]] | None = None,
) -> LeakageAudit:
    rows = list(rows)
    active_splits = split_names_for_rows(rows)
    forbidden_pairs = forbidden_pairs or list(combinations(active_splits, 2))
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
    active_splits = split_names_for_rows(rows)
    active_rule_ids = ["counterfactual_inseparable", "object_pair_holdout"]
    if "test_topology_heldout" in active_splits:
        active_rule_ids.append("topology_holdout")
    if "test_operation_heldout" in active_splits:
        active_rule_ids.append("operation_holdout")
    for split in active_splits:
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
            group_holdout_rule_ids=active_rule_ids,
        )
    audit = audit_group_leakage(rows)
    group_holdout_rules = [
        GroupHoldoutRule(
            rule_id="counterfactual_inseparable",
            description="Rows sharing counterfactual_group_id must not cross splits.",
            group_fields=["counterfactual_group_id"],
            forbidden_cross_split_pairs=[list(pair) for pair in combinations(active_splits, 2)],
            status=audit.status,
        ),
        GroupHoldoutRule(
            rule_id="object_pair_holdout",
            description="Rows sharing object-pair or assembly IDs must not cross splits.",
            group_fields=["base_object_pair_id", "assembly_group_id"],
            forbidden_cross_split_pairs=[list(pair) for pair in combinations(active_splits, 2)],
            status=audit.status,
        ),
    ]
    if "test_topology_heldout" in active_splits:
        group_holdout_rules.append(
            GroupHoldoutRule(
                rule_id="topology_holdout",
                description="Rows with selected topology tags are assigned to test_topology_heldout.",
                group_fields=["metadata.topology_tags"],
                forbidden_cross_split_pairs=[],
                status=audit.status,
            )
        )
    if "test_operation_heldout" in active_splits:
        group_holdout_rules.append(
            GroupHoldoutRule(
                rule_id="operation_holdout",
                description="Rows with selected CadQuery operations are assigned to test_operation_heldout.",
                group_fields=["metadata.cadquery_ops"],
                forbidden_cross_split_pairs=[],
                status=audit.status,
            )
        )
    return SplitManifest(
        dataset_version="v0.1",
        split_names=active_splits,
        splits=splits,
        group_holdout_rules=group_holdout_rules,
        leakage_audit=audit,
    )


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return dict(sorted(counts.items()))
