"""Shared prompt and answer helpers."""

from __future__ import annotations

from intersectionqa.generation.assembly import build_assembly_script
from intersectionqa.geometry.transforms import format_transform_block
from intersectionqa.enums import Split, TaskType
from intersectionqa.hashing import sha256_json
from intersectionqa.schema import GeometryRecord, Hashes, PublicTaskRow

TASK_PREFIX = {
    TaskType.BINARY_INTERFERENCE: "intersectionqa_binary",
    TaskType.RELATION_CLASSIFICATION: "intersectionqa_relation",
    TaskType.VOLUME_BUCKET: "intersectionqa_volume_bucket",
    TaskType.CLEARANCE_BUCKET: "intersectionqa_clearance_bucket",
    TaskType.PAIRWISE_INTERFERENCE: "intersectionqa_pairwise",
    TaskType.RANKING_NORMALIZED_INTERSECTION: "intersectionqa_ranking",
    TaskType.REPAIR_DIRECTION: "intersectionedit_repair_direction",
    TaskType.TOLERANCE_FIT: "intersectionqa_tolerance_fit",
}


def object_code_from_script(script: str) -> str:
    marker = "\ndef _place("
    return script.split(marker, 1)[0].strip() + "\n"


def transforms_text(record: GeometryRecord) -> str:
    return format_transform_block(record.transform_a, record.transform_b)


def script_for_prompt(record: GeometryRecord) -> str:
    return build_assembly_script(
        object_code_from_script(record.assembly_script),
        record.transform_a,
        record.transform_b,
    )


def strict_parse(output: str, allowed: set[str]) -> str | None:
    stripped = output.strip(" \t\r\n")
    return stripped if stripped in allowed else None


def row_hashes(record: GeometryRecord, prompt: str, task_type: TaskType, template_version: str) -> Hashes:
    return Hashes(
        source_code_hash=record.hashes.source_code_hash,
        object_hash=record.hashes.object_hash,
        transform_hash=record.hashes.transform_hash,
        geometry_hash=record.hashes.geometry_hash,
        config_hash=record.hashes.config_hash,
        prompt_hash=sha256_json(
            {
                "template_version": template_version,
                "task_type": task_type,
                "prompt": prompt,
                "geometry_ids": [record.geometry_id],
            }
        ),
    )


def group_row_hashes(
    records: list[GeometryRecord],
    prompt: str,
    task_type: TaskType,
    template_version: str,
) -> Hashes:
    geometry_ids = [record.geometry_id for record in records]
    return Hashes(
        source_code_hash=sha256_json([record.hashes.source_code_hash for record in records]),
        object_hash=sha256_json([record.hashes.object_hash for record in records]),
        transform_hash=sha256_json([record.hashes.transform_hash for record in records]),
        geometry_hash=sha256_json([record.hashes.geometry_hash for record in records]),
        config_hash=records[0].hashes.config_hash,
        prompt_hash=sha256_json(
            {
                "template_version": template_version,
                "task_type": task_type,
                "prompt": prompt,
                "geometry_ids": geometry_ids,
            }
        ),
    )


def public_row(
    *,
    record: GeometryRecord,
    task_type: TaskType,
    answer: str,
    prompt: str,
    row_number: int,
    split: Split,
    template_version: str,
    extras: dict[str, object] | None = None,
) -> PublicTaskRow:
    row_id = f"{TASK_PREFIX[task_type]}_{row_number:06d}"
    metadata = {
        "prompt_template_version": template_version,
        "split_group": record.counterfactual_group_id
        or record.assembly_group_id
        or record.base_object_pair_id,
        "candidate_strategy": record.metadata.get("candidate_strategy"),
        "source_subtrees": record.metadata.get("source_subtrees"),
        "generator_ids": record.metadata.get("generator_ids", [record.metadata.get("generator_id")]),
        "cadquery_ops": record.metadata.get("cadquery_ops", []),
        "topology_tags": record.metadata.get("topology_tags", []),
        "artifact_ids": record.metadata.get("artifact_ids", {}),
    }
    if extras:
        metadata.update(extras)
    return PublicTaskRow(
        id=row_id,
        dataset_version="v0.1",
        split=split,
        task_type=task_type,
        prompt=prompt,
        answer=answer,
        script=script_for_prompt(record),
        geometry_ids=[record.geometry_id],
        source=record.source,
        generator_id=record.metadata.get("generator_ids", [None])[0],
        base_object_pair_id=record.base_object_pair_id,
        assembly_group_id=record.assembly_group_id,
        counterfactual_group_id=record.counterfactual_group_id,
        variant_id=record.variant_id,
        changed_parameter=record.changed_parameter,
        changed_value=record.changed_value,
        labels=record.labels,
        diagnostics=record.diagnostics,
        difficulty_tags=record.difficulty_tags,
        label_policy=record.label_policy,
        hashes=row_hashes(record, prompt, task_type, template_version),
        metadata=metadata,
    )


def public_group_row(
    *,
    records: list[GeometryRecord],
    task_type: TaskType,
    answer: str,
    prompt: str,
    row_number: int,
    split: Split,
    template_version: str,
    derived_variant_id: str,
    changed_value: object,
    script: str,
    extras: dict[str, object] | None = None,
) -> PublicTaskRow:
    representative = records[0]
    group_id = representative.counterfactual_group_id
    metadata = {
        "prompt_template_version": template_version,
        "split_group": group_id or representative.assembly_group_id or representative.base_object_pair_id,
        "candidate_strategy": "counterfactual_derived_prompt",
        "source_subtrees": representative.metadata.get("source_subtrees"),
        "generator_ids": representative.metadata.get("generator_ids", [representative.metadata.get("generator_id")]),
        "cadquery_ops": representative.metadata.get("cadquery_ops", []),
        "topology_tags": representative.metadata.get("topology_tags", []),
        "artifact_ids": representative.metadata.get("artifact_ids", {}),
        "source_geometry_ids": [record.geometry_id for record in records],
        "source_variant_ids": [record.variant_id for record in records],
    }
    if extras:
        metadata.update(extras)
    return PublicTaskRow(
        id=f"{TASK_PREFIX[task_type]}_{row_number:06d}",
        dataset_version="v0.1",
        split=split,
        task_type=task_type,
        prompt=prompt,
        answer=answer,
        script=script,
        geometry_ids=[record.geometry_id for record in records],
        source=representative.source,
        generator_id=representative.metadata.get("generator_ids", [None])[0],
        base_object_pair_id=representative.base_object_pair_id,
        assembly_group_id=representative.assembly_group_id,
        counterfactual_group_id=group_id,
        variant_id=derived_variant_id,
        changed_parameter=representative.changed_parameter,
        changed_value=changed_value,
        labels=representative.labels,
        diagnostics=representative.diagnostics,
        difficulty_tags=representative.difficulty_tags,
        label_policy=representative.label_policy,
        hashes=group_row_hashes(records, prompt, task_type, template_version),
        metadata=metadata,
    )
