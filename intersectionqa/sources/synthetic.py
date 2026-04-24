"""Deterministic synthetic fixtures for golden and smoke runs only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intersectionqa.generation.assembly import TwoObjectAssembly
from intersectionqa.geometry.bbox import (
    AABB,
    aabb_intersection_volume,
    aabb_minimum_distance,
    aabb_overlap,
    box_aabb,
    transform_aabb,
)
from intersectionqa.geometry.labels import RawGeometry, derive_labels
from intersectionqa.hashing import sha256_json, sha256_text
from intersectionqa.schema import (
    ArtifactIds,
    GeometryRecord,
    Hashes,
    LabelPolicy,
    SourceObjectRecord,
    Transform,
)

GENERATOR_ID = "gen_synthetic_primitives_v01"


@dataclass(frozen=True)
class SyntheticFixture:
    name: str
    object_a: SourceObjectRecord
    object_b: SourceObjectRecord
    bbox_a: AABB
    bbox_b: AABB
    transform_a: Transform
    transform_b: Transform
    raw_geometry: RawGeometry
    changed_value: Any
    difficulty_tags: list[str]
    metadata: dict[str, Any]


def synthetic_source_object(object_id: str, function_name: str, dimensions: tuple[float, float, float]) -> SourceObjectRecord:
    width, depth, height = dimensions
    code = "\n".join(
        [
            f"def {function_name}():",
            f"    return cq.Workplane(\"XY\").box({width:.6g}, {depth:.6g}, {height:.6g})",
            "",
        ]
    )
    source_code_hash = sha256_text(code)
    object_hash = sha256_json(
        {"source": "synthetic", "function": function_name, "dimensions": dimensions}
    )
    return SourceObjectRecord(
        object_id=object_id,
        source="synthetic",
        source_id=f"synthetic_box_{width:g}_{depth:g}_{height:g}",
        generator_id=GENERATOR_ID,
        source_path=None,
        source_license="cc-by-4.0",
        object_name=f"box_{width:g}_{depth:g}_{height:g}",
        normalized_code=code,
        object_function_name=function_name,
        cadquery_ops=["box"],
        topology_tags=["box", "primitive"],
        metadata={
            "units": "mm",
            "source_subset": "synthetic_primitives",
            "parameters": {"width": width, "depth": depth, "height": height},
        },
        hashes=Hashes(
            source_code_hash=source_code_hash,
            object_hash=object_hash,
            transform_hash=None,
            geometry_hash=None,
            config_hash=None,
            prompt_hash=None,
        ),
    )


def synthetic_source_records() -> list[SourceObjectRecord]:
    return [
        synthetic_source_object("obj_000001", "object_a", (10.0, 10.0, 10.0)),
        synthetic_source_object("obj_000002", "object_b", (10.0, 10.0, 10.0)),
        synthetic_source_object("obj_000003", "object_b", (6.0, 6.0, 6.0)),
    ]


def synthetic_fixtures(policy: LabelPolicy) -> list[SyntheticFixture]:
    box10_a, box10_b, box6_b = synthetic_source_records()
    bbox10 = box_aabb(10.0, 10.0, 10.0)
    bbox6 = box_aabb(6.0, 6.0, 6.0)

    def fixture(
        name: str,
        object_b: SourceObjectRecord,
        bbox_b: AABB,
        tx: float,
        intersection_volume: float | None = None,
        minimum_distance: float | None = None,
        contains_b_in_a: bool = False,
        rotation_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
        tags: list[str] | None = None,
        strategy: str = "golden_box_fixture",
    ) -> SyntheticFixture:
        transform_a = Transform(translation=(0.0, 0.0, 0.0), rotation_xyz_deg=(0.0, 0.0, 0.0))
        transform_b = Transform(translation=(tx, 0.0, 0.0), rotation_xyz_deg=rotation_b)
        world_a = transform_aabb(bbox10, transform_a)
        world_b = transform_aabb(bbox_b, transform_b)
        raw_intersection = (
            aabb_intersection_volume(world_a, world_b)
            if intersection_volume is None
            else intersection_volume
        )
        raw_distance = (
            aabb_minimum_distance(world_a, world_b)
            if minimum_distance is None
            else minimum_distance
        )
        raw = RawGeometry(
            volume_a=bbox10.volume,
            volume_b=bbox_b.volume,
            intersection_volume=raw_intersection,
            minimum_distance=raw_distance,
            contains_a_in_b=False,
            contains_b_in_a=contains_b_in_a,
            aabb_overlap=aabb_overlap(world_a, world_b),
            boolean_status="ok",
            distance_status="ok" if raw_intersection <= policy.epsilon_volume(bbox10.volume, bbox_b.volume) else "skipped_positive_overlap",
        )
        return SyntheticFixture(
            name=name,
            object_a=box10_a,
            object_b=object_b,
            bbox_a=world_a,
            bbox_b=world_b,
            transform_a=transform_a,
            transform_b=transform_b,
            raw_geometry=raw,
            changed_value=tx,
            difficulty_tags=["axis_aligned", "primitive_fixture", *(tags or [])],
            metadata={"candidate_strategy": strategy},
        )

    return [
        fixture("separated_boxes", box10_b, bbox10, 20.0),
        fixture("touching_boxes", box10_b, bbox10, 10.0, tags=["contact_vs_interference", "near_boundary"]),
        fixture("tiny_overlap_boxes", box10_b, bbox10, 9.99, tags=["near_boundary", "tiny_overlap"]),
        fixture("near_miss_boxes", box10_b, bbox10, 10.5, tags=["near_boundary", "near_miss"]),
        fixture("clear_overlap_boxes", box10_b, bbox10, 7.0),
        fixture(
            "contained_boxes",
            box6_b,
            bbox6,
            0.0,
            intersection_volume=216.0,
            minimum_distance=0.0,
            contains_b_in_a=True,
            tags=["contained"],
        ),
        fixture(
            "rotated_disjoint_boxes",
            box10_b,
            bbox10,
            25.0,
            rotation_b=(0.0, 0.0, 45.0),
            tags=["rotated"],
            strategy="rotated_box_fixture",
        ),
    ]


def fixture_geometry_records(policy: LabelPolicy, config_hash: str) -> list[GeometryRecord]:
    records: list[GeometryRecord] = []
    for index, fixture in enumerate(synthetic_fixtures(policy), start=1):
        group = _fixture_group(index, fixture.name)
        labels, diagnostics = derive_labels(fixture.raw_geometry, policy)
        assembly = TwoObjectAssembly(
            fixture.object_a,
            fixture.object_b,
            fixture.transform_a,
            fixture.transform_b,
        )
        transform_hash = sha256_json(
            {
                "transform_a": fixture.transform_a.model_dump(mode="json"),
                "transform_b": fixture.transform_b.model_dump(mode="json"),
                "fixture": fixture.name,
            }
        )
        object_hash = sha256_json(
            [fixture.object_a.hashes.object_hash, fixture.object_b.hashes.object_hash]
        )
        geometry_hash = sha256_json(
            {
                "object_hash": object_hash,
                "transform_hash": transform_hash,
                "labels": labels.model_dump(mode="json"),
                "policy": policy.model_dump(mode="json"),
            }
        )
        source_code_hash = sha256_text(assembly.object_code)
        records.append(
            GeometryRecord(
                geometry_id=f"geom_{index:06d}",
                source="synthetic",
                object_a_id=fixture.object_a.object_id,
                object_b_id=fixture.object_b.object_id,
                base_object_pair_id=group["base_object_pair_id"],
                assembly_group_id=group["assembly_group_id"],
                counterfactual_group_id=group["counterfactual_group_id"],
                variant_id=group["variant_id"],
                changed_parameter="transform_b.translation[0]",
                changed_value=fixture.changed_value,
                transform_a=fixture.transform_a,
                transform_b=fixture.transform_b,
                assembly_script=assembly.script(),
                labels=labels,
                diagnostics=diagnostics,
                difficulty_tags=fixture.difficulty_tags,
                label_policy=policy,
                hashes=Hashes(
                    source_code_hash=source_code_hash,
                    object_hash=object_hash,
                    transform_hash=transform_hash,
                    geometry_hash=geometry_hash,
                    config_hash=config_hash,
                    prompt_hash=None,
                ),
                metadata={
                    **fixture.metadata,
                    "generator_ids": [GENERATOR_ID],
                    "artifact_ids": ArtifactIds().model_dump(mode="json"),
                    "bbox_a": fixture.bbox_a.to_schema().model_dump(mode="json"),
                    "bbox_b": fixture.bbox_b.to_schema().model_dump(mode="json"),
                    "fixture_name": fixture.name,
                },
            )
        )
    return records


def _fixture_group(index: int, name: str) -> dict[str, str]:
    sweep_names = {
        "separated_boxes": "v01",
        "touching_boxes": "v02",
        "tiny_overlap_boxes": "v03",
        "near_miss_boxes": "v04",
        "clear_overlap_boxes": "v05",
    }
    if name in sweep_names:
        return {
            "base_object_pair_id": "pair_000001",
            "assembly_group_id": "asmgrp_000001",
            "counterfactual_group_id": "cfg_000001",
            "variant_id": f"cfg_000001_{sweep_names[name]}",
        }
    if name == "contained_boxes":
        return {
            "base_object_pair_id": "pair_000002",
            "assembly_group_id": "asmgrp_000002",
            "counterfactual_group_id": "cfg_000002",
            "variant_id": "cfg_000002_v01",
        }
    if name == "rotated_disjoint_boxes":
        return {
            "base_object_pair_id": "pair_000003",
            "assembly_group_id": "asmgrp_000003",
            "counterfactual_group_id": "cfg_000003",
            "variant_id": "cfg_000003_v01",
        }
    return {
        "base_object_pair_id": f"pair_{index:06d}",
        "assembly_group_id": f"asmgrp_{index:06d}",
        "counterfactual_group_id": f"cfg_{index:06d}",
        "variant_id": f"cfg_{index:06d}_v01",
    }
