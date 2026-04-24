"""Rigid transform helpers for v0.1 records."""

from __future__ import annotations

from intersectionqa.schema import Transform

IDENTITY_TRANSFORM = Transform(
    translation=(0.0, 0.0, 0.0),
    rotation_xyz_deg=(0.0, 0.0, 0.0),
    rotation_order="XYZ",
)


def transform_to_dict(transform: Transform) -> dict[str, object]:
    return {
        "translation": list(transform.translation),
        "rotation_xyz_deg": list(transform.rotation_xyz_deg),
        "rotation_order": transform.rotation_order,
    }


def format_transform_block(transform_a: Transform, transform_b: Transform) -> str:
    return "\n".join(
        [
            "object_a:",
            f"  translation: {list(transform_a.translation)}",
            f"  rotation_xyz_deg: {list(transform_a.rotation_xyz_deg)}",
            f"  rotation_order: {transform_a.rotation_order}",
            "object_b:",
            f"  translation: {list(transform_b.translation)}",
            f"  rotation_xyz_deg: {list(transform_b.rotation_xyz_deg)}",
            f"  rotation_order: {transform_b.rotation_order}",
        ]
    )
