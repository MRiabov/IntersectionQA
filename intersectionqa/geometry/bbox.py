"""Axis-aligned bounding-box primitives and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

from intersectionqa.schema import BoundingBox, Transform


@dataclass(frozen=True)
class AABB:
    min: tuple[float, float, float]
    max: tuple[float, float, float]

    def to_schema(self) -> BoundingBox:
        return BoundingBox(min=self.min, max=self.max)

    @property
    def volume(self) -> float:
        dx = self.max[0] - self.min[0]
        dy = self.max[1] - self.min[1]
        dz = self.max[2] - self.min[2]
        return dx * dy * dz


def aabb_overlap(a: AABB, b: AABB) -> bool:
    return all(a.min[i] <= b.max[i] and b.min[i] <= a.max[i] for i in range(3))


def aabb_intersection_volume(a: AABB, b: AABB) -> float:
    spans = [min(a.max[i], b.max[i]) - max(a.min[i], b.min[i]) for i in range(3)]
    if any(span <= 0.0 for span in spans):
        return 0.0
    return spans[0] * spans[1] * spans[2]


def aabb_minimum_distance(a: AABB, b: AABB) -> float:
    squared = 0.0
    for i in range(3):
        if a.max[i] < b.min[i]:
            gap = b.min[i] - a.max[i]
        elif b.max[i] < a.min[i]:
            gap = a.min[i] - b.max[i]
        else:
            gap = 0.0
        squared += gap * gap
    return squared**0.5


def box_aabb(width: float, depth: float, height: float) -> AABB:
    return AABB(
        min=(-width / 2.0, -depth / 2.0, -height / 2.0),
        max=(width / 2.0, depth / 2.0, height / 2.0),
    )


def transform_aabb(aabb: AABB, transform: Transform) -> AABB:
    corners = [
        (x, y, z)
        for x in (aabb.min[0], aabb.max[0])
        for y in (aabb.min[1], aabb.max[1])
        for z in (aabb.min[2], aabb.max[2])
    ]
    rx, ry, rz = (radians(v) for v in transform.rotation_xyz_deg)
    cx, sx = cos(rx), sin(rx)
    cy, sy = cos(ry), sin(ry)
    cz, sz = cos(rz), sin(rz)

    transformed: list[tuple[float, float, float]] = []
    for x, y, z in corners:
        y, z = y * cx - z * sx, y * sx + z * cx
        x, z = x * cy + z * sy, -x * sy + z * cy
        x, y = x * cz - y * sz, x * sz + y * cz
        transformed.append(
            (
                x + transform.translation[0],
                y + transform.translation[1],
                z + transform.translation[2],
            )
        )
    return AABB(
        min=tuple(min(p[i] for p in transformed) for i in range(3)),
        max=tuple(max(p[i] for p in transformed) for i in range(3)),
    )
