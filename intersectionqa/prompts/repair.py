"""Conservative repair-direction prompt generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intersectionqa.enums import Relation, Split, TaskType
from intersectionqa.prompts.common import object_code_from_script, public_row, transforms_text
from intersectionqa.schema import BoundingBox, GeometryRecord, PublicTaskRow

TEMPLATE_VERSION = "repair_direction_v01"
TRANSLATION_TEMPLATE_VERSION = "repair_translation_v01"
REPAIR_POLICY_NAME = "conservative_aabb_separating_translation_v01"
ALLOWED_ANSWERS = {"+x", "-x", "+y", "-y", "+z", "-z"}

_AXES = ("x", "y", "z")
_TIE_BREAK_ORDER = ("+x", "-x", "+y", "-y", "+z", "-z")


@dataclass(frozen=True)
class RepairMove:
    direction: str
    magnitude_mm: float
    translation_vector_mm: tuple[float, float, float]

    def to_metadata(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "magnitude_mm": self.magnitude_mm,
            "translation_vector_mm": list(self.translation_vector_mm),
        }


def repair_direction_answer(record: GeometryRecord) -> str:
    return repair_plan(record).direction


def repair_translation_answer(record: GeometryRecord) -> str:
    return repair_translation_answer_from_move(repair_plan(record))


def repair_translation_answer_from_move(move: RepairMove) -> str:
    return f"{move.direction} {move.magnitude_mm:.6f}"


def repair_plan(record: GeometryRecord) -> RepairMove:
    candidates = repair_candidates(record)
    return min(
        candidates,
        key=lambda move: (
            move.magnitude_mm,
            _TIE_BREAK_ORDER.index(move.direction),
        ),
    )


def repair_candidates(record: GeometryRecord) -> list[RepairMove]:
    bbox_a = _bbox_from_metadata(record, "bbox_a")
    bbox_b = _bbox_from_metadata(record, "bbox_b")
    clearance = float(record.label_policy.epsilon_distance_mm)
    candidates: list[RepairMove] = []
    for axis_index, axis_name in enumerate(_AXES):
        positive_delta = bbox_a.max[axis_index] - bbox_b.min[axis_index] + clearance
        negative_delta = bbox_a.min[axis_index] - bbox_b.max[axis_index] - clearance
        candidates.append(_move(f"+{axis_name}", axis_index, max(0.0, positive_delta)))
        candidates.append(_move(f"-{axis_name}", axis_index, min(0.0, negative_delta)))
    return candidates


def repair_metadata(record: GeometryRecord) -> dict[str, object]:
    selected = repair_plan(record)
    return {
        "repair_policy": REPAIR_POLICY_NAME,
        "repair_policy_note": (
            "Selects the smallest single-axis translation that separates stored "
            "world-space AABBs by label_policy.epsilon_distance_mm; ties use "
            "+x, -x, +y, -y, +z, -z order."
        ),
        "movable_object": "object_b",
        "fixed_object": "object_a",
        "candidate_direction_labels": list(_TIE_BREAK_ORDER),
        "selected_direction": selected.direction,
        "selected_magnitude_mm": selected.magnitude_mm,
        "selected_translation_vector_mm": list(selected.translation_vector_mm),
        "candidate_moves": [
            candidate.to_metadata() for candidate in repair_candidates(record)
        ],
    }


def make_repair_direction_prompt(record: GeometryRecord) -> str:
    return f"""You are given two CadQuery object-construction functions and assembly transforms.

Assume:
- Units are millimetres.
- Euler rotations are XYZ order, in degrees.
- object_a is fixed.
- object_b is movable by translation only.
- Choose the signed world-axis direction for moving object_b that repairs
  positive-volume interference with the smallest conservative single-axis move.
- Do not execute code.

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Question: Which signed axis direction should object_b move?

Answer with exactly one token: +x, -x, +y, -y, +z, or -z"""


def make_repair_translation_prompt(record: GeometryRecord) -> str:
    return f"""You are given two CadQuery object-construction functions and assembly transforms.

Assume:
- Units are millimetres.
- Euler rotations are XYZ order, in degrees.
- object_a is fixed.
- object_b is movable by translation only.
- Choose the signed world-axis direction and movement magnitude for object_b
  that repairs positive-volume interference with the smallest conservative
  single-axis move.
- The policy is conservative: separate the stored world-space AABBs by the
  label-policy contact tolerance.
- Do not execute code.

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Question: What signed axis direction and magnitude should object_b move?

Answer with exactly two tokens: one of +x, -x, +y, -y, +z, -z followed by a non-negative decimal magnitude in millimetres with six digits after the decimal point."""


def materialize_repair_direction_row(
    record: GeometryRecord,
    row_number: int,
    split: str,
) -> PublicTaskRow | None:
    if record.labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}:
        return None
    plan = repair_plan(record)
    return public_row(
        record=record,
        task_type=TaskType.REPAIR_DIRECTION,
        answer=plan.direction,
        prompt=make_repair_direction_prompt(record),
        row_number=row_number,
        split=Split(split),
        template_version=TEMPLATE_VERSION,
        extras=repair_metadata(record),
    )


def materialize_repair_translation_row(
    record: GeometryRecord,
    row_number: int,
    split: str,
) -> PublicTaskRow | None:
    if record.labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}:
        return None
    plan = repair_plan(record)
    return public_row(
        record=record,
        task_type=TaskType.REPAIR_TRANSLATION,
        answer=repair_translation_answer_from_move(plan),
        prompt=make_repair_translation_prompt(record),
        row_number=row_number,
        split=Split(split),
        template_version=TRANSLATION_TEMPLATE_VERSION,
        extras=repair_metadata(record),
    )


def _bbox_from_metadata(record: GeometryRecord, key: str) -> BoundingBox:
    value: Any = record.metadata.get(key)
    if value is None:
        raise ValueError(f"repair_direction requires {key} metadata")
    return BoundingBox.model_validate(value)


def _move(direction: str, axis_index: int, delta: float) -> RepairMove:
    vector = [0.0, 0.0, 0.0]
    vector[axis_index] = float(delta)
    return RepairMove(
        direction=direction,
        magnitude_mm=abs(float(delta)),
        translation_vector_mm=(vector[0], vector[1], vector[2]),
    )
