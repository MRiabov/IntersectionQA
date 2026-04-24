"""Relation-classification prompt generation."""

from __future__ import annotations

from intersectionqa.prompts.common import object_code_from_script, public_row, transforms_text
from intersectionqa.schema import GeometryRecord, PublicTaskRow

TEMPLATE_VERSION = "relation_classification_v01"
ALLOWED_ANSWERS = {
    "disjoint",
    "touching",
    "near_miss",
    "intersecting",
    "contained",
    "invalid",
}


def make_relation_prompt(record: GeometryRecord) -> str:
    return f"""You are given two CadQuery object-construction functions and assembly transforms.

Assume:
- Units are millimetres.
- Euler rotations are XYZ order, in degrees.
- Interference means positive-volume overlap.
- Touching at a face, edge, or point is not interference.
- Do not execute code.

Choose the relation after transforms:
- disjoint: positive clearance above the near-miss threshold
- touching: no positive-volume overlap, but contact within tolerance
- near_miss: no contact or overlap, but small positive clearance
- intersecting: positive-volume overlap
- contained: one solid is fully inside the other
- invalid: geometry failed or the label is unavailable

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Answer with exactly one label."""


def materialize_relation_row(record: GeometryRecord, row_number: int, split: str) -> PublicTaskRow:
    return public_row(
        record=record,
        task_type="relation_classification",
        answer=record.labels.relation,
        prompt=make_relation_prompt(record),
        row_number=row_number,
        split=split,
        template_version=TEMPLATE_VERSION,
    )
