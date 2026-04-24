"""Tolerance-aware fit prompt generation."""

from __future__ import annotations

from intersectionqa.enums import Split, TaskType
from intersectionqa.prompts.common import object_code_from_script, public_row, transforms_text
from intersectionqa.schema import GeometryRecord, PublicTaskRow

TEMPLATE_VERSION = "tolerance_fit_v01"
ALLOWED_ANSWERS = {"yes", "no"}
DEFAULT_REQUIRED_CLEARANCE_MM = 1.0


def tolerance_fit_answer(
    record: GeometryRecord,
    required_clearance_mm: float = DEFAULT_REQUIRED_CLEARANCE_MM,
) -> str:
    distance = record.labels.minimum_distance
    if distance is None:
        raise ValueError("tolerance-fit prompt requires minimum_distance")
    if distance <= record.label_policy.epsilon_distance_mm:
        return "no"
    return "yes" if distance >= required_clearance_mm else "no"


def make_tolerance_fit_prompt(
    record: GeometryRecord,
    required_clearance_mm: float = DEFAULT_REQUIRED_CLEARANCE_MM,
) -> str:
    return f"""You are given two CadQuery object-construction functions and assembly transforms.

Assume:
- Units are millimetres.
- Euler rotations are XYZ order, in degrees.
- The required clearance is {required_clearance_mm:g} mm.
- Positive-volume overlap and touching both fail the requirement.
- Do not execute code.

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Question: Does the transformed assembly satisfy the required clearance?

Answer with exactly one token: yes or no"""


def materialize_tolerance_fit_row(record: GeometryRecord, row_number: int, split: str) -> PublicTaskRow:
    required_clearance_mm = DEFAULT_REQUIRED_CLEARANCE_MM
    return public_row(
        record=record,
        task_type=TaskType.TOLERANCE_FIT,
        answer=tolerance_fit_answer(record, required_clearance_mm),
        prompt=make_tolerance_fit_prompt(record, required_clearance_mm),
        row_number=row_number,
        split=Split(split),
        template_version=TEMPLATE_VERSION,
        extras={"required_clearance_mm": required_clearance_mm},
    )
