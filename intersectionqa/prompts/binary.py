"""Binary interference prompt generation."""

from __future__ import annotations

from intersectionqa.geometry.labels import binary_answer
from intersectionqa.prompts.common import object_code_from_script, public_row, transforms_text
from intersectionqa.schema import GeometryRecord, PublicTaskRow

TEMPLATE_VERSION = "binary_interference_v01"
ALLOWED_ANSWERS = {"yes", "no"}


def make_binary_prompt(record: GeometryRecord) -> str:
    return f"""You are given two CadQuery object-construction functions and assembly transforms.

Assume:
- Units are millimetres.
- Euler rotations are XYZ order, in degrees.
- "interference" means positive-volume overlap.
- Touching at a face, edge, or point is not interference.
- Do not execute code.

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Question: After the transforms are applied, do object_a and object_b have positive-volume interference?

Answer with exactly one token: yes or no"""


def materialize_binary_row(record: GeometryRecord, row_number: int, split: str) -> PublicTaskRow:
    return public_row(
        record=record,
        task_type="binary_interference",
        answer=binary_answer(record.labels.relation),
        prompt=make_binary_prompt(record),
        row_number=row_number,
        split=split,
        template_version=TEMPLATE_VERSION,
    )
