"""Volume-bucket prompt generation."""

from __future__ import annotations

from intersectionqa.geometry.labels import VOLUME_BUCKETS, volume_bucket
from intersectionqa.prompts.common import object_code_from_script, public_row, transforms_text
from intersectionqa.schema import GeometryRecord, PublicTaskRow

TEMPLATE_VERSION = "volume_bucket_v01"
ALLOWED_ANSWERS = set(VOLUME_BUCKETS)


def make_volume_bucket_prompt(record: GeometryRecord) -> str:
    buckets = "\n".join(VOLUME_BUCKETS)
    return f"""You are given two CadQuery object-construction functions and assembly transforms.

Assume:
- Units are millimetres.
- Euler rotations are XYZ order, in degrees.
- Normalized intersection means:
  intersection_volume / min(volume(object_a), volume(object_b))
- Touching without positive-volume overlap has normalized intersection 0.
- Do not execute code.

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Question: Which bucket contains the normalized intersection volume?

Allowed answers:
{buckets}

Answer with exactly one bucket string."""


def materialize_volume_bucket_row(record: GeometryRecord, row_number: int, split: str) -> PublicTaskRow:
    return public_row(
        record=record,
        task_type="volume_bucket",
        answer=volume_bucket(record.labels, record.label_policy),
        prompt=make_volume_bucket_prompt(record),
        row_number=row_number,
        split=split,
        template_version=TEMPLATE_VERSION,
        extras={"volume_bucket_boundaries": VOLUME_BUCKETS},
    )
