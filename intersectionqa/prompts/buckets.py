"""Bucket prompt generation."""

from __future__ import annotations

from intersectionqa.enums import Split, TaskType
from intersectionqa.geometry.labels import VOLUME_BUCKETS, volume_bucket
from intersectionqa.prompts.common import object_code_from_script, public_row, transforms_text
from intersectionqa.schema import GeometryRecord, PublicTaskRow

TEMPLATE_VERSION = "volume_bucket_v01"
ALLOWED_ANSWERS = set(VOLUME_BUCKETS)
CLEARANCE_TEMPLATE_VERSION = "clearance_bucket_v01"
CLEARANCE_BUCKETS = [
    "intersecting",
    "touching",
    "(0, 0.1]",
    "(0.1, 1]",
    "(1, 5]",
    ">5",
]
CLEARANCE_ALLOWED_ANSWERS = set(CLEARANCE_BUCKETS)


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
        task_type=TaskType.VOLUME_BUCKET,
        answer=volume_bucket(record.labels, record.label_policy),
        prompt=make_volume_bucket_prompt(record),
        row_number=row_number,
        split=Split(split),
        template_version=TEMPLATE_VERSION,
        extras={"volume_bucket_boundaries": VOLUME_BUCKETS},
    )


def clearance_bucket(record: GeometryRecord) -> str:
    if record.labels.intersection_volume is not None and record.labels.volume_a is not None and record.labels.volume_b is not None:
        epsilon = record.label_policy.epsilon_volume(record.labels.volume_a, record.labels.volume_b)
        if record.labels.intersection_volume > epsilon:
            return "intersecting"
    distance = record.labels.minimum_distance
    if distance is None:
        raise ValueError("clearance bucket requires minimum_distance")
    if distance <= record.label_policy.epsilon_distance_mm:
        return "touching"
    if distance <= 0.1:
        return "(0, 0.1]"
    if distance <= 1.0:
        return "(0.1, 1]"
    if distance <= 5.0:
        return "(1, 5]"
    return ">5"


def make_clearance_bucket_prompt(record: GeometryRecord) -> str:
    buckets = "\n".join(CLEARANCE_BUCKETS)
    return f"""You are given two CadQuery object-construction functions and assembly transforms.

Assume:
- Units are millimetres.
- Euler rotations are XYZ order, in degrees.
- Clearance is the minimum distance between object_a and object_b after transforms.
- Positive-volume overlap is reported as intersecting.
- Contact within tolerance is reported as touching.
- Do not execute code.

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Question: Which bucket contains the clearance relation or distance?

Allowed answers:
{buckets}

Answer with exactly one bucket string."""


def materialize_clearance_bucket_row(record: GeometryRecord, row_number: int, split: str) -> PublicTaskRow:
    return public_row(
        record=record,
        task_type=TaskType.CLEARANCE_BUCKET,
        answer=clearance_bucket(record),
        prompt=make_clearance_bucket_prompt(record),
        row_number=row_number,
        split=Split(split),
        template_version=CLEARANCE_TEMPLATE_VERSION,
        extras={"clearance_bucket_boundaries_mm": CLEARANCE_BUCKETS},
    )
