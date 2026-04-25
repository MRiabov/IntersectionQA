"""Counterfactual prompt formats derived from geometry groups."""

from __future__ import annotations

from intersectionqa.enums import Relation, Split, TaskType
from intersectionqa.prompts.common import object_code_from_script, public_group_row, transforms_text
from intersectionqa.schema import GeometryRecord, PublicTaskRow

TEMPLATE_VERSION = "counterfactual_pairwise_interference_v01"
ALLOWED_ANSWERS = {"A", "B", "both", "neither"}


def pairwise_records(records: list[GeometryRecord]) -> tuple[GeometryRecord, GeometryRecord] | None:
    ordered = sorted(records, key=lambda item: item.geometry_id)
    positives = [record for record in ordered if _has_interference(record)]
    negatives = [record for record in ordered if not _has_interference(record)]
    if positives and negatives:
        return negatives[0], positives[0]
    if len(ordered) >= 2:
        return ordered[0], ordered[1]
    return None


def balanced_pairwise_records(records: list[GeometryRecord]) -> list[tuple[GeometryRecord, GeometryRecord]]:
    """Build deterministic pairwise comparisons for all answer classes available in a group."""
    ordered = sorted(records, key=lambda item: item.geometry_id)
    positives = [record for record in ordered if _has_interference(record)]
    negatives = [record for record in ordered if not _has_interference(record)]
    pairs: list[tuple[GeometryRecord, GeometryRecord]] = []
    if positives and negatives:
        pairs.append((positives[0], negatives[0]))  # A
        pairs.append((negatives[0], positives[0]))  # B
    if len(positives) >= 2:
        pairs.append((positives[0], positives[1]))  # both
    if len(negatives) >= 2:
        pairs.append((negatives[0], negatives[1]))  # neither
    if pairs:
        return pairs
    fallback = pairwise_records(records)
    return [fallback] if fallback is not None else []


def pairwise_answer(record_a: GeometryRecord, record_b: GeometryRecord) -> str:
    a_interferes = _has_interference(record_a)
    b_interferes = _has_interference(record_b)
    if a_interferes and b_interferes:
        return "both"
    if a_interferes:
        return "A"
    if b_interferes:
        return "B"
    return "neither"


def make_pairwise_prompt(record_a: GeometryRecord, record_b: GeometryRecord) -> str:
    return f"""You are given one CadQuery object-pair definition and two assembly transform variants.

Assume:
- Units are millimetres.
- Euler rotations are XYZ order, in degrees.
- Interference means positive-volume overlap.
- Touching at a face, edge, or point is not interference.
- Do not execute code.

Object code:
```python
{object_code_from_script(record_a.assembly_script).strip()}
```

Variant A ({record_a.variant_id or record_a.geometry_id}):
{transforms_text(record_a)}

Variant B ({record_b.variant_id or record_b.geometry_id}):
{transforms_text(record_b)}

Question: Which variant or variants have positive-volume interference?

Allowed answers:
A
B
both
neither

Answer with exactly one allowed answer."""


def materialize_pairwise_row(records: list[GeometryRecord], row_number: int, split: str) -> PublicTaskRow | None:
    pair = pairwise_records(records)
    if pair is None:
        return None
    record_a, record_b = pair
    return _materialize_pairwise_pair(record_a, record_b, row_number, split)


def materialize_pairwise_rows(records: list[GeometryRecord], row_number: int, split: str) -> list[PublicTaskRow]:
    rows: list[PublicTaskRow] = []
    for offset, (record_a, record_b) in enumerate(balanced_pairwise_records(records)):
        rows.append(_materialize_pairwise_pair(record_a, record_b, row_number + offset, split))
    return rows


def _materialize_pairwise_pair(
    record_a: GeometryRecord,
    record_b: GeometryRecord,
    row_number: int,
    split: str,
) -> PublicTaskRow:
    answer = pairwise_answer(record_a, record_b)
    prompt = make_pairwise_prompt(record_a, record_b)
    group_id = record_a.counterfactual_group_id or record_a.assembly_group_id
    return public_group_row(
        records=[record_a, record_b],
        task_type=TaskType.PAIRWISE_INTERFERENCE,
        answer=answer,
        prompt=prompt,
        row_number=row_number,
        split=Split(split),
        template_version=TEMPLATE_VERSION,
        derived_variant_id=f"{group_id}_pairwise_{row_number:06d}",
        changed_value=[record_a.changed_value, record_b.changed_value],
        script="\n\n".join([record_a.assembly_script, record_b.assembly_script]),
        extras={
            "counterfactual_prompt_format": "pairwise_interference",
            "variant_labels": {
                "A": {
                    "geometry_id": record_a.geometry_id,
                    "variant_id": record_a.variant_id,
                    "relation": record_a.labels.relation,
                    "interferes": _has_interference(record_a),
                },
                "B": {
                    "geometry_id": record_b.geometry_id,
                    "variant_id": record_b.variant_id,
                    "relation": record_b.labels.relation,
                    "interferes": _has_interference(record_b),
                },
            },
        },
    )


def _has_interference(record: GeometryRecord) -> bool:
    return record.labels.relation in {Relation.INTERSECTING, Relation.CONTAINED}
