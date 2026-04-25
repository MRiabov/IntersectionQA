"""Ranking prompt generation for counterfactual variants."""

from __future__ import annotations

from intersectionqa.enums import Split, TaskType
from intersectionqa.prompts.common import object_code_from_script, public_group_row, transforms_text
from intersectionqa.schema import GeometryRecord, PublicTaskRow

TEMPLATE_VERSION = "ranking_normalized_intersection_v01"
LETTERS = "ABCDE"


def ranking_records(records: list[GeometryRecord]) -> list[GeometryRecord]:
    eligible = [
        record
        for record in sorted(records, key=lambda item: item.geometry_id)
        if record.labels.normalized_intersection is not None
    ]
    return eligible[:5] if len(eligible) >= 3 else []


def ranking_answer(records: list[GeometryRecord]) -> str:
    indexed = list(enumerate(records))
    ordered = sorted(
        indexed,
        key=lambda item: (-(item[1].labels.normalized_intersection or 0.0), LETTERS[item[0]]),
    )
    return "".join(LETTERS[index] for index, _ in ordered)


def make_ranking_prompt(records: list[GeometryRecord]) -> str:
    variants = "\n\n".join(
        f"Variant {LETTERS[index]} ({record.variant_id or record.geometry_id}):\n{transforms_text(record)}"
        for index, record in enumerate(records)
    )
    return f"""You are given one CadQuery object-pair definition and several assembly transform variants.

Assume:
- Units are millimetres.
- Euler rotations are XYZ order, in degrees.
- Rank by normalized intersection volume, largest first.
- Normalized intersection is intersection_volume / min(volume(object_a), volume(object_b)).
- Variants with equal normalized intersection are ordered alphabetically by letter.
- Do not execute code.

Object code:
```python
{object_code_from_script(records[0].assembly_script).strip()}
```

{variants}

Question: Rank the variants from largest to smallest normalized intersection volume.

Answer with exactly one compact letter string using each shown letter once."""


def materialize_ranking_row(records: list[GeometryRecord], row_number: int, split: str) -> PublicTaskRow | None:
    selected = ranking_records(records)
    if not selected:
        return None
    answer = ranking_answer(selected)
    prompt = make_ranking_prompt(selected)
    group_id = selected[0].counterfactual_group_id or selected[0].assembly_group_id
    return public_group_row(
        records=selected,
        task_type=TaskType.RANKING_NORMALIZED_INTERSECTION,
        answer=answer,
        prompt=prompt,
        row_number=row_number,
        split=Split(split),
        template_version=TEMPLATE_VERSION,
        derived_variant_id=f"{group_id}_ranking_{row_number:06d}",
        changed_value=[record.changed_value for record in selected],
        script="\n\n".join(record.assembly_script for record in selected),
        extras={
            "ranking_target": "normalized_intersection",
            "tie_break": "alphabetical_variant_letter",
            "variant_values": {
                LETTERS[index]: {
                    "geometry_id": record.geometry_id,
                    "variant_id": record.variant_id,
                    "normalized_intersection": record.labels.normalized_intersection,
                }
                for index, record in enumerate(selected)
            },
        },
    )


def materialize_ranking_rows(records: list[GeometryRecord], row_number: int, split: str) -> list[PublicTaskRow]:
    row = materialize_ranking_row(records, row_number, split)
    return [row] if row is not None else []
