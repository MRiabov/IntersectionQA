"""Conservative repair-direction prompt generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from intersectionqa.enums import Relation, Split, TaskType
from intersectionqa.geometry.cadquery_exec import apply_transform, measure_shape_pair, object_to_shape
from intersectionqa.geometry.labels import derive_labels
from intersectionqa.geometry.transforms import IDENTITY_TRANSFORM
from intersectionqa.prompts.common import object_code_from_script, public_row, transforms_text
from intersectionqa.schema import BoundingBox, GeometryRecord, PublicTaskRow, Transform

TEMPLATE_VERSION = "repair_direction_v01"
TRANSLATION_TEMPLATE_VERSION = "repair_translation_v01"
REPAIR_POLICY_NAME = "conservative_aabb_separating_translation_v01"
AXIS_ALIGNED_REPAIR_TEMPLATE_VERSION = "axis_aligned_repair_v01"
AXIS_ALIGNED_REPAIR_VECTOR_TEMPLATE_VERSION = "axis_aligned_repair_vector_v01"
AXIS_ALIGNED_REPAIR_PROGRAM_TEMPLATE_VERSION = "axis_aligned_repair_program_v01"
TARGET_CLEARANCE_REPAIR_TEMPLATE_VERSION = "target_clearance_repair_v01"
TARGET_CLEARANCE_MOVE_TEMPLATE_VERSION = "target_clearance_move_v01"
TARGET_CONTACT_MOVE_TEMPLATE_VERSION = "target_contact_move_v01"
CENTROID_DISTANCE_MOVE_TEMPLATE_VERSION = "centroid_distance_move_v01"
EDIT_CANDIDATE_SELECTION_TEMPLATE_VERSION = "edit_candidate_selection_v01"
EDIT_CANDIDATE_RANKING_TEMPLATE_VERSION = "edit_candidate_ranking_v01"
EXACT_REPAIR_POLICY_NAME = "exact_axis_aligned_cardinal_search_v01"
CENTROID_DISTANCE_POLICY_NAME = "exact_centroid_direction_move_v01"
TARGET_CLEARANCE_MM = 1.0
CENTROID_DISTANCE_DELTA_MM = 5.0
NUMERIC_LABEL_DECIMAL_MM = 0.1
NUMERIC_ACCEPTANCE_TOLERANCE_MM = 0.15
ALLOWED_ANSWERS = {"+x", "-x", "+y", "-y", "+z", "-z"}

_AXES = ("x", "y", "z")
_TIE_BREAK_ORDER = ("+x", "-x", "+y", "-y", "+z", "-z")
_DIRECTION_VECTORS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}
_CANDIDATE_LABELS = ("A", "B", "C", "D")
_EXACT_CANDIDATE_CACHE: dict[tuple[str, float], list["ExactRepairMove"]] = {}


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


@dataclass(frozen=True)
class ExactRepairMove:
    direction: str
    exact_magnitude_mm: float
    label_magnitude_mm: float
    translation_vector_mm: tuple[float, float, float]
    final_relation: str
    final_clearance_mm: float | None
    final_intersection_volume: float | None
    satisfies_target: bool

    def answer(self) -> str:
        return f"direction={self.direction}, distance_mm={self.label_magnitude_mm:.1f}"

    def to_metadata(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "exact_magnitude_mm": self.exact_magnitude_mm,
            "label_magnitude_mm": self.label_magnitude_mm,
            "translation_vector_mm": list(self.translation_vector_mm),
            "final_relation": self.final_relation,
            "final_clearance_mm": self.final_clearance_mm,
            "final_intersection_volume": self.final_intersection_volume,
            "satisfies_target": self.satisfies_target,
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
        "edit_split_group": _edit_split_group(record, "conservative_repair_v01"),
        "edit_family": "conservative_axis_aligned_repair",
        **_edit_counterfactual_metadata(
            record,
            edit_family="conservative_axis_aligned_repair",
            target_descriptor="non_intersection",
        ),
        "candidate_direction_labels": list(_TIE_BREAK_ORDER),
        "selected_direction": selected.direction,
        "selected_magnitude_mm": selected.magnitude_mm,
        "selected_translation_vector_mm": list(selected.translation_vector_mm),
        "candidate_moves": [
            candidate.to_metadata() for candidate in repair_candidates(record)
        ],
    }


def exact_axis_aligned_repair_plan(
    record: GeometryRecord,
    *,
    target_clearance_mm: float = 0.0,
) -> ExactRepairMove:
    candidates = exact_axis_aligned_repair_candidates(
        record,
        target_clearance_mm=target_clearance_mm,
    )
    return min(
        candidates,
        key=lambda move: (
            move.exact_magnitude_mm,
            _TIE_BREAK_ORDER.index(move.direction),
        ),
    )


def exact_axis_aligned_repair_candidates(
    record: GeometryRecord,
    *,
    target_clearance_mm: float = 0.0,
) -> list[ExactRepairMove]:
    if target_clearance_mm < 0.0 or not math.isfinite(target_clearance_mm):
        raise ValueError("target_clearance_mm must be finite and non-negative")
    cache_key = (record.hashes.geometry_hash or record.geometry_id, float(target_clearance_mm))
    if cache_key in _EXACT_CANDIDATE_CACHE:
        return list(_EXACT_CANDIDATE_CACHE[cache_key])
    shape_a, shape_b = _placed_shapes(record)
    candidates: list[ExactRepairMove] = []
    for direction in _TIE_BREAK_ORDER:
        exact_magnitude = _minimum_exact_magnitude(
            record,
            shape_a,
            shape_b,
            direction,
            target_clearance_mm,
        )
        label_magnitude = _label_magnitude(exact_magnitude)
        candidates.append(
            _exact_move_from_magnitude(
                record,
                shape_a,
                shape_b,
                direction,
                exact_magnitude,
                label_magnitude,
                target_clearance_mm,
            )
        )
    _EXACT_CANDIDATE_CACHE[cache_key] = candidates
    return list(candidates)


def exact_repair_metadata(
    record: GeometryRecord,
    *,
    target_clearance_mm: float,
) -> dict[str, object]:
    candidates = exact_axis_aligned_repair_candidates(
        record,
        target_clearance_mm=target_clearance_mm,
    )
    selected = min(
        candidates,
        key=lambda move: (
            move.exact_magnitude_mm,
            _TIE_BREAK_ORDER.index(move.direction),
        ),
    )
    ordered = sorted(
        candidates,
        key=lambda move: (
            move.exact_magnitude_mm,
            _TIE_BREAK_ORDER.index(move.direction),
        ),
    )
    best = ordered[0].exact_magnitude_mm
    second = ordered[1].exact_magnitude_mm if len(ordered) > 1 else None
    best_direction_margin_mm = None if second is None else second - best
    ambiguous = best_direction_margin_mm is not None and best_direction_margin_mm < NUMERIC_ACCEPTANCE_TOLERANCE_MM
    target_type = "non_intersection" if target_clearance_mm <= 0.0 else "clearance"
    edit_family = (
        "axis_aligned_target_clearance_repair"
        if target_clearance_mm > 0.0
        else "axis_aligned_intersection_repair"
    )
    metadata = {
        "edit_policy": EXACT_REPAIR_POLICY_NAME,
        "repair_policy": EXACT_REPAIR_POLICY_NAME,
        "fixed_object": "object_a",
        "movable_object": "object_b",
        "edit_split_group": _edit_split_group(
            record,
            f"{target_type}_target_{target_clearance_mm:.1f}_v01",
        ),
        "edit_family": edit_family,
        **_edit_counterfactual_metadata(
            record,
            edit_family=edit_family,
            target_descriptor=f"{target_type}:{target_clearance_mm:.1f}",
        ),
        "initial_state": _initial_state_metadata(record),
        "target": {
            "type": target_type,
            "target_clearance_mm": target_clearance_mm,
            "allow_touching": target_clearance_mm <= 0.0,
        },
        "allowed_edit": {
            "edit_type": "translation",
            "directions": list(_TIE_BREAK_ORDER),
            "rotation_allowed": False,
            "movable_only": "object_b",
        },
        "numeric_output_policy": {
            "label_decimal_mm": NUMERIC_LABEL_DECIMAL_MM,
            "acceptance_tolerance_mm": NUMERIC_ACCEPTANCE_TOLERANCE_MM,
        },
        "selected_direction": selected.direction,
        "selected_exact_magnitude_mm": selected.exact_magnitude_mm,
        "selected_magnitude_mm": selected.label_magnitude_mm,
        "selected_translation_vector_mm": list(selected.translation_vector_mm),
        "structured_answer": {
            "direction": selected.direction,
            "distance_mm": selected.label_magnitude_mm,
            "translation": list(selected.translation_vector_mm),
        },
        "verification": {
            "final_relation": selected.final_relation,
            "final_clearance_mm": selected.final_clearance_mm,
            "final_intersection_volume": selected.final_intersection_volume,
            "movement_magnitude": selected.label_magnitude_mm,
            "satisfies_target": selected.satisfies_target,
        },
        "edit_diagnostics": {
            "best_direction_margin_mm": best_direction_margin_mm,
            "num_valid_directions": len([item for item in candidates if item.satisfies_target]),
            "ambiguous": ambiguous,
            "difficulty": (
                "axis_aligned_target_clearance_repair"
                if target_clearance_mm > 0.0
                else "axis_aligned_intersection_repair"
            ),
            "sft_include": not ambiguous,
            "rl_include": True,
        },
        "candidate_moves": [candidate.to_metadata() for candidate in candidates],
    }
    return metadata


def candidate_edit_metadata(record: GeometryRecord) -> dict[str, object]:
    shape_a, shape_b = _placed_shapes(record)
    plan = exact_axis_aligned_repair_plan(
        record,
        target_clearance_mm=TARGET_CLEARANCE_MM,
    )
    base = plan.label_magnitude_mm
    under = max(0.0, _round_one_decimal(base - max(0.2, base * 0.25)))
    over = _round_one_decimal(base + max(0.5, base * 0.25))
    wrong_direction = next(direction for direction in _TIE_BREAK_ORDER if direction != plan.direction)
    wrong_candidates = exact_axis_aligned_repair_candidates(record, target_clearance_mm=TARGET_CLEARANCE_MM)
    wrong_exact = next(candidate for candidate in wrong_candidates if candidate.direction == wrong_direction)
    raw_candidates = [
        ("under_corrected", plan.direction, under),
        ("wrong_axis_or_sign", wrong_direction, wrong_exact.label_magnitude_mm),
        ("best_exact", plan.direction, base),
        ("large_valid", plan.direction, over),
    ]
    candidates = {
        label: _candidate_edit_item(
            record,
            shape_a,
            shape_b,
            label,
            strategy,
            direction,
            magnitude,
            TARGET_CLEARANCE_MM,
        )
        for label, (strategy, direction, magnitude) in zip(_CANDIDATE_LABELS, raw_candidates, strict=True)
    }
    ranking = sorted(candidates.items(), key=lambda item: _candidate_rank_key(item[0], item[1]))
    answer = ranking[0][0]
    return {
        "edit_policy": EXACT_REPAIR_POLICY_NAME,
        "fixed_object": "object_a",
        "movable_object": "object_b",
        "edit_split_group": _edit_split_group(
            record,
            f"candidate_target_clearance_{TARGET_CLEARANCE_MM:.1f}_v01",
        ),
        "edit_family": "target_clearance_candidate_set",
        **_edit_counterfactual_metadata(
            record,
            edit_family="target_clearance_candidate_set",
            target_descriptor=f"clearance:{TARGET_CLEARANCE_MM:.1f}",
        ),
        "initial_state": _initial_state_metadata(record),
        "target": {
            "type": "clearance",
            "target_clearance_mm": TARGET_CLEARANCE_MM,
            "allow_touching": False,
        },
        "allowed_edit": {
            "edit_type": "candidate_translation",
            "directions": list(_TIE_BREAK_ORDER),
            "rotation_allowed": False,
            "movable_only": "object_b",
        },
        "candidate_edits": candidates,
        "candidate_selection_answer": answer,
        "candidate_ranking_answer": "".join(label for label, _ in ranking),
    }


def target_clearance_move_metadata(record: GeometryRecord) -> dict[str, object] | None:
    return _axis_clearance_move_metadata(
        record,
        target_clearance_mm=TARGET_CLEARANCE_MM,
        edit_family="target_clearance_move",
        difficulty="axis_aligned_target_clearance_move",
        suffix=f"move_target_clearance_{TARGET_CLEARANCE_MM:.1f}_v01",
    )


def target_contact_move_metadata(record: GeometryRecord) -> dict[str, object] | None:
    return _axis_clearance_move_metadata(
        record,
        target_clearance_mm=0.0,
        edit_family="target_contact_move",
        difficulty="axis_aligned_target_contact_move",
        suffix="move_target_contact_v01",
    )


def centroid_distance_move_metadata(record: GeometryRecord) -> dict[str, object] | None:
    if record.labels.relation not in {Relation.DISJOINT, Relation.NEAR_MISS, Relation.TOUCHING}:
        return None
    shape_a, shape_b = _placed_shapes(record)
    center_a = _shape_center(shape_a)
    center_b = _shape_center(shape_b)
    initial_distance = _distance(center_a, center_b)
    direction = _unit_vector(tuple(center_b[index] - center_a[index] for index in range(3)))
    if direction is None:
        return None
    signed_exact_distance = CENTROID_DISTANCE_DELTA_MM
    signed_label_distance = _round_one_decimal(signed_exact_distance)
    target_distance = initial_distance + signed_exact_distance
    vector = tuple(component * signed_label_distance for component in direction)
    moved_b = apply_transform(
        shape_b,
        Transform(translation=vector, rotation_xyz_deg=(0.0, 0.0, 0.0)),
    )
    raw = measure_shape_pair(
        shape_a,
        moved_b,
        IDENTITY_TRANSFORM,
        IDENTITY_TRANSFORM,
        record.label_policy,
    )
    labels, _ = derive_labels(raw, record.label_policy)
    if labels.relation in {Relation.INTERSECTING, Relation.CONTAINED}:
        return None
    final_center_b = _shape_center(moved_b)
    final_distance = _distance(center_a, final_center_b)
    distance_error = abs(final_distance - target_distance)
    if distance_error > NUMERIC_ACCEPTANCE_TOLERANCE_MM:
        return None
    return {
        "edit_policy": CENTROID_DISTANCE_POLICY_NAME,
        "fixed_object": "object_a",
        "movable_object": "object_b",
        "edit_split_group": _edit_split_group(
            record,
            f"centroid_distance_target_{target_distance:.1f}_v01",
        ),
        "edit_family": "centroid_distance_move",
        **_edit_counterfactual_metadata(
            record,
            edit_family="centroid_distance_move",
            target_descriptor=f"centroid_distance:{target_distance:.1f}",
        ),
        "initial_state": {
            **_initial_state_metadata(record),
            "centroid_a": list(center_a),
            "centroid_b": list(center_b),
            "centroid_distance_mm": initial_distance,
        },
        "target": {
            "type": "centroid_distance",
            "target_centroid_distance_mm": target_distance,
            "preserve_non_intersection": True,
        },
        "allowed_edit": {
            "edit_type": "signed_centroid_direction_distance",
            "direction_vector": list(direction),
            "positive_distance_effect": "moves object_b farther from object_a along the centroid-to-centroid direction",
            "negative_distance_effect": "moves object_b closer to object_a along the centroid-to-centroid direction",
            "rotation_allowed": False,
            "movable_only": "object_b",
        },
        "numeric_output_policy": {
            "label_decimal_mm": NUMERIC_LABEL_DECIMAL_MM,
            "acceptance_tolerance_mm": NUMERIC_ACCEPTANCE_TOLERANCE_MM,
        },
        "selected_signed_exact_distance_mm": signed_exact_distance,
        "selected_signed_distance_mm": signed_label_distance,
        "selected_translation_vector_mm": list(vector),
        "structured_answer": {
            "distance_mm": signed_label_distance,
            "direction_vector": list(direction),
            "translation": list(vector),
        },
        "verification": {
            "final_relation": str(labels.relation),
            "final_centroid_distance_mm": final_distance,
            "centroid_distance_error_mm": distance_error,
            "final_clearance_mm": labels.minimum_distance,
            "movement_magnitude": abs(signed_label_distance),
            "satisfies_target": True,
        },
        "edit_diagnostics": {
            "difficulty": "centroid_distance_move",
            "move_kind": "farther",
            "sft_include": True,
            "rl_include": True,
        },
    }


def _axis_clearance_move_metadata(
    record: GeometryRecord,
    *,
    target_clearance_mm: float,
    edit_family: str,
    difficulty: str,
    suffix: str,
) -> dict[str, object] | None:
    if record.labels.minimum_distance is None:
        return None
    if record.labels.relation not in {Relation.DISJOINT, Relation.NEAR_MISS}:
        return None
    away_direction = _single_aabb_away_direction(record)
    if away_direction is None:
        return None
    initial_clearance = float(record.labels.minimum_distance)
    signed_exact_distance = target_clearance_mm - initial_clearance
    signed_label_distance = _round_one_decimal(signed_exact_distance)
    shape_a, shape_b = _placed_shapes(record)
    vector = _translation_vector(away_direction, signed_label_distance)
    moved_b = apply_transform(
        shape_b,
        Transform(translation=vector, rotation_xyz_deg=(0.0, 0.0, 0.0)),
    )
    raw = measure_shape_pair(
        shape_a,
        moved_b,
        IDENTITY_TRANSFORM,
        IDENTITY_TRANSFORM,
        record.label_policy,
    )
    labels, _ = derive_labels(raw, record.label_policy)
    final_clearance = labels.minimum_distance
    if labels.relation in {Relation.INTERSECTING, Relation.CONTAINED}:
        return None
    if final_clearance is None:
        return None
    clearance_error = abs(final_clearance - target_clearance_mm)
    if clearance_error > NUMERIC_ACCEPTANCE_TOLERANCE_MM:
        return None
    direction_label = "farther" if signed_label_distance > 0.0 else "closer" if signed_label_distance < 0.0 else "none"
    return {
        "edit_policy": EXACT_REPAIR_POLICY_NAME,
        "fixed_object": "object_a",
        "movable_object": "object_b",
        "edit_split_group": _edit_split_group(
            record,
            f"{suffix}_{away_direction}",
        ),
        "edit_family": edit_family,
        **_edit_counterfactual_metadata(
            record,
            edit_family=edit_family,
            target_descriptor=f"clearance:{target_clearance_mm:.1f}",
        ),
        "initial_state": _initial_state_metadata(record),
        "target": {
            "type": "clearance",
            "target_clearance_mm": target_clearance_mm,
            "allow_touching": target_clearance_mm <= 0.0,
        },
        "allowed_edit": {
            "edit_type": "signed_translation_distance",
            "direction": away_direction,
            "positive_distance_effect": "moves object_b farther from object_a",
            "negative_distance_effect": "moves object_b closer to object_a",
            "rotation_allowed": False,
            "movable_only": "object_b",
        },
        "numeric_output_policy": {
            "label_decimal_mm": NUMERIC_LABEL_DECIMAL_MM,
            "acceptance_tolerance_mm": NUMERIC_ACCEPTANCE_TOLERANCE_MM,
        },
        "selected_direction": away_direction,
        "selected_signed_exact_distance_mm": signed_exact_distance,
        "selected_signed_distance_mm": signed_label_distance,
        "selected_translation_vector_mm": list(vector),
        "structured_answer": {
            "distance_mm": signed_label_distance,
            "direction": away_direction,
            "translation": list(vector),
        },
        "verification": {
            "initial_clearance_mm": initial_clearance,
            "final_relation": str(labels.relation),
            "final_clearance_mm": final_clearance,
            "clearance_error_mm": clearance_error,
            "movement_magnitude": abs(signed_label_distance),
            "satisfies_target": True,
        },
        "edit_diagnostics": {
            "difficulty": difficulty,
            "move_kind": direction_label,
            "aabb_separation_direction": away_direction,
            "sft_include": True,
            "rl_include": True,
        },
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


def make_axis_aligned_repair_prompt(record: GeometryRecord) -> str:
    return f"""You are given two CadQuery object-construction functions and assembly transforms.

Object "object_a" is fixed. Object "object_b" may be translated.

Goal:
Move "object_b" by the smallest axis-aligned translation that removes positive-volume interference.

Allowed directions:
+x, -x, +y, -y, +z, -z

Definitions:
- Interference means positive-volume overlap.
- Touching is allowed.
- Use millimetres.
- Do not execute code.

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Final answer format:
direction=<one of +x,-x,+y,-y,+z,-z>, distance_mm=<one decimal>"""


def make_axis_aligned_repair_vector_prompt(record: GeometryRecord) -> str:
    return f"""You are given two CadQuery object-construction functions and assembly transforms.

Object "object_a" is fixed. Object "object_b" may be translated.

Goal:
Move "object_b" by the smallest axis-aligned translation vector that removes positive-volume interference.

Definitions:
- The vector is in world coordinates.
- Interference means positive-volume overlap.
- Touching is allowed.
- Use millimetres.
- Do not execute code.

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Final answer format:
dx=<one decimal>, dy=<one decimal>, dz=<one decimal>"""


def make_axis_aligned_repair_program_prompt(record: GeometryRecord) -> str:
    return f"""You are given two CadQuery object-construction functions and assembly transforms.

Object "object_a" is fixed. Object "object_b" may be translated.

Goal:
Write the single edit statement that moves "object_b" by the smallest axis-aligned translation that removes positive-volume interference.

Definitions:
- The translation tuple is in world coordinates.
- Interference means positive-volume overlap.
- Touching is allowed.
- Use millimetres.
- Do not execute code.

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Final answer format:
object_b = object_b.translate((<dx one decimal>, <dy one decimal>, <dz one decimal>))"""


def make_target_clearance_repair_prompt(record: GeometryRecord) -> str:
    return f"""You are given two CadQuery object-construction functions and assembly transforms.

Object "object_a" is fixed. Object "object_b" may be translated.

Goal:
Move "object_b" by the smallest axis-aligned translation such that the two objects have at least {TARGET_CLEARANCE_MM:.1f} mm clearance.

Allowed directions:
+x, -x, +y, -y, +z, -z

Definitions:
- Clearance is the minimum surface-to-surface distance between the two objects.
- If the objects intersect, clearance is invalid until interference is resolved.
- Use millimetres.
- Do not execute code.

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Final answer format:
direction=<one of +x,-x,+y,-y,+z,-z>, distance_mm=<one decimal>"""


def make_target_clearance_move_prompt(record: GeometryRecord, metadata: dict[str, object]) -> str:
    allowed_edit = metadata.get("allowed_edit")
    if not isinstance(allowed_edit, dict):
        raise ValueError("target clearance move metadata requires allowed_edit")
    direction = str(allowed_edit["direction"])
    return f"""The two CadQuery objects are currently non-intersecting.

Object "object_a" is fixed. Object "object_b" may be translated only along this signed world-axis direction:
{direction}

Goal:
Move "object_b" so that the minimum surface-to-surface clearance between the objects is {TARGET_CLEARANCE_MM:.1f} mm.

Definitions:
- Positive distance moves object_b farther from object_a along {direction}.
- Negative distance moves object_b closer to object_a along the opposite direction.
- Clearance is the exact minimum surface-to-surface distance.
- Use millimetres.
- Do not execute code.

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Final answer format:
distance_mm=<signed number with one decimal>"""


def make_target_contact_move_prompt(record: GeometryRecord, metadata: dict[str, object]) -> str:
    allowed_edit = metadata.get("allowed_edit")
    if not isinstance(allowed_edit, dict):
        raise ValueError("target contact move metadata requires allowed_edit")
    direction = str(allowed_edit["direction"])
    return f"""The two CadQuery objects are currently non-intersecting.

Object "object_a" is fixed. Object "object_b" may be translated only along this signed world-axis direction:
{direction}

Goal:
Move "object_b" until it just touches object_a without positive-volume overlap.

Definitions:
- Positive distance moves object_b farther from object_a along {direction}.
- Negative distance moves object_b closer to object_a along the opposite direction.
- Touching means zero clearance without positive-volume interference.
- Use millimetres.
- Do not execute code.

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Final answer format:
distance_mm=<signed number with one decimal>"""


def make_centroid_distance_move_prompt(record: GeometryRecord, metadata: dict[str, object]) -> str:
    allowed_edit = metadata.get("allowed_edit")
    target = metadata.get("target")
    if not isinstance(allowed_edit, dict) or not isinstance(target, dict):
        raise ValueError("centroid distance move metadata requires allowed_edit and target")
    direction_vector = allowed_edit["direction_vector"]
    if not isinstance(direction_vector, list | tuple) or len(direction_vector) != 3:
        raise ValueError("centroid distance move metadata requires direction_vector")
    dx, dy, dz = (float(direction_vector[0]), float(direction_vector[1]), float(direction_vector[2]))
    target_distance = float(target["target_centroid_distance_mm"])
    return f"""The two CadQuery objects are currently non-intersecting.

Object "object_a" is fixed. Object "object_b" may be translated only along the current centroid-to-centroid direction:
({dx:.6f}, {dy:.6f}, {dz:.6f})

Goal:
Move "object_b" so that the centroid-to-centroid distance between the objects is {target_distance:.1f} mm while preserving non-intersection.

Definitions:
- Positive distance moves object_b farther from object_a along the centroid-to-centroid direction.
- Negative distance moves object_b closer to object_a along the centroid-to-centroid direction.
- Centroid distance is the exact distance between object centroids after assembly transforms.
- Use millimetres.
- Do not execute code.

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Final answer format:
distance_mm=<signed number with one decimal>"""


def make_candidate_selection_prompt(record: GeometryRecord, metadata: dict[str, object]) -> str:
    candidates = metadata["candidate_edits"]
    if not isinstance(candidates, dict):
        raise ValueError("candidate selection metadata requires candidate_edits")
    candidate_lines = "\n".join(_candidate_prompt_line(label, candidates[label]) for label in _CANDIDATE_LABELS)
    return f"""You are given two CadQuery object-construction functions and assembly transforms.

Goal:
Choose the edit that satisfies the target condition with the smallest movement.

Target condition:
The objects must not intersect and must have at least {TARGET_CLEARANCE_MM:.1f} mm clearance.

Candidate edits:
{candidate_lines}

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Final answer format:
Return exactly one letter: A, B, C, or D."""


def make_candidate_ranking_prompt(record: GeometryRecord, metadata: dict[str, object]) -> str:
    candidates = metadata["candidate_edits"]
    if not isinstance(candidates, dict):
        raise ValueError("candidate ranking metadata requires candidate_edits")
    candidate_lines = "\n".join(_candidate_prompt_line(label, candidates[label]) for label in _CANDIDATE_LABELS)
    return f"""Rank the candidate edits from best to worst.

A better edit:
1. satisfies the target clearance,
2. avoids positive-volume interference,
3. uses a smaller translation magnitude.

Target clearance: {TARGET_CLEARANCE_MM:.1f} mm

Candidate edits:
{candidate_lines}

Object code:
```python
{object_code_from_script(record.assembly_script).strip()}
```

Transforms:
{transforms_text(record)}

Final answer format:
Return exactly four capital letters, such as DBCA."""


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


def materialize_axis_aligned_repair_row(
    record: GeometryRecord,
    row_number: int,
    split: str,
) -> PublicTaskRow | None:
    if record.labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}:
        return None
    metadata = exact_repair_metadata(record, target_clearance_mm=0.0)
    prompt = make_axis_aligned_repair_prompt(record)
    return public_row(
        record=record,
        task_type=TaskType.AXIS_ALIGNED_REPAIR,
        answer=_axis_answer_from_metadata(metadata),
        prompt=prompt,
        row_number=row_number,
        split=Split(split),
        template_version=AXIS_ALIGNED_REPAIR_TEMPLATE_VERSION,
        extras=metadata,
    )


def materialize_axis_aligned_repair_vector_row(
    record: GeometryRecord,
    row_number: int,
    split: str,
) -> PublicTaskRow | None:
    if record.labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}:
        return None
    metadata = exact_repair_metadata(record, target_clearance_mm=0.0)
    metadata = {
        **metadata,
        "output_format": "translation_vector",
        "edit_family": "axis_aligned_intersection_repair_vector",
        **_edit_counterfactual_metadata(
            record,
            edit_family="axis_aligned_intersection_repair_vector",
            target_descriptor="non_intersection_vector",
        ),
    }
    return public_row(
        record=record,
        task_type=TaskType.AXIS_ALIGNED_REPAIR_VECTOR,
        answer=_vector_answer_from_metadata(metadata),
        prompt=make_axis_aligned_repair_vector_prompt(record),
        row_number=row_number,
        split=Split(split),
        template_version=AXIS_ALIGNED_REPAIR_VECTOR_TEMPLATE_VERSION,
        extras=metadata,
    )


def materialize_axis_aligned_repair_program_row(
    record: GeometryRecord,
    row_number: int,
    split: str,
) -> PublicTaskRow | None:
    if record.labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}:
        return None
    metadata = exact_repair_metadata(record, target_clearance_mm=0.0)
    metadata = {
        **metadata,
        "output_format": "edit_program",
        "edit_family": "axis_aligned_intersection_repair_program",
        "edit_program_language": "cadquery_python",
        **_edit_counterfactual_metadata(
            record,
            edit_family="axis_aligned_intersection_repair_program",
            target_descriptor="non_intersection_program",
        ),
    }
    return public_row(
        record=record,
        task_type=TaskType.AXIS_ALIGNED_REPAIR_PROGRAM,
        answer=_edit_program_answer_from_metadata(metadata),
        prompt=make_axis_aligned_repair_program_prompt(record),
        row_number=row_number,
        split=Split(split),
        template_version=AXIS_ALIGNED_REPAIR_PROGRAM_TEMPLATE_VERSION,
        extras=metadata,
    )


def materialize_target_clearance_repair_row(
    record: GeometryRecord,
    row_number: int,
    split: str,
) -> PublicTaskRow | None:
    if record.labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}:
        return None
    metadata = exact_repair_metadata(record, target_clearance_mm=TARGET_CLEARANCE_MM)
    prompt = make_target_clearance_repair_prompt(record)
    return public_row(
        record=record,
        task_type=TaskType.TARGET_CLEARANCE_REPAIR,
        answer=_axis_answer_from_metadata(metadata),
        prompt=prompt,
        row_number=row_number,
        split=Split(split),
        template_version=TARGET_CLEARANCE_REPAIR_TEMPLATE_VERSION,
        extras=metadata,
    )


def materialize_target_clearance_move_row(
    record: GeometryRecord,
    row_number: int,
    split: str,
) -> PublicTaskRow | None:
    metadata = target_clearance_move_metadata(record)
    if metadata is None:
        return None
    return public_row(
        record=record,
        task_type=TaskType.TARGET_CLEARANCE_MOVE,
        answer=_signed_distance_answer_from_metadata(metadata),
        prompt=make_target_clearance_move_prompt(record, metadata),
        row_number=row_number,
        split=Split(split),
        template_version=TARGET_CLEARANCE_MOVE_TEMPLATE_VERSION,
        extras=metadata,
    )


def materialize_target_contact_move_row(
    record: GeometryRecord,
    row_number: int,
    split: str,
) -> PublicTaskRow | None:
    metadata = target_contact_move_metadata(record)
    if metadata is None:
        return None
    return public_row(
        record=record,
        task_type=TaskType.TARGET_CONTACT_MOVE,
        answer=_signed_distance_answer_from_metadata(metadata),
        prompt=make_target_contact_move_prompt(record, metadata),
        row_number=row_number,
        split=Split(split),
        template_version=TARGET_CONTACT_MOVE_TEMPLATE_VERSION,
        extras=metadata,
    )


def materialize_centroid_distance_move_row(
    record: GeometryRecord,
    row_number: int,
    split: str,
) -> PublicTaskRow | None:
    metadata = centroid_distance_move_metadata(record)
    if metadata is None:
        return None
    return public_row(
        record=record,
        task_type=TaskType.CENTROID_DISTANCE_MOVE,
        answer=_signed_distance_answer_from_metadata(metadata),
        prompt=make_centroid_distance_move_prompt(record, metadata),
        row_number=row_number,
        split=Split(split),
        template_version=CENTROID_DISTANCE_MOVE_TEMPLATE_VERSION,
        extras=metadata,
    )


def materialize_candidate_selection_row(
    record: GeometryRecord,
    row_number: int,
    split: str,
) -> PublicTaskRow | None:
    if record.labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}:
        return None
    metadata = candidate_edit_metadata(record)
    return public_row(
        record=record,
        task_type=TaskType.EDIT_CANDIDATE_SELECTION,
        answer=str(metadata["candidate_selection_answer"]),
        prompt=make_candidate_selection_prompt(record, metadata),
        row_number=row_number,
        split=Split(split),
        template_version=EDIT_CANDIDATE_SELECTION_TEMPLATE_VERSION,
        extras=metadata,
    )


def materialize_candidate_ranking_row(
    record: GeometryRecord,
    row_number: int,
    split: str,
) -> PublicTaskRow | None:
    if record.labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}:
        return None
    metadata = candidate_edit_metadata(record)
    return public_row(
        record=record,
        task_type=TaskType.EDIT_CANDIDATE_RANKING,
        answer=str(metadata["candidate_ranking_answer"]),
        prompt=make_candidate_ranking_prompt(record, metadata),
        row_number=row_number,
        split=Split(split),
        template_version=EDIT_CANDIDATE_RANKING_TEMPLATE_VERSION,
        extras=metadata,
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


def _placed_shapes(record: GeometryRecord) -> tuple[Any, Any]:
    import cadquery as cq

    namespace: dict[str, Any] = {"cq": cq, "cadquery": cq, "__builtins__": __builtins__}
    exec(compile(record.assembly_script, "<intersectionedit-materialize>", "exec"), namespace)
    assembly = namespace.get("assembly")
    if not callable(assembly):
        raise ValueError("assembly script does not define assembly()")
    placed_a, placed_b = assembly()
    return object_to_shape(placed_a), object_to_shape(placed_b)


def _minimum_exact_magnitude(
    record: GeometryRecord,
    shape_a: Any,
    shape_b: Any,
    direction: str,
    target_clearance_mm: float,
) -> float:
    if _satisfies_target(record, shape_a, shape_b, direction, 0.0, target_clearance_mm)[0]:
        return 0.0
    hi = max(
        _aabb_candidate_magnitude(record, direction) + target_clearance_mm + record.label_policy.epsilon_distance_mm,
        target_clearance_mm + 1.0,
        1.0,
    )
    for _ in range(40):
        if _satisfies_target(record, shape_a, shape_b, direction, hi, target_clearance_mm)[0]:
            break
        hi *= 2.0
    else:
        raise ValueError(f"could not bracket repair distance for {direction}")

    lo = 0.0
    for _ in range(22):
        mid = (lo + hi) / 2.0
        if _satisfies_target(record, shape_a, shape_b, direction, mid, target_clearance_mm)[0]:
            hi = mid
        else:
            lo = mid
    return hi


def _exact_move_from_magnitude(
    record: GeometryRecord,
    shape_a: Any,
    shape_b: Any,
    direction: str,
    exact_magnitude: float,
    label_magnitude: float,
    target_clearance_mm: float,
) -> ExactRepairMove:
    satisfies, labels = _satisfies_target(
        record,
        shape_a,
        shape_b,
        direction,
        label_magnitude,
        target_clearance_mm,
    )
    return ExactRepairMove(
        direction=direction,
        exact_magnitude_mm=exact_magnitude,
        label_magnitude_mm=label_magnitude,
        translation_vector_mm=_translation_vector(direction, label_magnitude),
        final_relation=str(labels.relation),
        final_clearance_mm=labels.minimum_distance,
        final_intersection_volume=labels.intersection_volume,
        satisfies_target=satisfies,
    )


def _satisfies_target(
    record: GeometryRecord,
    shape_a: Any,
    shape_b: Any,
    direction: str,
    magnitude_mm: float,
    target_clearance_mm: float,
) -> tuple[bool, Any]:
    moved_b = apply_transform(
        shape_b,
        Transform(
            translation=_translation_vector(direction, magnitude_mm),
            rotation_xyz_deg=(0.0, 0.0, 0.0),
        ),
    )
    raw = measure_shape_pair(
        shape_a,
        moved_b,
        IDENTITY_TRANSFORM,
        IDENTITY_TRANSFORM,
        record.label_policy,
    )
    labels, _ = derive_labels(raw, record.label_policy)
    non_intersecting = labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}
    if target_clearance_mm <= 0.0:
        return non_intersecting, labels
    clearance = labels.minimum_distance
    return (
        non_intersecting
        and clearance is not None
        and clearance + record.label_policy.epsilon_distance_mm >= target_clearance_mm,
        labels,
    )


def _translation_vector(direction: str, magnitude_mm: float) -> tuple[float, float, float]:
    unit = _DIRECTION_VECTORS[direction]
    return tuple(component * magnitude_mm for component in unit)


def _aabb_candidate_magnitude(record: GeometryRecord, direction: str) -> float:
    return next(candidate.magnitude_mm for candidate in repair_candidates(record) if candidate.direction == direction)


def _label_magnitude(exact_magnitude: float) -> float:
    if exact_magnitude <= 0.0:
        return 0.0
    scale = 1.0 / NUMERIC_LABEL_DECIMAL_MM
    return math.ceil((exact_magnitude - 1e-9) * scale) / scale


def _round_one_decimal(value: float) -> float:
    return round(value + 1e-12, 1)


def _shape_center(shape: Any) -> tuple[float, float, float]:
    center = shape.Center()
    values = center.toTuple() if hasattr(center, "toTuple") else center
    return tuple(float(values[index]) for index in range(3))


def _unit_vector(vector: tuple[float, float, float]) -> tuple[float, float, float] | None:
    magnitude = _distance((0.0, 0.0, 0.0), vector)
    if magnitude <= 1e-9 or not math.isfinite(magnitude):
        return None
    return tuple(component / magnitude for component in vector)


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _initial_state_metadata(record: GeometryRecord) -> dict[str, object]:
    return {
        "relation": str(record.labels.relation),
        "intersection_volume": record.labels.intersection_volume,
        "normalized_intersection": record.labels.normalized_intersection,
        "minimum_distance": record.labels.minimum_distance,
    }


def _edit_split_group(record: GeometryRecord, suffix: str) -> str:
    base = (
        record.counterfactual_group_id
        or record.assembly_group_id
        or record.base_object_pair_id
        or record.geometry_id
    )
    return f"{base}:{suffix}"


def _edit_counterfactual_metadata(
    record: GeometryRecord,
    *,
    edit_family: str,
    target_descriptor: str,
) -> dict[str, object]:
    base = (
        record.counterfactual_group_id
        or record.assembly_group_id
        or record.base_object_pair_id
        or record.geometry_id
    )
    dimensions = ["edit_family", "target"]
    if record.changed_parameter:
        dimensions.append("initial_translation" if record.changed_parameter.startswith("transform_") else "source_variant")
    return {
        "edit_counterfactual_group_id": f"{base}:intersectionedit_v01",
        "edit_counterfactual_dimensions": dimensions,
        "edit_counterfactual_variant": {
            "edit_family": edit_family,
            "target_descriptor": target_descriptor,
            "source_counterfactual_group_id": record.counterfactual_group_id,
            "source_variant_id": record.variant_id,
            "source_changed_parameter": record.changed_parameter,
            "source_changed_value": record.changed_value,
        },
    }


def _signed_distance_answer_from_metadata(metadata: dict[str, object]) -> str:
    structured = metadata["structured_answer"]
    if not isinstance(structured, dict):
        raise ValueError("target clearance move metadata requires structured_answer")
    return f"distance_mm={float(structured['distance_mm']):.1f}"


def _vector_answer_from_metadata(metadata: dict[str, object]) -> str:
    vector = metadata["selected_translation_vector_mm"]
    if not isinstance(vector, list | tuple) or len(vector) != 3:
        raise ValueError("vector repair metadata requires selected_translation_vector_mm")
    return f"dx={float(vector[0]):.1f}, dy={float(vector[1]):.1f}, dz={float(vector[2]):.1f}"


def _edit_program_answer_from_metadata(metadata: dict[str, object]) -> str:
    vector = metadata["selected_translation_vector_mm"]
    if not isinstance(vector, list | tuple) or len(vector) != 3:
        raise ValueError("program repair metadata requires selected_translation_vector_mm")
    return (
        "object_b = object_b.translate("
        f"({float(vector[0]):.1f}, {float(vector[1]):.1f}, {float(vector[2]):.1f})"
        ")"
    )


def _axis_answer_from_metadata(metadata: dict[str, object]) -> str:
    structured = metadata["structured_answer"]
    if not isinstance(structured, dict):
        raise ValueError("exact repair metadata requires structured_answer")
    return f"direction={structured['direction']}, distance_mm={float(structured['distance_mm']):.1f}"


def _candidate_edit_item(
    record: GeometryRecord,
    shape_a: Any,
    shape_b: Any,
    label: str,
    strategy: str,
    direction: str,
    magnitude: float,
    target_clearance_mm: float,
) -> dict[str, object]:
    satisfies, labels = _satisfies_target(record, shape_a, shape_b, direction, magnitude, target_clearance_mm)
    non_intersecting = labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}
    clearance = labels.minimum_distance
    return {
        "label": label,
        "strategy": strategy,
        "direction": direction,
        "magnitude_mm": magnitude,
        "translation_vector_mm": list(_translation_vector(direction, magnitude)),
        "movement_magnitude": magnitude,
        "final_relation": str(labels.relation),
        "final_clearance_mm": clearance,
        "final_intersection_volume": labels.intersection_volume,
        "satisfies_target": satisfies,
        "no_interference": non_intersecting,
        "clearance_error_mm": (
            abs(clearance - target_clearance_mm) if clearance is not None else None
        ),
    }


def _candidate_rank_key(label: str, candidate: dict[str, object]) -> tuple[int, int, float, float, str]:
    clearance_error = candidate.get("clearance_error_mm")
    return (
        0 if candidate.get("satisfies_target") else 1,
        0 if candidate.get("no_interference") else 1,
        float(candidate["movement_magnitude"]),
        float(clearance_error) if isinstance(clearance_error, int | float) else math.inf,
        label,
    )


def _candidate_prompt_line(label: str, candidate: object) -> str:
    if not isinstance(candidate, dict):
        raise ValueError("candidate prompt line requires candidate metadata")
    vector = candidate["translation_vector_mm"]
    if not isinstance(vector, list):
        raise ValueError("candidate prompt line requires translation_vector_mm")
    dx, dy, dz = (float(vector[0]), float(vector[1]), float(vector[2]))
    return f'{label}. translate "object_b" by ({dx:.1f}, {dy:.1f}, {dz:.1f})'


def _single_aabb_away_direction(record: GeometryRecord) -> str | None:
    bbox_a = _bbox_from_metadata(record, "bbox_a")
    bbox_b = _bbox_from_metadata(record, "bbox_b")
    separated: list[str] = []
    for axis_index, axis_name in enumerate(_AXES):
        if bbox_b.min[axis_index] > bbox_a.max[axis_index]:
            separated.append(f"+{axis_name}")
        elif bbox_b.max[axis_index] < bbox_a.min[axis_index]:
            separated.append(f"-{axis_name}")
    if len(separated) != 1:
        return None
    return separated[0]
