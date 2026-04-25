"""IntersectionEdit difficulty taxonomy helpers."""

from __future__ import annotations

from intersectionqa.enums import TaskType
from intersectionqa.schema import PublicTaskRow

EDIT_TASK_TYPES = {
    TaskType.REPAIR_DIRECTION,
    TaskType.REPAIR_TRANSLATION,
    TaskType.AXIS_ALIGNED_REPAIR,
    TaskType.AXIS_ALIGNED_REPAIR_VECTOR,
    TaskType.AXIS_ALIGNED_REPAIR_PROGRAM,
    TaskType.TARGET_CLEARANCE_REPAIR,
    TaskType.TARGET_CLEARANCE_MOVE,
    TaskType.TARGET_CONTACT_MOVE,
    TaskType.CENTROID_DISTANCE_MOVE,
    TaskType.EDIT_CANDIDATE_SELECTION,
    TaskType.EDIT_CANDIDATE_RANKING,
}

EDIT_DIFFICULTY_TAGS = {
    "axis_aligned_intersection_repair",
    "axis_aligned_intersection_repair_vector",
    "axis_aligned_intersection_repair_program",
    "conservative_axis_aligned_repair",
    "axis_aligned_target_clearance_repair",
    "axis_aligned_target_clearance_move",
    "axis_aligned_target_contact_move",
    "centroid_distance_move",
    "target_clearance_candidate_set",
    "candidate_selection",
    "candidate_ranking",
    "output_direction",
    "output_axis_distance",
    "output_signed_distance",
    "output_vector",
    "output_edit_program",
    "output_candidate_selection",
    "output_candidate_ranking",
    "ambiguous_direction",
    "move_closer",
    "move_farther",
    "move_none",
    "initial_axis_aligned",
    "initial_rotated",
    "initial_cavity_or_concavity",
    "initial_near_boundary",
    "initial_aabb_exact_disagreement",
    "counterfactual_edit_group",
    "multi_object_planned",
}


def edit_difficulty_tags(row: PublicTaskRow) -> list[str]:
    if row.task_type not in EDIT_TASK_TYPES:
        return []
    tags: set[str] = set()
    metadata = row.metadata
    diagnostics = metadata.get("edit_diagnostics")
    if isinstance(diagnostics, dict):
        difficulty = diagnostics.get("difficulty")
        if isinstance(difficulty, str) and difficulty:
            tags.add(difficulty)
        move_kind = diagnostics.get("move_kind")
        if move_kind in {"closer", "farther", "none"}:
            tags.add(f"move_{move_kind}")
        if diagnostics.get("ambiguous") is True:
            tags.add("ambiguous_direction")
    edit_family = metadata.get("edit_family")
    if isinstance(edit_family, str) and edit_family in EDIT_DIFFICULTY_TAGS:
        tags.add(edit_family)

    tags.add(_output_tag(row.task_type))
    if metadata.get("edit_counterfactual_group_id"):
        tags.add("counterfactual_edit_group")
    if "axis_aligned" in row.difficulty_tags:
        tags.add("initial_axis_aligned")
    if "rotated" in row.difficulty_tags:
        tags.add("initial_rotated")
    if any(tag in row.difficulty_tags for tag in {"cavity_targeted", "compound_boolean"}):
        tags.add("initial_cavity_or_concavity")
    if "near_boundary" in row.difficulty_tags:
        tags.add("initial_near_boundary")
    if "aabb_exact_disagreement" in row.difficulty_tags:
        tags.add("initial_aabb_exact_disagreement")

    unknown = tags - EDIT_DIFFICULTY_TAGS
    if unknown:
        tags.update(f"unknown:{tag}" for tag in unknown)
        tags -= unknown
    return sorted(tags)


def _output_tag(task_type: TaskType) -> str:
    if task_type == TaskType.REPAIR_DIRECTION:
        return "output_direction"
    if task_type in {TaskType.REPAIR_TRANSLATION, TaskType.AXIS_ALIGNED_REPAIR, TaskType.TARGET_CLEARANCE_REPAIR}:
        return "output_axis_distance"
    if task_type in {TaskType.TARGET_CLEARANCE_MOVE, TaskType.TARGET_CONTACT_MOVE, TaskType.CENTROID_DISTANCE_MOVE}:
        return "output_signed_distance"
    if task_type == TaskType.AXIS_ALIGNED_REPAIR_VECTOR:
        return "output_vector"
    if task_type == TaskType.AXIS_ALIGNED_REPAIR_PROGRAM:
        return "output_edit_program"
    if task_type == TaskType.EDIT_CANDIDATE_SELECTION:
        return "output_candidate_selection"
    return "output_candidate_ranking"
