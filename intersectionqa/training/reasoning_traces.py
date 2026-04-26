"""Internal reasoning-trace targets for supervised GRPO bootstrapping."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

from intersectionqa.enums import TaskType
from intersectionqa.schema import PublicTaskRow


def reasoning_target_text(row: PublicTaskRow) -> str:
    """Return a short tagged reasoning completion while preserving the canonical answer."""
    return f"<think>{reasoning_sentence(row)}</think><answer>{row.answer}</answer>"


def reasoning_sentence(row: PublicTaskRow) -> str:
    metadata = row.metadata
    task_type = row.task_type
    if task_type == TaskType.REPAIR_DIRECTION:
        return _repair_direction_sentence(metadata)
    if task_type == TaskType.REPAIR_TRANSLATION:
        return _repair_translation_sentence(metadata)
    if task_type in {
        TaskType.AXIS_ALIGNED_REPAIR,
        TaskType.TARGET_CLEARANCE_REPAIR,
        TaskType.AXIS_ALIGNED_REPAIR_VECTOR,
        TaskType.AXIS_ALIGNED_REPAIR_PROGRAM,
    }:
        return _exact_repair_sentence(metadata)
    if task_type in {
        TaskType.TARGET_CLEARANCE_MOVE,
        TaskType.TARGET_CONTACT_MOVE,
        TaskType.CENTROID_DISTANCE_MOVE,
    }:
        return _signed_distance_sentence(row)
    if task_type == TaskType.EDIT_CANDIDATE_SELECTION:
        return "Compare the candidate edits by target satisfaction first, then non-intersection, then smaller movement."
    if task_type == TaskType.EDIT_CANDIDATE_RANKING:
        return "Rank the candidate edits by target satisfaction, non-intersection, and then movement magnitude."
    if task_type == TaskType.BINARY_INTERFERENCE:
        return "Classify positive-volume overlap or containment as yes, otherwise no."
    if task_type == TaskType.RELATION_CLASSIFICATION:
        return "Select the canonical spatial relation label from the exact geometry state."
    if task_type == TaskType.TOLERANCE_FIT:
        return "Compare the exact clearance with the required clearance threshold."
    if task_type == TaskType.CLEARANCE_BUCKET:
        return "Place the exact clearance into the requested clearance bucket."
    if task_type == TaskType.VOLUME_BUCKET:
        return "Place the normalized intersection volume into the requested volume bucket."
    if task_type == TaskType.PAIRWISE_INTERFERENCE:
        return "Compare both variants and select the one with positive-volume interference."
    if task_type == TaskType.RANKING_NORMALIZED_INTERSECTION:
        return "Rank variants by normalized intersection volume from largest to smallest."
    return "Apply the task definition and emit the canonical answer."


def add_reasoning_target(row: PublicTaskRow) -> dict[str, Any]:
    payload = row.model_dump(mode="json")
    payload["target_text"] = reasoning_target_text(row)
    payload["canonical_answer"] = row.answer
    payload["supervision"] = {
        "target_text_format": "think_answer_v01",
        "target_text_source": "trusted_row_metadata",
    }
    return payload


def write_reasoning_sft_dataset(
    *,
    dataset_dir: Path,
    output_dir: Path,
    splits: Iterable[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "splits": {},
    }
    for split in splits:
        rows = _load_rows(dataset_dir / f"{split}.jsonl")
        output_path = output_dir / f"{split}.jsonl"
        task_counts = Counter(str(row.task_type) for row in rows)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(add_reasoning_target(row), sort_keys=True) + "\n")
        report["splits"][split] = {
            "input_path": str(dataset_dir / f"{split}.jsonl"),
            "output_path": str(output_path),
            "row_count": len(rows),
            "task_counts": dict(sorted(task_counts.items())),
        }
    (output_dir / "reasoning_sft_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _load_rows(path: Path) -> list[PublicTaskRow]:
    if not path.exists():
        raise FileNotFoundError(f"missing split file: {path}")
    rows: list[PublicTaskRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(PublicTaskRow.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid public row: {exc}") from exc
    return rows


def _repair_direction_sentence(metadata: dict[str, Any]) -> str:
    direction = metadata.get("selected_direction")
    magnitude = _finite_float(metadata.get("selected_magnitude_mm"))
    if direction is not None and magnitude is not None:
        return f"Compare the six conservative AABB separating moves and choose {direction}, the smallest move at {magnitude:.1f} mm."
    return "Compare the six conservative AABB separating moves and choose the direction with the smallest movement."


def _repair_translation_sentence(metadata: dict[str, Any]) -> str:
    direction = metadata.get("selected_direction")
    magnitude = _finite_float(metadata.get("selected_magnitude_mm"))
    if direction is not None and magnitude is not None:
        return f"The smallest conservative AABB repair moves object_b along {direction} for {magnitude:.6f} mm."
    return "Return the smallest conservative AABB repair direction and magnitude."


def _exact_repair_sentence(metadata: dict[str, Any]) -> str:
    answer = metadata.get("structured_answer")
    if isinstance(answer, dict) and "direction" in answer and "distance_mm" in answer:
        return f"The best exact axis-aligned edit is {answer['direction']} with {float(answer['distance_mm']):.1f} mm movement."
    return "Search the allowed axis-aligned edits and return the smallest verified repair."


def _signed_distance_sentence(row: PublicTaskRow) -> str:
    signed_distance = _finite_float(row.metadata.get("selected_signed_distance_mm"))
    allowed_edit = row.metadata.get("allowed_edit")
    direction = None
    if isinstance(allowed_edit, dict):
        direction = allowed_edit.get("direction") or allowed_edit.get("direction_vector")
    if signed_distance is None:
        return "Use the signed movement convention and return the target-satisfying distance."
    if row.task_type == TaskType.CENTROID_DISTANCE_MOVE:
        return f"Move along the centroid direction by the signed distance {signed_distance:.1f} mm to reach the target centroid distance."
    if direction is None:
        return f"Use the signed movement convention and move {signed_distance:.1f} mm to satisfy the target."
    return f"Along the allowed direction {direction}, the signed target-satisfying movement is {signed_distance:.1f} mm."


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result
