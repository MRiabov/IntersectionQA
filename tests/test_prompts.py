import pytest

from intersectionqa.config import DatasetConfig
from intersectionqa.enums import Relation, TaskType
from intersectionqa.geometry.cadquery_exec import measure_source_pair
from intersectionqa.geometry.labels import derive_labels
from intersectionqa.evaluation.parsing import parse_answer
from intersectionqa.prompts.binary import ALLOWED_ANSWERS as BINARY_ALLOWED
from intersectionqa.prompts.binary import make_binary_prompt
from intersectionqa.prompts.buckets import ALLOWED_ANSWERS as BUCKET_ALLOWED
from intersectionqa.prompts.buckets import clearance_bucket
from intersectionqa.prompts.counterfactual import balanced_pairwise_records, pairwise_answer, pairwise_records
from intersectionqa.prompts.fit import tolerance_fit_answer
from intersectionqa.prompts.ranking import ranking_answer, ranking_records
from intersectionqa.prompts.common import strict_parse
from intersectionqa.prompts.materialize import materialize_rows
from intersectionqa.prompts.repair import ALLOWED_ANSWERS as REPAIR_ALLOWED
from intersectionqa.prompts.repair import make_repair_direction_prompt, make_repair_translation_prompt
from intersectionqa.prompts.repair import repair_candidates, repair_metadata, repair_plan
from intersectionqa.prompts.relation import ALLOWED_ANSWERS as RELATION_ALLOWED
from intersectionqa.schema import Transform
from intersectionqa.sources.synthetic import fixture_geometry_records, synthetic_source_records
from intersectionqa.splits.grouped import assign_geometry_splits


def test_strict_parsers_reject_prose_and_case_changes():
    assert strict_parse("yes", BINARY_ALLOWED) == "yes"
    assert strict_parse(" yes\n", BINARY_ALLOWED) == "yes"
    assert strict_parse("Yes", BINARY_ALLOWED) is None
    assert strict_parse("yes.", BINARY_ALLOWED) is None
    assert strict_parse("the answer is yes", BINARY_ALLOWED) is None
    assert strict_parse("touching", RELATION_ALLOWED) == "touching"
    assert strict_parse("(0, 0.01]", BUCKET_ALLOWED) == "(0, 0.01]"
    assert parse_answer(TaskType.CLEARANCE_BUCKET, "(0.1, 1]") == "(0.1, 1]"
    assert parse_answer(TaskType.PAIRWISE_INTERFERENCE, "both") == "both"
    assert parse_answer(TaskType.RANKING_NORMALIZED_INTERSECTION, "CBA") == "CBA"
    assert parse_answer(TaskType.TOLERANCE_FIT, "no") == "no"
    assert strict_parse("+x", REPAIR_ALLOWED) == "+x"
    assert parse_answer(TaskType.REPAIR_DIRECTION, "-z") == "-z"
    assert parse_answer(TaskType.REPAIR_DIRECTION, "no_valid_move") is None
    assert parse_answer(TaskType.REPAIR_TRANSLATION, "+x 1.250000") == "+x 1.250000"
    assert parse_answer(TaskType.REPAIR_TRANSLATION, "+x 1.25") is None
    assert parse_answer(TaskType.REPAIR_TRANSLATION, "+x -1.250000") is None


def test_prompts_do_not_leak_stored_labels_or_diagnostics():
    config = DatasetConfig()
    record = fixture_geometry_records(config.label_policy, config.config_hash)[2]
    prompt = make_binary_prompt(record)
    assert str(record.labels.intersection_volume) not in prompt
    assert record.labels.relation not in prompt
    assert "aabb_overlap" not in prompt
    assert "label_status" not in prompt

    repair_prompt = make_repair_direction_prompt(record)
    assert str(record.labels.intersection_volume) not in repair_prompt
    assert record.labels.relation not in repair_prompt
    assert "aabb_overlap" not in repair_prompt
    assert "selected_direction" not in repair_prompt

    repair_translation_prompt = make_repair_translation_prompt(record)
    assert str(record.labels.intersection_volume) not in repair_translation_prompt
    assert record.labels.relation not in repair_translation_prompt
    assert "selected_magnitude_mm" not in repair_translation_prompt


def test_materialized_rows_validate_answers():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    splits = assign_geometry_splits(records, config.seed)
    rows = materialize_rows(records, splits, config.smoke.task_types)
    assert len(rows) == len(records) * 3
    assert all(row.hashes.prompt_hash for row in rows)


def test_clearance_and_tolerance_prompt_answers_use_exact_distance():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)

    assert clearance_bucket(records[0]) == ">5"
    assert clearance_bucket(records[1]) == "touching"
    assert clearance_bucket(records[2]) == "intersecting"
    assert clearance_bucket(records[3]) == "(0.1, 1]"
    assert tolerance_fit_answer(records[0]) == "yes"
    assert tolerance_fit_answer(records[1]) == "no"
    assert tolerance_fit_answer(records[3]) == "no"


def test_counterfactual_pairwise_and_ranking_rows_materialize_from_group():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    splits = assign_geometry_splits(records, config.seed)
    task_types = [
        TaskType.BINARY_INTERFERENCE,
        TaskType.RELATION_CLASSIFICATION,
        TaskType.VOLUME_BUCKET,
        TaskType.CLEARANCE_BUCKET,
        TaskType.TOLERANCE_FIT,
        TaskType.PAIRWISE_INTERFERENCE,
        TaskType.RANKING_NORMALIZED_INTERSECTION,
    ]

    rows = materialize_rows(records, splits, task_types)
    by_task = {task_type: [row for row in rows if row.task_type == task_type] for task_type in task_types}

    assert len(rows) == len(records) * 5 + 5
    assert len(by_task[TaskType.PAIRWISE_INTERFERENCE]) == 4
    assert len(by_task[TaskType.RANKING_NORMALIZED_INTERSECTION]) == 1

    group = [record for record in records if record.counterfactual_group_id == "cfg_000001"]
    pair = pairwise_records(group)
    assert pair is not None
    assert pairwise_answer(*pair) == "B"
    balanced_pairs = balanced_pairwise_records(group)
    pairwise_rows = by_task[TaskType.PAIRWISE_INTERFERENCE]
    assert [row.answer for row in pairwise_rows] == [pairwise_answer(*pair) for pair in balanced_pairs]
    assert {row.answer for row in pairwise_rows} == {"A", "B", "both", "neither"}
    assert all(row.counterfactual_group_id == "cfg_000001" for row in pairwise_rows)
    assert all(row.metadata["source_variant_ids"] for row in pairwise_rows)

    ranked = ranking_records(group)
    assert len(ranked) == 5
    assert by_task[TaskType.RANKING_NORMALIZED_INTERSECTION][0].answer == ranking_answer(ranked)
    assert by_task[TaskType.RANKING_NORMALIZED_INTERSECTION][0].geometry_ids == [
        record.geometry_id for record in ranked
    ]


def test_repair_direction_rows_use_positive_overlap_records_only():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    splits = assign_geometry_splits(records, config.seed)

    rows = materialize_rows(records, splits, [TaskType.REPAIR_DIRECTION])
    positive_overlap_records = [
        record
        for record in records
        if record.labels.relation in {Relation.INTERSECTING, Relation.CONTAINED}
    ]

    assert len(rows) == len(positive_overlap_records)
    assert {row.geometry_ids[0] for row in rows} == {
        record.geometry_id for record in positive_overlap_records
    }
    assert all(row.task_type == TaskType.REPAIR_DIRECTION for row in rows)
    assert all(row.answer in REPAIR_ALLOWED for row in rows)
    assert all(row.answer == row.metadata["selected_direction"] for row in rows)
    assert all(
        row.metadata["repair_policy"] == "conservative_aabb_separating_translation_v01"
        for row in rows
    )
    assert all(len(row.metadata["candidate_moves"]) == 6 for row in rows)


def test_repair_translation_rows_use_canonical_six_decimal_answer():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    splits = assign_geometry_splits(records, config.seed)

    rows = materialize_rows(records, splits, [TaskType.REPAIR_TRANSLATION])

    assert rows
    assert all(row.task_type == TaskType.REPAIR_TRANSLATION for row in rows)
    for row in rows:
        direction, magnitude = row.answer.split(" ")
        assert direction == row.metadata["selected_direction"]
        assert magnitude == f"{float(row.metadata['selected_magnitude_mm']):.6f}"
        assert parse_answer(TaskType.REPAIR_TRANSLATION, row.answer) == row.answer


def test_repair_direction_policy_chooses_smallest_aabb_separating_move():
    config = DatasetConfig()
    base = fixture_geometry_records(config.label_policy, config.config_hash)[2]
    record = _record_with_bboxes(
        base,
        bbox_a={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        bbox_b={"min": [0.5, -0.2, -0.2], "max": [2.5, 0.2, 0.2]},
    )

    candidates = {move.direction: move for move in repair_candidates(record)}
    plan = repair_plan(record)
    metadata = repair_metadata(record)

    assert plan.direction == "+x"
    assert plan.magnitude_mm == pytest.approx(0.5001)
    assert plan.translation_vector_mm == pytest.approx((0.5001, 0.0, 0.0))
    assert candidates["+x"].translation_vector_mm == pytest.approx((0.5001, 0.0, 0.0))
    assert candidates["-x"].translation_vector_mm == pytest.approx((-3.5001, 0.0, 0.0))
    assert metadata["selected_direction"] == "+x"
    assert metadata["selected_translation_vector_mm"] == pytest.approx([0.5001, 0.0, 0.0])
    assert _aabb_disjoint_after_move(record.metadata["bbox_a"], record.metadata["bbox_b"], plan.translation_vector_mm)


def test_repair_direction_policy_uses_documented_tie_break_order():
    config = DatasetConfig()
    base = fixture_geometry_records(config.label_policy, config.config_hash)[2]
    record = _record_with_bboxes(
        base,
        bbox_a={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        bbox_b={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
    )

    plan = repair_plan(record)
    metadata = repair_metadata(record)

    assert plan.direction == "+x"
    assert plan.magnitude_mm == pytest.approx(2.0001)
    assert metadata["candidate_direction_labels"] == ["+x", "-x", "+y", "-y", "+z", "-z"]


def test_repair_direction_move_removes_exact_synthetic_overlap():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    splits = assign_geometry_splits(records, config.seed)
    rows = materialize_rows(records, splits, [TaskType.REPAIR_DIRECTION])
    source_by_id = {record.object_id: record for record in synthetic_source_records()}
    geometry_by_id = {record.geometry_id: record for record in records}

    for row in rows:
        record = geometry_by_id[row.geometry_ids[0]]
        vector = row.metadata["selected_translation_vector_mm"]
        moved_transform_b = Transform(
            translation=tuple(
                record.transform_b.translation[index] + vector[index] for index in range(3)
            ),
            rotation_xyz_deg=record.transform_b.rotation_xyz_deg,
        )
        raw_geometry = measure_source_pair(
            source_by_id[record.object_a_id],
            source_by_id[record.object_b_id],
            record.transform_a,
            moved_transform_b,
            record.label_policy,
        )
        labels, _ = derive_labels(raw_geometry, record.label_policy)

        assert labels.relation not in {Relation.INTERSECTING, Relation.CONTAINED}


def _record_with_bboxes(record, *, bbox_a: dict[str, list[float]], bbox_b: dict[str, list[float]]):
    return record.model_copy(
        update={
            "metadata": {
                **record.metadata,
                "bbox_a": bbox_a,
                "bbox_b": bbox_b,
            }
        }
    )


def _aabb_disjoint_after_move(
    bbox_a: dict[str, list[float]],
    bbox_b: dict[str, list[float]],
    vector: tuple[float, float, float],
) -> bool:
    moved_b = {
        "min": [bbox_b["min"][index] + vector[index] for index in range(3)],
        "max": [bbox_b["max"][index] + vector[index] for index in range(3)],
    }
    return any(
        moved_b["min"][index] > bbox_a["max"][index]
        or moved_b["max"][index] < bbox_a["min"][index]
        for index in range(3)
    )
