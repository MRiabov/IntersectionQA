from intersectionqa.config import DatasetConfig
from intersectionqa.enums import TaskType
from intersectionqa.evaluation.parsing import parse_answer
from intersectionqa.prompts.binary import ALLOWED_ANSWERS as BINARY_ALLOWED
from intersectionqa.prompts.binary import make_binary_prompt
from intersectionqa.prompts.buckets import ALLOWED_ANSWERS as BUCKET_ALLOWED
from intersectionqa.prompts.buckets import clearance_bucket
from intersectionqa.prompts.counterfactual import pairwise_answer, pairwise_records
from intersectionqa.prompts.fit import tolerance_fit_answer
from intersectionqa.prompts.ranking import ranking_answer, ranking_records
from intersectionqa.prompts.common import strict_parse
from intersectionqa.prompts.materialize import materialize_rows
from intersectionqa.prompts.relation import ALLOWED_ANSWERS as RELATION_ALLOWED
from intersectionqa.sources.synthetic import fixture_geometry_records
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


def test_prompts_do_not_leak_stored_labels_or_diagnostics():
    config = DatasetConfig()
    record = fixture_geometry_records(config.label_policy, config.config_hash)[2]
    prompt = make_binary_prompt(record)
    assert str(record.labels.intersection_volume) not in prompt
    assert record.labels.relation not in prompt
    assert "aabb_overlap" not in prompt
    assert "label_status" not in prompt


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

    assert len(rows) == len(records) * 5 + 2
    assert len(by_task[TaskType.PAIRWISE_INTERFERENCE]) == 1
    assert len(by_task[TaskType.RANKING_NORMALIZED_INTERSECTION]) == 1

    group = [record for record in records if record.counterfactual_group_id == "cfg_000001"]
    pair = pairwise_records(group)
    assert pair is not None
    assert by_task[TaskType.PAIRWISE_INTERFERENCE][0].answer == pairwise_answer(*pair)
    assert by_task[TaskType.PAIRWISE_INTERFERENCE][0].counterfactual_group_id == "cfg_000001"
    assert by_task[TaskType.PAIRWISE_INTERFERENCE][0].metadata["source_variant_ids"]

    ranked = ranking_records(group)
    assert len(ranked) == 5
    assert by_task[TaskType.RANKING_NORMALIZED_INTERSECTION][0].answer == ranking_answer(ranked)
    assert by_task[TaskType.RANKING_NORMALIZED_INTERSECTION][0].geometry_ids == [
        record.geometry_id for record in ranked
    ]
