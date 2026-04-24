from intersectionqa.config import DatasetConfig
from intersectionqa.prompts.binary import ALLOWED_ANSWERS as BINARY_ALLOWED
from intersectionqa.prompts.binary import make_binary_prompt
from intersectionqa.prompts.buckets import ALLOWED_ANSWERS as BUCKET_ALLOWED
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
