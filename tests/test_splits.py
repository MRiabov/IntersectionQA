from intersectionqa.config import DatasetConfig
from intersectionqa.prompts.materialize import materialize_rows
from intersectionqa.sources.synthetic import fixture_geometry_records
from intersectionqa.splits.grouped import assign_geometry_splits, audit_group_leakage


def test_group_split_is_deterministic_and_leak_free():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    first = assign_geometry_splits(records, config.seed)
    second = assign_geometry_splits(records, config.seed)
    assert first == second

    rows = materialize_rows(records, first, config.smoke.task_types)
    audit = audit_group_leakage(rows)
    assert audit.status == "pass"
    assert audit.violation_count == 0


def test_near_boundary_records_go_to_hard_split():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    splits = assign_geometry_splits(records, config.seed)
    for record in records:
        if "near_boundary" in record.difficulty_tags:
            assert splits[record.geometry_id] == "test_near_boundary"
