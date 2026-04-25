from intersectionqa.config import DatasetConfig
from intersectionqa.enums import TaskType
from intersectionqa.export.jsonl import write_split_files
from intersectionqa.export.parquet import write_parquet_files
from intersectionqa.prompts.materialize import materialize_rows
from scripts.prepare_intersectionedit_training_splits import prepare_intersectionedit_training_splits
from intersectionqa.sources.synthetic import fixture_geometry_records
from intersectionqa.splits.grouped import (
    INTERNAL_EVAL_SPLIT,
    INTERNAL_TRAIN_SPLIT,
    assign_geometry_splits,
    audit_group_leakage,
    partition_internal_train_eval_rows,
    split_manifest,
    training_split_group,
)


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


def test_near_boundary_records_use_bounded_group_split_policy():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    splits = assign_geometry_splits(records, config.seed)
    near_boundary_splits = {
        splits[record.geometry_id]
        for record in records
        if "near_boundary" in record.difficulty_tags
    }
    assert near_boundary_splits
    assert near_boundary_splits <= {
        "train",
        "validation",
        "test_random",
        "test_object_pair_heldout",
        "test_near_boundary",
    }

    by_assembly = {}
    for record in records:
        by_assembly.setdefault(record.assembly_group_id, set()).add(splits[record.geometry_id])
    assert all(len(group_splits) == 1 for group_splits in by_assembly.values())


def test_synthetic_counterfactual_group_has_label_diversity():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    group_records = [record for record in records if record.counterfactual_group_id == "cfg_000001"]
    assert len(group_records) >= 2
    assert len({record.labels.relation for record in group_records}) >= 2
    assert {record.changed_parameter for record in group_records} == {"transform_b.translation[0]"}


def test_intersectionedit_rows_expose_group_safe_internal_training_split_groups():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    splits = assign_geometry_splits(records, config.seed)
    rows = materialize_rows(
        records,
        splits,
        [
            TaskType.AXIS_ALIGNED_REPAIR,
            TaskType.AXIS_ALIGNED_REPAIR_VECTOR,
            TaskType.AXIS_ALIGNED_REPAIR_PROGRAM,
            TaskType.TARGET_CLEARANCE_REPAIR,
            TaskType.TARGET_CLEARANCE_MOVE,
            TaskType.TARGET_CONTACT_MOVE,
            TaskType.CENTROID_DISTANCE_MOVE,
            TaskType.EDIT_CANDIDATE_SELECTION,
            TaskType.EDIT_CANDIDATE_RANKING,
        ],
    )

    assert rows
    assert all(row.metadata.get("edit_split_group") for row in rows)
    assert all(row.metadata.get("edit_counterfactual_group_id") for row in rows)

    train_rows, eval_rows, report = partition_internal_train_eval_rows(
        rows,
        config.seed,
        eval_fraction=0.35,
    )

    assert train_rows
    assert eval_rows
    assert report["schema"] == "intersectionqa_internal_train_eval_split_v1"
    grouped_splits: dict[str, set[str]] = {}
    for row in train_rows:
        grouped_splits.setdefault(training_split_group(row), set()).add(INTERNAL_TRAIN_SPLIT)
    for row in eval_rows:
        grouped_splits.setdefault(training_split_group(row), set()).add(INTERNAL_EVAL_SPLIT)
    assert all(len(values) == 1 for values in grouped_splits.values())


def test_intersectionedit_rows_expose_counterfactual_edit_variant_metadata():
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    splits = assign_geometry_splits(records, config.seed)
    rows = materialize_rows(
        records,
        splits,
        [
            TaskType.AXIS_ALIGNED_REPAIR,
            TaskType.AXIS_ALIGNED_REPAIR_VECTOR,
            TaskType.AXIS_ALIGNED_REPAIR_PROGRAM,
            TaskType.TARGET_CLEARANCE_REPAIR,
            TaskType.TARGET_CLEARANCE_MOVE,
            TaskType.TARGET_CONTACT_MOVE,
            TaskType.CENTROID_DISTANCE_MOVE,
            TaskType.EDIT_CANDIDATE_SELECTION,
            TaskType.EDIT_CANDIDATE_RANKING,
        ],
    )

    counterfactual_rows = [
        row for row in rows if row.counterfactual_group_id == "cfg_000001"
    ]

    assert counterfactual_rows
    assert {training_split_group(row) for row in counterfactual_rows} == {
        "cfg_000001:intersectionedit_v01"
    }
    assert {
        row.metadata["edit_counterfactual_variant"]["source_variant_id"]
        for row in counterfactual_rows
    } >= {"cfg_000001_v01", "cfg_000001_v02"}
    assert all(
        "initial_translation" in row.metadata["edit_counterfactual_dimensions"]
        for row in counterfactual_rows
    )
    assert {
        row.metadata["edit_counterfactual_variant"]["edit_family"]
        for row in counterfactual_rows
    } >= {"axis_aligned_intersection_repair", "axis_aligned_intersection_repair_vector", "centroid_distance_move"}


def test_prepare_intersectionedit_training_splits_writes_group_safe_files(tmp_path):
    config = DatasetConfig()
    records = fixture_geometry_records(config.label_policy, config.config_hash)
    splits = assign_geometry_splits(records, config.seed)
    rows = materialize_rows(
        records,
        splits,
        [
            TaskType.AXIS_ALIGNED_REPAIR,
            TaskType.AXIS_ALIGNED_REPAIR_VECTOR,
            TaskType.TARGET_CLEARANCE_REPAIR,
            TaskType.TARGET_CLEARANCE_MOVE,
            TaskType.CENTROID_DISTANCE_MOVE,
            TaskType.EDIT_CANDIDATE_SELECTION,
        ],
    )
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "edit_splits"
    write_split_files(rows, dataset_dir)

    report = prepare_intersectionedit_training_splits(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        seed=config.seed,
        eval_fraction=0.35,
        source_splits=["train", "validation", "test_random", "test_object_pair_heldout", "test_near_boundary"],
        mode="sft",
    )

    train_rows = (output_dir / "inner_train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    eval_rows = (output_dir / "inner_eval.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert report["schema"] == "intersectionedit_training_splits_v1"
    assert 0 < report["selected_rows"] <= len(rows)
    assert report["selected_rows"] == len(train_rows) + len(eval_rows)
    assert train_rows
    assert eval_rows
    assert (output_dir / "report.json").exists()


def test_topology_heldout_split_uses_rare_topology_metadata():
    config = DatasetConfig()
    records = _cadevolve_like_records(
        [
            ("ring", ["ring"], ["circle", "revolve"]),
            ("box_a", ["box"], ["box"]),
            ("box_b", ["box"], ["box"]),
            ("box_c", ["box"], ["box"]),
            ("box_d", ["box"], ["box"]),
        ],
        config,
    )

    splits = assign_geometry_splits(records, config.seed)

    ring_record = next(record for record in records if record.metadata["topology_tags"] == ["ring"])
    assert splits[ring_record.geometry_id] == "test_topology_heldout"
    train_topologies = {
        tag
        for record in records
        if splits[record.geometry_id] == "train"
        for tag in record.metadata["topology_tags"]
    }
    assert "ring" not in train_topologies


def test_operation_heldout_split_uses_rare_cadquery_operation_metadata():
    config = DatasetConfig()
    records = _cadevolve_like_records(
        [
            ("filleted", ["box"], ["box", "fillet"]),
            ("box_a", ["box"], ["box"]),
            ("box_b", ["box"], ["box"]),
            ("box_c", ["box"], ["box"]),
            ("box_d", ["box"], ["box"]),
        ],
        config,
    )

    splits = assign_geometry_splits(records, config.seed)

    fillet_record = next(record for record in records if "fillet" in record.metadata["cadquery_ops"])
    assert splits[fillet_record.geometry_id] == "test_operation_heldout"
    train_ops = {
        op
        for record in records
        if splits[record.geometry_id] == "train"
        for op in record.metadata["cadquery_ops"]
    }
    assert "fillet" not in train_ops


def test_extension_split_exports_are_written_only_when_rows_use_them(tmp_path):
    config = DatasetConfig()
    records = _cadevolve_like_records(
        [
            ("ring", ["ring"], ["circle", "revolve"]),
            ("box_a", ["box"], ["box"]),
            ("box_b", ["box"], ["box"]),
            ("box_c", ["box"], ["box"]),
            ("box_d", ["box"], ["box"]),
        ],
        config,
    )
    splits = assign_geometry_splits(records, config.seed)
    rows = materialize_rows(records, splits, [TaskType.BINARY_INTERFERENCE])

    split_summary = write_split_files(rows, tmp_path)
    parquet_counts = write_parquet_files(rows, tmp_path / "parquet")
    manifest = split_manifest(rows)

    assert "test_topology_heldout" in split_summary
    assert (tmp_path / "test_topology_heldout.jsonl").exists()
    assert parquet_counts["test_topology_heldout.parquet"] > 0
    assert "test_topology_heldout" in manifest.split_names
    assert any(rule.rule_id == "topology_holdout" for rule in manifest.group_holdout_rules)
    assert not (tmp_path / "test_operation_heldout.jsonl").exists()


def _cadevolve_like_records(
    specs: list[tuple[str, list[str], list[str]]],
    config: DatasetConfig,
):
    fixtures = fixture_geometry_records(config.label_policy, config.config_hash)
    records = []
    for index, (name, topology_tags, cadquery_ops) in enumerate(specs, start=1):
        fixture = fixtures[(index - 1) % len(fixtures)]
        records.append(
            fixture.model_copy(
                update={
                    "geometry_id": f"geom_cadevolve_meta_{index:06d}",
                    "source": "cadevolve",
                    "base_object_pair_id": f"pair_meta_{index:06d}",
                    "assembly_group_id": f"asmgrp_meta_{index:06d}",
                    "counterfactual_group_id": None,
                    "variant_id": None,
                    "metadata": {
                        **fixture.metadata,
                        "fixture_name": name,
                        "generator_ids": ["gen_meta_shared"],
                        "topology_tags": topology_tags,
                        "cadquery_ops": cadquery_ops,
                    },
                }
            )
        )
    return records
