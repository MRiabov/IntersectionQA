from intersectionqa.config import DatasetConfig
from intersectionqa.enums import Relation, TaskType
from intersectionqa.evaluation.repair import verify_repair_rows_exact
from intersectionqa.generation.cadevolve_assemblies import generate_cadevolve_geometry_records
from intersectionqa.generation.geometry_cache import GeometryLabelCache
from intersectionqa.hashing import sha256_json, sha256_text
from intersectionqa.prompts.materialize import materialize_rows
from intersectionqa.schema import Hashes, SourceObjectRecord
from intersectionqa.sources.synthetic import synthetic_source_object
from intersectionqa.sources.validation import validate_source_object
from intersectionqa.splits.grouped import assign_geometry_splits


def _cadevolve_box(object_id: str, source_path: str, dimensions: tuple[float, float, float]):
    source = synthetic_source_object(object_id, "object_source", dimensions)
    return source.model_copy(
        update={
            "source": "cadevolve",
            "source_id": source_path,
            "source_path": source_path,
            "source_license": "apache-2.0",
            "generator_id": "cadevolve_test_generator",
            "metadata": {
                **source.metadata,
                "source_tree": source_path.split("/", 1)[0],
                "source_subset": "/".join(source_path.split("/")[:2]),
            },
        }
    )


def _cadevolve_code_object(
    object_id: str,
    source_path: str,
    object_name: str,
    function_name: str,
    code: str,
    cadquery_ops: list[str],
    topology_tags: list[str],
) -> SourceObjectRecord:
    return SourceObjectRecord(
        object_id=object_id,
        source="cadevolve",
        source_id=source_path,
        generator_id="cadevolve_test_generator",
        source_path=source_path,
        source_license="apache-2.0",
        object_name=object_name,
        normalized_code=code,
        object_function_name=function_name,
        cadquery_ops=cadquery_ops,
        topology_tags=topology_tags,
        metadata={
            "units": "mm",
            "source_tree": source_path.split("/", 1)[0],
            "source_subset": "/".join(source_path.split("/")[:2]),
        },
        hashes=Hashes(
            source_code_hash=sha256_text(code),
            object_hash=sha256_json(
                {"source": "cadevolve", "source_path": source_path, "code": code}
            ),
            transform_hash=None,
            geometry_hash=None,
            config_hash=None,
            prompt_hash=None,
        ),
    )


def test_cadevolve_candidate_generation_aliases_sources_and_preserves_provenance():
    config = DatasetConfig()
    source_a = _cadevolve_box("obj_ca", "CADEvolve-P/test/a.py", (10.0, 10.0, 10.0))
    source_b = _cadevolve_box("obj_cb", "CADEvolve-C/test/b.py", (8.0, 8.0, 8.0))
    validations = [
        validate_source_object(
            source,
            config_hash=config.config_hash,
            validated_at_version="test",
            isolated=False,
        )
        for source in (source_a, source_b)
    ]
    generated = generate_cadevolve_geometry_records(
        [source_a, source_b],
        {validation.object_id: validation for validation in validations},
        policy=config.label_policy,
        config_hash=config.config_hash,
        max_records=4,
    )

    assert not generated.failures
    assert len(generated.records) == 4
    assert [record.geometry_id for record in generated.records] == [
        "geom_cadevolve_000001",
        "geom_cadevolve_000002",
        "geom_cadevolve_000003",
        "geom_cadevolve_000004",
    ]
    assert {record.labels.relation for record in generated.records} == {
        Relation.DISJOINT,
        Relation.TOUCHING,
        Relation.NEAR_MISS,
        Relation.INTERSECTING,
    }
    record = generated.records[0]
    assert record.source == "cadevolve"
    assert "def object_a():" in record.assembly_script
    assert "def object_b():" in record.assembly_script
    assert record.metadata["candidate_strategy"] == "clear_disjoint"
    assert record.metadata["candidate_transform_values"]["placement_direction"] in {
        "+x",
        "-x",
        "+y",
        "-y",
        "+z",
        "-z",
    }
    assert record.metadata["pre_balance_geometry_id"] == "geom_cadevolve_000001"
    assert record.metadata["source_paths"] == ["CADEvolve-C/test/b.py", "CADEvolve-P/test/a.py"]
    assert record.metadata["cadquery_version"] is not None


def test_cadevolve_gap_placement_spreads_repair_directions():
    config = DatasetConfig()
    sources = [
        _cadevolve_box("obj_ca", "CADEvolve-P/test/a.py", (10.0, 10.0, 10.0)),
        _cadevolve_box("obj_cb", "CADEvolve-C/test/b.py", (8.0, 8.0, 8.0)),
        _cadevolve_box("obj_cc", "CADEvolve-S/test/c.py", (7.0, 7.0, 7.0)),
        _cadevolve_box("obj_cd", "CADEvolve-T/test/d.py", (6.0, 6.0, 6.0)),
    ]
    validations = [
        validate_source_object(
            source,
            config_hash=config.config_hash,
            validated_at_version="test",
            isolated=False,
        )
        for source in sources
    ]
    generated = generate_cadevolve_geometry_records(
        sources,
        {validation.object_id: validation for validation in validations},
        policy=config.label_policy,
        config_hash=config.config_hash,
        max_records=24,
        relation_balance=False,
    )
    rows = materialize_rows(
        generated.records,
        assign_geometry_splits(generated.records, config.seed),
        [TaskType.REPAIR_DIRECTION],
    )
    directions = {row.answer for row in rows}

    assert directions & {"+x", "+y", "+z"}
    assert directions & {"-x", "-y", "-z"}


def test_cadevolve_candidate_generation_reuses_geometry_label_cache(tmp_path, capsys):
    config = DatasetConfig()
    source_a = _cadevolve_box("obj_ca", "CADEvolve-P/test/a.py", (10.0, 10.0, 10.0))
    source_b = _cadevolve_box("obj_cb", "CADEvolve-C/test/b.py", (8.0, 8.0, 8.0))
    validations = [
        validate_source_object(
            source,
            config_hash=config.config_hash,
            validated_at_version="test",
            isolated=False,
        )
        for source in (source_a, source_b)
    ]
    args = (
        [source_a, source_b],
        {validation.object_id: validation for validation in validations},
    )
    cache = GeometryLabelCache(tmp_path / "labels")

    first = generate_cadevolve_geometry_records(
        *args,
        policy=config.label_policy,
        config_hash=config.config_hash,
        max_records=4,
        geometry_cache=cache,
    )
    first_logs = capsys.readouterr().err
    second = generate_cadevolve_geometry_records(
        *args,
        policy=config.label_policy,
        config_hash=config.config_hash,
        max_records=4,
        geometry_cache=cache,
    )
    second_logs = capsys.readouterr().err

    assert [record.hashes.geometry_hash for record in second.records] == [
        record.hashes.geometry_hash for record in first.records
    ]
    assert [record.labels for record in second.records] == [record.labels for record in first.records]
    assert "cache_hits=0" in first_logs
    assert "cache_hits=8" in second_logs
    assert all(record.metadata["geometry_label_cache_hit"] is True for record in second.records)
    assert list((tmp_path / "labels").glob("*/*.json"))


def test_cadevolve_candidate_generation_includes_broad_placement_examples():
    config = DatasetConfig()
    source_a = _cadevolve_box("obj_ca", "CADEvolve-P/test/a.py", (10.0, 10.0, 10.0))
    source_b = _cadevolve_box("obj_cb", "CADEvolve-C/test/b.py", (8.0, 8.0, 8.0))
    validations = [
        validate_source_object(
            source,
            config_hash=config.config_hash,
            validated_at_version="test",
            isolated=False,
        )
        for source in (source_a, source_b)
    ]

    generated = generate_cadevolve_geometry_records(
        [source_a, source_b],
        {validation.object_id: validation for validation in validations},
        policy=config.label_policy,
        config_hash=config.config_hash,
        max_records=5,
    )

    assert any(record.metadata["candidate_strategy"] == "broad_random_disjoint" for record in generated.records)
    assert any("broad_placement" in record.difficulty_tags for record in generated.records)


def test_cadevolve_repair_rows_exact_verify_for_generated_overlap_examples():
    config = DatasetConfig()
    source_a = _cadevolve_box("obj_ca", "CADEvolve-P/test/a.py", (10.0, 10.0, 10.0))
    source_b = _cadevolve_box("obj_cb", "CADEvolve-C/test/b.py", (8.0, 8.0, 8.0))
    validations = [
        validate_source_object(
            source,
            config_hash=config.config_hash,
            validated_at_version="test",
            isolated=False,
        )
        for source in (source_a, source_b)
    ]
    generated = generate_cadevolve_geometry_records(
        [source_a, source_b],
        {validation.object_id: validation for validation in validations},
        policy=config.label_policy,
        config_hash=config.config_hash,
        max_records=4,
    )
    splits = assign_geometry_splits(generated.records, config.seed)
    rows = materialize_rows(
        generated.records,
        splits,
        [TaskType.REPAIR_DIRECTION, TaskType.REPAIR_TRANSLATION],
    )

    result = verify_repair_rows_exact(rows)

    assert rows
    assert any(row.source == "cadevolve" for row in rows)
    assert {TaskType.REPAIR_DIRECTION, TaskType.REPAIR_TRANSLATION} <= {
        row.task_type for row in rows
    }
    assert result.report["row_count"] == len(rows)
    assert result.report["repair_success_rate"] == 1.0


def test_cadevolve_candidate_generation_includes_cavity_targeted_examples():
    config = DatasetConfig()
    ring = _cadevolve_code_object(
        "obj_ring",
        "CADEvolve-P/test/ring.py",
        "ring_with_cutout",
        "ring_object",
        "\n".join(
            [
                "def ring_object():",
                "    outer = cq.Workplane('XY').circle(5.0).extrude(4.0)",
                "    inner = cq.Workplane('XY').circle(2.0).extrude(4.0)",
                "    return outer.cut(inner)",
                "",
            ]
        ),
        ["circle", "extrude", "cut"],
        ["ring", "cutout"],
    )
    probe = _cadevolve_code_object(
        "obj_probe",
        "CADEvolve-C/test/probe.py",
        "center_probe",
        "probe_object",
        "\n".join(
            [
                "def probe_object():",
                "    return cq.Workplane('XY').circle(0.5).extrude(6.0)",
                "",
            ]
        ),
        ["circle", "extrude"],
        ["cylinder"],
    )
    validations = [
        validate_source_object(
            source,
            config_hash=config.config_hash,
            validated_at_version="test",
            isolated=False,
        )
        for source in (ring, probe)
    ]

    generated = generate_cadevolve_geometry_records(
        [ring, probe],
        {validation.object_id: validation for validation in validations},
        policy=config.label_policy,
        config_hash=config.config_hash,
        max_records=8,
    )

    cavity_records = [
        record
        for record in generated.records
        if record.metadata["candidate_strategy"] == "cavity_center_probe"
    ]
    assert cavity_records
    assert any(record.labels.relation == Relation.DISJOINT for record in cavity_records)
    assert all("cavity_targeted" in record.difficulty_tags for record in cavity_records)
    assert any("aabb_exact_disagreement" in record.difficulty_tags for record in cavity_records)
    assert any(
        record.diagnostics.aabb_overlap is True and record.diagnostics.exact_overlap is False
        for record in cavity_records
    )
