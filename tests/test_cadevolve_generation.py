from intersectionqa.config import DatasetConfig
from intersectionqa.generation.cadevolve_assemblies import generate_cadevolve_geometry_records
from intersectionqa.generation.geometry_cache import GeometryLabelCache
from intersectionqa.sources.synthetic import synthetic_source_object
from intersectionqa.sources.validation import validate_source_object


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
    record = generated.records[0]
    assert record.source == "cadevolve"
    assert "def object_a():" in record.assembly_script
    assert "def object_b():" in record.assembly_script
    assert record.metadata["candidate_strategy"] == "clear_disjoint"
    assert record.metadata["source_paths"] == ["CADEvolve-C/test/b.py", "CADEvolve-P/test/a.py"]
    assert record.metadata["cadquery_version"] is not None


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
    assert "cache_hits=4" in second_logs
    assert all(record.metadata["geometry_label_cache_hit"] is True for record in second.records)
    assert list((tmp_path / "labels").glob("*/*.json"))
