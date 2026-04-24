import io
import json
import tarfile

import pytest

import intersectionqa.sources.cadevolve as cadevolve
from intersectionqa.sources.cadevolve import CadevolveTarLoader


def test_cadevolve_loader_reads_executable_prefixes(tmp_path):
    archive_path = tmp_path / "cadevolve.tar"
    with tarfile.open(archive_path, "w") as archive:
        code = b"import cadquery as cq\nresult = cq.Workplane('XY').box(1, 2, 3)\n"
        info = tarfile.TarInfo("./CADEvolve-P/example.py")
        info.size = len(code)
        archive.addfile(info, io.BytesIO(code))

        ignored = b"result = None\n"
        ignored_info = tarfile.TarInfo("./CADEvolve-G/metadata.py")
        ignored_info.size = len(ignored)
        archive.addfile(ignored_info, io.BytesIO(ignored))

    result = CadevolveTarLoader(archive_path, "sha256:" + "0" * 64).load()
    assert result.scanned_count == 1
    assert len(result.records) == 1
    record = result.records[0]
    assert record.source == "cadevolve"
    assert record.source_path == "CADEvolve-P/example.py"
    assert record.generator_id == "cadevolve_cadevolve_p"
    assert "object_source" in record.normalized_code
    assert record.metadata["validation_status"] == "not_run"


def test_cadevolve_loader_sorts_members_before_limiting(tmp_path):
    archive_path = tmp_path / "cadevolve.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name in [
            "./CADEvolve-P/z.py",
            "./CADEvolve-P/a.py",
            "./CADEvolve-P/m.py",
        ]:
            code = f"import cadquery as cq\nresult = cq.Workplane('XY').box({len(name)}, 1, 1)\n".encode()
            info = tarfile.TarInfo(name)
            info.size = len(code)
            archive.addfile(info, io.BytesIO(code))

    result = CadevolveTarLoader(archive_path, "sha256:" + "0" * 64).load(limit=2)

    assert [record.source_path for record in result.records] == [
        "CADEvolve-P/a.py",
        "CADEvolve-P/m.py",
    ]
    assert [record.object_id for record in result.records] == [
        "obj_cadevolve_000001",
        "obj_cadevolve_000002",
    ]


def test_cadevolve_loader_applies_offset_after_sorting(tmp_path):
    archive_path = tmp_path / "cadevolve.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name in [
            "./CADEvolve-P/z.py",
            "./CADEvolve-P/a.py",
            "./CADEvolve-P/m.py",
        ]:
            code = b"import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)\n"
            info = tarfile.TarInfo(name)
            info.size = len(code)
            archive.addfile(info, io.BytesIO(code))

    result = CadevolveTarLoader(archive_path, "sha256:" + "0" * 64).load(limit=1, offset=1)

    assert [record.source_path for record in result.records] == ["CADEvolve-P/m.py"]
    assert [record.object_id for record in result.records] == ["obj_cadevolve_000001"]


def test_cadevolve_loader_reuses_member_index_cache(tmp_path, monkeypatch):
    archive_path = tmp_path / "cadevolve.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name in [
            "./CADEvolve-P/z.py",
            "./CADEvolve-P/a.py",
        ]:
            code = b"import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)\n"
            info = tarfile.TarInfo(name)
            info.size = len(code)
            archive.addfile(info, io.BytesIO(code))

    cache_dir = tmp_path / "index-cache"
    loader = CadevolveTarLoader(
        archive_path,
        "sha256:" + "0" * 64,
        member_index_cache_dir=cache_dir,
    )
    first = loader.load(limit=1)

    real_open = tarfile.open

    def fail_open(*args, **kwargs):
        raise AssertionError("tarfile.open should not be needed when member index is cached")

    monkeypatch.setattr(tarfile, "open", fail_open)
    try:
        second = loader.load(limit=1)
    finally:
        monkeypatch.setattr(tarfile, "open", real_open)

    assert [record.source_path for record in first.records] == ["CADEvolve-P/a.py"]
    assert [record.source_path for record in second.records] == ["CADEvolve-P/a.py"]
    assert list(cache_dir.glob("*/*.json"))


def test_cadevolve_loader_materializes_extracted_source_cache(tmp_path, monkeypatch):
    archive_path = tmp_path / "cadevolve.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name in [
            "./CADEvolve-P/z.py",
            "./CADEvolve-P/a.py",
        ]:
            code = f"import cadquery as cq\nresult = cq.Workplane('XY').box({len(name)}, 1, 1)\n".encode()
            info = tarfile.TarInfo(name)
            info.size = len(code)
            archive.addfile(info, io.BytesIO(code))

    index_cache_dir = tmp_path / "index-cache"
    extracted_cache_dir = tmp_path / "source-cache"
    loader = CadevolveTarLoader(
        archive_path,
        "sha256:" + "0" * 64,
        member_index_cache_dir=index_cache_dir,
        extracted_source_cache_dir=extracted_cache_dir,
    )
    first = loader.load(limit=1)
    cache_root = loader.extracted_source_cache_root()
    assert cache_root is not None
    assert (cache_root / "CADEvolve-P" / "a.py").exists()
    manifest = json.loads((cache_root / "extraction_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "intersectionqa_cadevolve_extracted_sources_v1"
    assert manifest["prefix_count"] == 1
    assert manifest["members"][0]["path"] == "CADEvolve-P/a.py"

    def fail_read(*args, **kwargs):
        raise AssertionError("tar member reads should not be needed when source cache is warm")

    def fail_index(*args, **kwargs):
        raise AssertionError("member index should not be needed when extracted prefix is warm")

    monkeypatch.setattr(cadevolve, "_read_indexed_member", fail_read)
    monkeypatch.setattr(CadevolveTarLoader, "_load_executable_member_index", fail_index)
    second = loader.load(limit=1)

    assert [record.source_path for record in first.records] == ["CADEvolve-P/a.py"]
    assert [record.hashes.source_code_hash for record in second.records] == [
        record.hashes.source_code_hash for record in first.records
    ]


def test_cadevolve_loader_prepares_extracted_source_prefix(tmp_path, monkeypatch):
    archive_path = tmp_path / "cadevolve.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name in [
            "./CADEvolve-P/z.py",
            "./CADEvolve-P/a.py",
            "./CADEvolve-P/m.py",
        ]:
            code = b"import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)\n"
            info = tarfile.TarInfo(name)
            info.size = len(code)
            archive.addfile(info, io.BytesIO(code))

    loader = CadevolveTarLoader(
        archive_path,
        "sha256:" + "0" * 64,
        member_index_cache_dir=tmp_path / "index-cache",
        extracted_source_cache_dir=tmp_path / "source-cache",
    )

    report = loader.prepare_extracted_sources(limit=2)

    assert report.selected_count == 2
    assert report.newly_extracted_count == 2
    assert report.already_present_count == 0
    assert (report.cache_root / "CADEvolve-P" / "a.py").exists()
    assert (report.cache_root / "CADEvolve-P" / "m.py").exists()

    def fail_index(*args, **kwargs):
        raise AssertionError("member index should not be needed after preparing prefix")

    monkeypatch.setattr(CadevolveTarLoader, "_load_executable_member_index", fail_index)
    second_report = loader.prepare_extracted_sources(limit=2)
    loaded = loader.load(limit=2)

    assert second_report.newly_extracted_count == 0
    assert second_report.already_present_count == 2
    assert [record.source_path for record in loaded.records] == [
        "CADEvolve-P/a.py",
        "CADEvolve-P/m.py",
    ]


def test_cadevolve_loader_reads_prepared_source_cache_without_archive(tmp_path, monkeypatch):
    archive_path = tmp_path / "cadevolve.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name in [
            "./CADEvolve-P/z.py",
            "./CADEvolve-P/a.py",
        ]:
            code = b"import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)\n"
            info = tarfile.TarInfo(name)
            info.size = len(code)
            archive.addfile(info, io.BytesIO(code))

    loader = CadevolveTarLoader(
        archive_path,
        "sha256:" + "0" * 64,
        extracted_source_cache_dir=tmp_path / "source-cache",
    )
    report = loader.prepare_extracted_sources(limit=2)

    def fail_index(*args, **kwargs):
        raise AssertionError("member index should not be needed when direct source cache is used")

    monkeypatch.setattr(CadevolveTarLoader, "_load_executable_member_index", fail_index)
    cached_loader = CadevolveTarLoader(
        None,
        "sha256:" + "0" * 64,
        extracted_source_cache_root=report.cache_root,
    )
    result = cached_loader.load(limit=2)

    assert result.scanned_count == 2
    assert [record.source_path for record in result.records] == [
        "CADEvolve-P/a.py",
        "CADEvolve-P/z.py",
    ]


def test_cadevolve_loader_rejects_invalid_explicit_source_cache_root(tmp_path):
    loader = CadevolveTarLoader(
        None,
        "sha256:" + "0" * 64,
        extracted_source_cache_root=tmp_path,
    )

    with pytest.raises(FileNotFoundError, match="explicit CADEvolve source cache root"):
        loader.load(limit=1)
