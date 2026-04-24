import io
import tarfile

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
