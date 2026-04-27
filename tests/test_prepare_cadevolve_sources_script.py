import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path


def test_prepare_cadevolve_sources_script_materializes_prefix(tmp_path):
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

    cache_dir = tmp_path / "source-cache"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.dataset.prepare_cadevolve_sources",
            "--cadevolve-archive",
            str(archive_path),
            "--cache-dir",
            str(cache_dir),
            "--limit",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["selected_count"] == 2
    assert payload["newly_extracted_count"] == 2
    assert payload["already_present_count"] == 0
    assert Path(payload["cache_root"]).exists()
    assert list(cache_dir.glob("*/*/CADEvolve-P/a.py"))
    assert list(cache_dir.glob("*/*/CADEvolve-P/m.py"))
