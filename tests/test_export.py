from intersectionqa.config import DatasetConfig
from intersectionqa.export.jsonl import read_jsonl, write_jsonl
from intersectionqa.pipeline import build_smoke_rows


def test_jsonl_export_round_trips(tmp_path):
    config = DatasetConfig(output_dir=tmp_path)
    rows, report = build_smoke_rows(config)
    path = tmp_path / "rows.jsonl"
    assert write_jsonl(rows, path) == len(rows)
    loaded = read_jsonl(path)
    assert [row.id for row in loaded] == [row.id for row in rows]
    assert report["leakage_audit_status"] == "pass"
