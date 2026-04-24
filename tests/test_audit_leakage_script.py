from intersectionqa.config import DatasetConfig
from intersectionqa.pipeline import build_smoke_rows
from intersectionqa.splits.grouped import audit_group_leakage


def test_audit_group_leakage_passes_smoke_rows():
    rows, _ = build_smoke_rows(DatasetConfig())
    audit = audit_group_leakage(rows)
    assert audit.status == "pass"
    assert audit.violation_count == 0
