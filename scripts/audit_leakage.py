"""Run split group-leakage audit for an exported dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import validate_dataset_dir
from intersectionqa.splits.grouped import audit_group_leakage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()

    configure_logging()
    rows = validate_dataset_dir(args.dataset_dir)
    audit = audit_group_leakage(rows)
    print(json.dumps(audit.__dict__, indent=2, sort_keys=True))
    if audit.status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
