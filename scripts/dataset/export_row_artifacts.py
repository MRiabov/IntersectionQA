"""Export local debug artifacts for one public dataset row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.logging import configure_logging
from scripts.dataset.internal.row_artifacts import export_row_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("row_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-step", action="store_true", help="write text/JSON artifacts only")
    args = parser.parse_args()

    configure_logging()
    manifest = export_row_artifacts(
        args.dataset_dir,
        args.row_id,
        args.output_dir,
        write_step=not args.no_step,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

