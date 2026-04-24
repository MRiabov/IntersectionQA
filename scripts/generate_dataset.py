"""Generate the v0.1 smoke dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.config import load_config
from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import write_smoke_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cadevolve-archive", type=Path, default=None)
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.cadevolve_archive is not None:
        config.cadevolve_archive = args.cadevolve_archive
    report = write_smoke_dataset(config)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
