"""Print compact dataset statistics for exported JSONL rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.evaluation.metrics import dataset_stats
from intersectionqa.logging import configure_logging
from intersectionqa.pipeline import validate_dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()

    configure_logging()
    rows = validate_dataset_dir(args.dataset_dir)
    print(json.dumps(dataset_stats(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
