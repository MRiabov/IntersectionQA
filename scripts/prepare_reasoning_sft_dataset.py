"""Prepare internal tagged-reasoning SFT rows from canonical public rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.training.reasoning_traces import write_reasoning_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["inner_train", "inner_eval"])
    args = parser.parse_args()

    report = write_reasoning_sft_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        splits=args.splits,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
