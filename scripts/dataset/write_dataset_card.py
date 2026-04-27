"""Write a Markdown dataset card for an exported IntersectionQA dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from intersectionqa.export.dataset_card import write_dataset_card
from intersectionqa.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    configure_logging()
    path = write_dataset_card(args.dataset_dir, args.output)
    print(path)


if __name__ == "__main__":
    main()
