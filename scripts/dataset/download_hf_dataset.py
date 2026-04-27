"""Download a Hugging Face dataset snapshot into a local dataset directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="MRiabov/IntersectionQA-90K")
    parser.add_argument("--output-dir", type=Path, default=Path("data/IntersectionQA-90K"))
    parser.add_argument("--revision")
    parser.add_argument("--allow-pattern", action="append", dest="allow_patterns")
    parser.add_argument("--ignore-pattern", action="append", dest="ignore_patterns")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output_dir,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
    )
    report = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "output_dir": str(args.output_dir),
        "snapshot_path": str(path),
    }
    (args.output_dir / "hf_download_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

