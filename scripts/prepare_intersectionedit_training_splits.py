"""Prepare group-safe IntersectionEdit inner train/eval JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from intersectionqa.evaluation.edit_difficulty import EDIT_TASK_TYPES
from intersectionqa.export.jsonl import read_jsonl, write_jsonl
from intersectionqa.schema import PublicTaskRow
from intersectionqa.splits.grouped import partition_internal_train_eval_rows, training_split_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20250214)
    parser.add_argument("--eval-fraction", type=float, default=0.10)
    parser.add_argument("--source-splits", nargs="+", default=["train"])
    parser.add_argument("--mode", choices=["sft", "rl"], default="sft")
    parser.add_argument("--scope", choices=["edit", "all"], default="edit")
    args = parser.parse_args()

    report = prepare_intersectionedit_training_splits(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        eval_fraction=args.eval_fraction,
        source_splits=args.source_splits,
        mode=args.mode,
        scope=args.scope,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def prepare_intersectionedit_training_splits(
    *,
    dataset_dir: Path,
    output_dir: Path,
    seed: int,
    eval_fraction: float,
    source_splits: list[str],
    mode: str,
    scope: str,
) -> dict[str, object]:
    rows = _load_source_rows(dataset_dir, source_splits)
    edit_rows = [
        row
        for row in rows
        if (scope == "all" or row.task_type in EDIT_TASK_TYPES) and _include_for_mode(row, mode)
    ]
    train_rows, eval_rows, split_report = partition_internal_train_eval_rows(
        edit_rows,
        seed,
        eval_fraction=eval_fraction,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_rows, output_dir / "inner_train.jsonl")
    write_jsonl(eval_rows, output_dir / "inner_eval.jsonl")
    report = {
        "schema": "intersectionedit_training_splits_v1",
        "mode": mode,
        "scope": scope,
        "source_splits": source_splits,
        "input_rows": len(rows),
        "selected_rows": len(edit_rows),
        "inner_split_report": split_report,
        "task_counts": dict(sorted(Counter(str(row.task_type) for row in edit_rows).items())),
        "group_counts": {
            "total": len({training_split_group(row) for row in edit_rows}),
            "inner_train": len({training_split_group(row) for row in train_rows}),
            "inner_eval": len({training_split_group(row) for row in eval_rows}),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _load_source_rows(dataset_dir: Path, source_splits: list[str]) -> list[PublicTaskRow]:
    rows: list[PublicTaskRow] = []
    for split in source_splits:
        path = dataset_dir / f"{split}.jsonl"
        if path.exists():
            rows.extend(read_jsonl(path))
    return rows


def _include_for_mode(row: PublicTaskRow, mode: str) -> bool:
    diagnostics = row.metadata.get("edit_diagnostics")
    if not isinstance(diagnostics, dict):
        return True
    key = "rl_include" if mode == "rl" else "sft_include"
    return diagnostics.get(key, True) is not False


if __name__ == "__main__":
    main()
