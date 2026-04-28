"""Build a label-balanced reasoning-SFT canary dataset from an existing SFT dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--target-per-answer", type=int, default=64)
    parser.add_argument("--max-repeat-per-row", type=int, default=8)
    args = parser.parse_args()

    report = prepare_balanced_canary_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        target_per_answer=args.target_per_answer,
        max_repeat_per_row=args.max_repeat_per_row,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def prepare_balanced_canary_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    seed: int = 3407,
    target_per_answer: int = 64,
    max_repeat_per_row: int = 8,
) -> dict[str, Any]:
    if target_per_answer <= 0:
        raise ValueError("target_per_answer must be positive")
    if max_repeat_per_row <= 0:
        raise ValueError("max_repeat_per_row must be positive")

    train_rows = read_jsonl(input_dir / "train.jsonl")
    validation_rows = read_jsonl(input_dir / "validation.jsonl")
    balanced_train_rows, sampling_report = balanced_train_sample(
        train_rows,
        seed=seed,
        target_per_answer=target_per_answer,
        max_repeat_per_row=max_repeat_per_row,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(balanced_train_rows, output_dir / "train.jsonl")
    write_jsonl(validation_rows, output_dir / "validation.jsonl")

    for extra_name in ("manual_audit_results.md", "manual_audit_notes.md", "manual_audit_sample.jsonl"):
        source = input_dir / extra_name
        if source.exists():
            (output_dir / extra_name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    report = {
        "schema": "intersectionqa_balanced_reasoning_sft_canary_dataset_v1",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "target_per_answer": target_per_answer,
        "max_repeat_per_row": max_repeat_per_row,
        "sampling_report": sampling_report,
        "source": {
            "train": summarize_rows(train_rows),
            "validation": summarize_rows(validation_rows),
        },
        "balanced": {
            "train": summarize_rows(balanced_train_rows),
            "validation": summarize_rows(validation_rows),
        },
    }
    (output_dir / "balanced_reasoning_sft_canary_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def balanced_train_sample(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    target_per_answer: int,
    max_repeat_per_row: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_task_answer: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task_answer[(str(row["task_type"]), str(row["answer"]))].append(row)

    sampled_rows: list[dict[str, Any]] = []
    strata = []
    for (task_type, answer), stratum_rows in sorted(by_task_answer.items()):
        shuffled = list(stratum_rows)
        rng.shuffle(shuffled)
        target = min(target_per_answer, len(shuffled) * max_repeat_per_row)
        sampled = repeated_sample(shuffled, target=target, rng=rng)
        sampled_rows.extend(sampled)
        strata.append(
            {
                "task_type": task_type,
                "answer": answer,
                "source_count": len(stratum_rows),
                "sampled_count": len(sampled),
                "max_possible_with_repeat_cap": len(stratum_rows) * max_repeat_per_row,
            }
        )

    rng.shuffle(sampled_rows)
    return sampled_rows, {
        "source_row_count": len(rows),
        "sampled_row_count": len(sampled_rows),
        "strata": strata,
    }


def repeated_sample(rows: list[dict[str, Any]], *, target: int, rng: random.Random) -> list[dict[str, Any]]:
    if target <= len(rows):
        return list(rows[:target])
    sampled: list[dict[str, Any]] = []
    while len(sampled) < target:
        shuffled = list(rows)
        rng.shuffle(shuffled)
        sampled.extend(shuffled)
    return sampled[:target]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_task[str(row["task_type"])][str(row["answer"])] += 1
    return {
        "row_count": len(rows),
        "task_answer_counts": {
            task_type: dict(counter.most_common())
            for task_type, counter in sorted(by_task.items())
        },
    }


if __name__ == "__main__":
    main()
