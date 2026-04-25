"""Audit answer-label balance by split and task for an exported dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from intersectionqa.export.jsonl import read_jsonl
from intersectionqa.splits.grouped import DEFAULT_SPLITS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/IntersectionQA-90K"))
    parser.add_argument("--min-share", type=float, default=0.10)
    parser.add_argument("--max-share", type=float, default=0.70)
    parser.add_argument("--min-count", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = audit_dataset(args.dataset_dir, min_share=args.min_share, max_share=args.max_share, min_count=args.min_count)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print_human(report)


def audit_dataset(dataset_dir: Path, *, min_share: float, max_share: float, min_count: int) -> dict[str, object]:
    by_split_task: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    row_counts: Counter[str] = Counter()
    for split in DEFAULT_SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            split_name = str(row.split)
            row_counts[split_name] += 1
            by_split_task[split_name][str(row.task_type)][row.answer] += 1

    findings: list[dict[str, object]] = []
    splits: dict[str, object] = {}
    for split, by_task in sorted(by_split_task.items()):
        task_report: dict[str, object] = {}
        for task, counts in sorted(by_task.items()):
            total = sum(counts.values())
            answer_report = {
                answer: {
                    "count": count,
                    "share": count / total if total else 0.0,
                }
                for answer, count in sorted(counts.items())
            }
            task_report[task] = {
                "total": total,
                "answers": answer_report,
            }
            for answer, count in sorted(counts.items()):
                share = count / total if total else 0.0
                if count < min_count or share < min_share or share > max_share:
                    findings.append(
                        {
                            "split": split,
                            "task_type": task,
                            "answer": answer,
                            "count": count,
                            "share": share,
                            "total": total,
                            "reason": _reason(count, share, min_count, min_share, max_share),
                        }
                    )
        splits[split] = {"row_count": row_counts[split], "tasks": task_report}
    return {
        "dataset_dir": str(dataset_dir),
        "thresholds": {
            "min_share": min_share,
            "max_share": max_share,
            "min_count": min_count,
        },
        "finding_count": len(findings),
        "findings": findings,
        "splits": splits,
    }


def _reason(count: int, share: float, min_count: int, min_share: float, max_share: float) -> str:
    reasons: list[str] = []
    if count < min_count:
        reasons.append("low_count")
    if share < min_share:
        reasons.append("low_share")
    if share > max_share:
        reasons.append("high_share")
    return ",".join(reasons)


def print_human(report: dict[str, object]) -> None:
    print(f"dataset_dir: {report['dataset_dir']}")
    print(f"findings: {report['finding_count']}")
    splits = report["splits"]
    assert isinstance(splits, dict)
    for split, split_report in splits.items():
        assert isinstance(split_report, dict)
        print(f"\n[{split}] rows={split_report['row_count']}")
        tasks = split_report["tasks"]
        assert isinstance(tasks, dict)
        for task, task_report in tasks.items():
            assert isinstance(task_report, dict)
            total = int(task_report["total"])
            answers = task_report["answers"]
            assert isinstance(answers, dict)
            pieces = [
                f"{answer}:{item['count']} ({item['share']:.1%})"
                for answer, item in sorted(
                    answers.items(),
                    key=lambda kv: (-kv[1]["count"], kv[0]),
                )
            ]
            print(f"  {task} n={total}: {', '.join(pieces)}")
    findings = report["findings"]
    assert isinstance(findings, list)
    if findings:
        print("\nfindings:")
        for finding in findings:
            print(
                "  "
                f"{finding['split']} {finding['task_type']} {finding['answer']}: "
                f"{finding['count']}/{finding['total']} ({finding['share']:.1%}) "
                f"[{finding['reason']}]"
            )


if __name__ == "__main__":
    main()
