"""Rebuild exported pairwise rows with balanced answer classes per counterfactual group."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from intersectionqa.enums import Relation, Split, TaskType
from intersectionqa.export.dataset_card import write_dataset_card
from intersectionqa.export.jsonl import (
    build_metadata,
    read_jsonl,
    read_metadata,
    validate_rows,
    write_metadata,
    write_split_files,
)
from intersectionqa.export.parquet import write_parquet_files
from intersectionqa.hashing import sha256_json
from intersectionqa.prompts.counterfactual import TEMPLATE_VERSION
from intersectionqa.prompts.common import TASK_PREFIX
from intersectionqa.schema import Hashes, PublicTaskRow
from intersectionqa.splits.grouped import DEFAULT_SPLITS, audit_group_leakage, split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/IntersectionQA-90K"))
    parser.add_argument(
        "--cap-answer-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Downsample rebuilt pairwise rows so A/B/both/neither have equal counts within each split.",
    )
    parser.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    rows = [row for split in DEFAULT_SPLITS for row in read_jsonl(dataset_dir / f"{split}.jsonl")]
    metadata = read_metadata(dataset_dir / "metadata.json")
    if metadata is None:
        raise ValueError(f"missing metadata.json in {dataset_dir}")

    existing_pairwise = [row for row in rows if row.task_type == TaskType.PAIRWISE_INTERFERENCE]
    non_pairwise = [row for row in rows if row.task_type != TaskType.PAIRWISE_INTERFERENCE]
    relation_by_split_group: dict[tuple[str, str], list[PublicTaskRow]] = defaultdict(list)
    for row in rows:
        if row.task_type != TaskType.RELATION_CLASSIFICATION or not row.counterfactual_group_id:
            continue
        relation_by_split_group[(str(row.split), row.counterfactual_group_id)].append(row)

    rebuilt_pairwise: list[PublicTaskRow] = []
    next_id = 1
    skipped_groups = 0
    for (split, group_id), group_rows in sorted(relation_by_split_group.items()):
        group_rows = sorted(group_rows, key=lambda row: row.geometry_ids[0])
        pairs = balanced_pairs(group_rows)
        if not pairs:
            skipped_groups += 1
            continue
        for row_a, row_b in pairs:
            rebuilt_pairwise.append(make_pairwise_row(row_a, row_b, next_id, split, group_id))
            next_id += 1

    uncapped_pairwise = rebuilt_pairwise
    if args.cap_answer_classes:
        rebuilt_pairwise = cap_answer_classes(rebuilt_pairwise)

    updated_rows = non_pairwise + rebuilt_pairwise
    validate_rows(updated_rows)
    audit = audit_group_leakage(updated_rows)
    if audit.status != "pass":
        raise ValueError(f"group leakage audit failed after pairwise rebalance: {audit.violations}")

    summary = {
        "dataset_dir": str(dataset_dir),
        "old_pairwise_rows": len(existing_pairwise),
        "old_pairwise_answer_counts": dict(Counter(row.answer for row in existing_pairwise)),
        "uncapped_pairwise_rows": len(uncapped_pairwise),
        "uncapped_pairwise_answer_counts": dict(Counter(row.answer for row in uncapped_pairwise)),
        "new_pairwise_rows": len(rebuilt_pairwise),
        "new_pairwise_answer_counts": dict(Counter(row.answer for row in rebuilt_pairwise)),
        "new_pairwise_answer_counts_by_split": answer_counts_by_split(rebuilt_pairwise),
        "counterfactual_relation_groups": len(relation_by_split_group),
        "skipped_groups": skipped_groups,
        "new_split_counts": dict(Counter(row.split for row in updated_rows)),
        "new_train_task_counts": dict(Counter(row.task_type for row in updated_rows if row.split == "train")),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return

    if args.backup:
        backup_dir = dataset_dir / "pre_pairwise_rebalance_backup"
        backup_dir.mkdir(exist_ok=True)
        for name in [
            *[f"{split}.jsonl" for split in DEFAULT_SPLITS],
            "metadata.json",
            "split_manifest.json",
            "parquet_manifest.json",
            "DATASET_CARD.md",
            "README.md",
        ]:
            path = dataset_dir / name
            if path.exists():
                shutil.copy2(path, backup_dir / name)

    split_summary = write_split_files(updated_rows, dataset_dir)
    (dataset_dir / "split_manifest.json").write_text(
        split_manifest(updated_rows).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    write_metadata(
        build_metadata(
            dataset_version=metadata.dataset_version,
            config_hash=metadata.config_hash,
            source_manifest_hash=metadata.source_manifest_hash,
            label_policy=metadata.label_policy,
            splits=split_summary,
            rows=updated_rows,
            license=metadata.license,
        ),
        dataset_dir / "metadata.json",
    )
    parquet_counts = write_parquet_files(updated_rows, dataset_dir / "parquet")
    (dataset_dir / "parquet_manifest.json").write_text(
        json.dumps({"files": parquet_counts, "compression": "zstd"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_dataset_card(dataset_dir)


def balanced_pairs(rows: list[PublicTaskRow]) -> list[tuple[PublicTaskRow, PublicTaskRow]]:
    positives = [row for row in rows if _has_interference(row)]
    negatives = [row for row in rows if not _has_interference(row)]
    pairs: list[tuple[PublicTaskRow, PublicTaskRow]] = []
    if positives and negatives:
        pairs.append((positives[0], negatives[0]))
        pairs.append((negatives[0], positives[0]))
    if len(positives) >= 2:
        pairs.append((positives[0], positives[1]))
    if len(negatives) >= 2:
        pairs.append((negatives[0], negatives[1]))
    return pairs


def cap_answer_classes(rows: list[PublicTaskRow]) -> list[PublicTaskRow]:
    by_split_answer: dict[tuple[str, str], list[PublicTaskRow]] = defaultdict(list)
    for row in rows:
        by_split_answer[(str(row.split), row.answer)].append(row)

    kept: list[PublicTaskRow] = []
    for split in sorted({split for split, _ in by_split_answer}):
        split_answers = {
            answer: by_split_answer.get((split, answer), [])
            for answer in ["A", "B", "both", "neither"]
        }
        if any(not answer_rows for answer_rows in split_answers.values()):
            for answer_rows in split_answers.values():
                kept.extend(answer_rows)
            continue
        cap = min(len(answer_rows) for answer_rows in split_answers.values())
        for answer, answer_rows in split_answers.items():
            kept.extend(sorted(answer_rows, key=stable_row_score)[:cap])
    return sorted(kept, key=lambda row: row.id)


def answer_counts_by_split(rows: list[PublicTaskRow]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[str(row.split)][row.answer] += 1
    return {split: dict(counter) for split, counter in sorted(counts.items())}


def stable_row_score(row: PublicTaskRow) -> str:
    return sha256_json(
        {
            "answer": row.answer,
            "counterfactual_group_id": row.counterfactual_group_id,
            "geometry_ids": row.geometry_ids,
            "variant_id": row.variant_id,
        }
    )


def make_pairwise_row(
    row_a: PublicTaskRow,
    row_b: PublicTaskRow,
    row_number: int,
    split: str,
    group_id: str,
) -> PublicTaskRow:
    answer = pairwise_answer(row_a, row_b)
    prompt = make_pairwise_prompt(row_a, row_b)
    geometry_ids = [row_a.geometry_ids[0], row_b.geometry_ids[0]]
    metadata = {
        "prompt_template_version": TEMPLATE_VERSION,
        "split_group": group_id,
        "candidate_strategy": "counterfactual_derived_prompt",
        "source_subtrees": row_a.metadata.get("source_subtrees"),
        "generator_ids": row_a.metadata.get("generator_ids"),
        "artifact_ids": row_a.metadata.get("artifact_ids", {}),
        "source_geometry_ids": geometry_ids,
        "source_variant_ids": [row_a.variant_id, row_b.variant_id],
        "counterfactual_prompt_format": "pairwise_interference",
        "variant_labels": {
            "A": _variant_label(row_a),
            "B": _variant_label(row_b),
        },
        "rebuilt_from_public_rows": True,
    }
    return row_a.model_copy(
        update={
            "id": f"{TASK_PREFIX[TaskType.PAIRWISE_INTERFERENCE]}_{row_number:06d}",
            "split": Split(split),
            "task_type": TaskType.PAIRWISE_INTERFERENCE,
            "prompt": prompt,
            "answer": answer,
            "script": "\n\n".join([row_a.script, row_b.script]),
            "geometry_ids": geometry_ids,
            "variant_id": f"{group_id}_pairwise_{row_number:06d}",
            "changed_value": [row_a.changed_value, row_b.changed_value],
            "labels": row_a.labels,
            "diagnostics": row_a.diagnostics,
            "difficulty_tags": sorted(set(row_a.difficulty_tags) | set(row_b.difficulty_tags)),
            "hashes": group_hashes([row_a, row_b], prompt, geometry_ids),
            "metadata": metadata,
        }
    )


def make_pairwise_prompt(row_a: PublicTaskRow, row_b: PublicTaskRow) -> str:
    return f"""You are given one CadQuery object-pair definition and two assembly transform variants.

Assume:
- Units are millimetres.
- Euler rotations are XYZ order, in degrees.
- Interference means positive-volume overlap.
- Touching at a face, edge, or point is not interference.
- Do not execute code.

Object code:
```python
{object_code(row_a).strip()}
```

Variant A ({row_a.variant_id or row_a.geometry_ids[0]}):
{transforms(row_a)}

Variant B ({row_b.variant_id or row_b.geometry_ids[0]}):
{transforms(row_b)}

Question: Which variant or variants have positive-volume interference?

Allowed answers:
A
B
both
neither

Answer with exactly one allowed answer."""


def object_code(row: PublicTaskRow) -> str:
    marker = "Object code:\n```python\n"
    start = row.prompt.index(marker) + len(marker)
    end = row.prompt.index("\n```\n\nTransforms:", start)
    return row.prompt[start:end].strip() + "\n"


def transforms(row: PublicTaskRow) -> str:
    marker = "\n\nTransforms:\n"
    start = row.prompt.index(marker) + len(marker)
    end_marker = "\n\nAnswer with exactly one label."
    end = row.prompt.index(end_marker, start)
    return row.prompt[start:end].strip()


def pairwise_answer(row_a: PublicTaskRow, row_b: PublicTaskRow) -> str:
    a_interferes = _has_interference(row_a)
    b_interferes = _has_interference(row_b)
    if a_interferes and b_interferes:
        return "both"
    if a_interferes:
        return "A"
    if b_interferes:
        return "B"
    return "neither"


def _variant_label(row: PublicTaskRow) -> dict[str, object]:
    return {
        "geometry_id": row.geometry_ids[0],
        "variant_id": row.variant_id,
        "relation": row.answer,
        "interferes": _has_interference(row),
    }


def _has_interference(row: PublicTaskRow) -> bool:
    return row.answer in {Relation.INTERSECTING, Relation.CONTAINED, "intersecting", "contained"}


def group_hashes(rows: list[PublicTaskRow], prompt: str, geometry_ids: list[str]) -> Hashes:
    return Hashes(
        source_code_hash=sha256_json([row.hashes.source_code_hash for row in rows]),
        object_hash=sha256_json([row.hashes.object_hash for row in rows]),
        transform_hash=sha256_json([row.hashes.transform_hash for row in rows]),
        geometry_hash=sha256_json([row.hashes.geometry_hash for row in rows]),
        config_hash=rows[0].hashes.config_hash,
        prompt_hash=sha256_json(
            {
                "template_version": TEMPLATE_VERSION,
                "task_type": TaskType.PAIRWISE_INTERFERENCE,
                "prompt": prompt,
                "geometry_ids": geometry_ids,
            }
        ),
    )


if __name__ == "__main__":
    main()
