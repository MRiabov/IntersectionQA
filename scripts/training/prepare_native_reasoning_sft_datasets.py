"""Build short- and long-trace SFT datasets from accepted native reasoning rows."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import re
from pathlib import Path
from typing import Any, Iterable

from intersectionqa.schema import PublicTaskRow
from intersectionqa.splits.grouped import partition_internal_train_eval_rows, training_split_group


EXTRA_SUPERVISION_KEYS = {"target_text", "canonical_answer", "supervision"}
THINK_RE = re.compile(r"<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>", re.DOTALL)
TARGET_FORMAT = "native_reasoning_think_answer_v01"
SHORT_TARGET_FORMAT = "native_reasoning_short_think_answer_v01"
SHORTENER_ID = "deterministic_native_reasoning_sentence_extractor_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="accepted_reasoning_sft.jsonl files")
    parser.add_argument("--short-output-dir", type=Path, required=True)
    parser.add_argument("--long-output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval-fraction", type=float, default=0.12)
    parser.add_argument("--min-short-tokens", type=int, default=128)
    parser.add_argument("--max-short-tokens", type=int, default=256)
    parser.add_argument("--audit-per-task", type=int, default=3)
    args = parser.parse_args()

    report = prepare_native_reasoning_sft_datasets(
        input_paths=args.input,
        short_output_dir=args.short_output_dir,
        long_output_dir=args.long_output_dir,
        seed=args.seed,
        eval_fraction=args.eval_fraction,
        min_short_tokens=args.min_short_tokens,
        max_short_tokens=args.max_short_tokens,
        audit_per_task=args.audit_per_task,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def prepare_native_reasoning_sft_datasets(
    *,
    input_paths: Iterable[Path],
    short_output_dir: Path,
    long_output_dir: Path,
    seed: int = 3407,
    eval_fraction: float = 0.12,
    min_short_tokens: int = 128,
    max_short_tokens: int = 256,
    audit_per_task: int = 3,
) -> dict[str, Any]:
    if min_short_tokens <= 0 or max_short_tokens < min_short_tokens:
        raise ValueError("short token bounds must be positive and ordered")
    loaded = load_accepted_rows(input_paths)
    public_rows = [public_row_from_payload(row) for row in loaded]
    train_public, eval_public, split_report = partition_internal_train_eval_rows(
        public_rows,
        seed,
        eval_fraction=eval_fraction,
        balance_task_answers=True,
    )
    split_by_id = {row.id: "validation" for row in eval_public}
    split_by_id.update({row.id: "train" for row in train_public})

    rows_by_split = {
        "train": [row for row in loaded if split_by_id[public_row_from_payload(row).id] == "train"],
        "validation": [row for row in loaded if split_by_id[public_row_from_payload(row).id] == "validation"],
    }

    short_rows = {
        split: [
            short_reasoning_payload(
                row,
                min_tokens=min_short_tokens,
                max_tokens=max_short_tokens,
            )
            for row in split_rows
        ]
        for split, split_rows in rows_by_split.items()
    }
    long_rows = {
        split: [long_reasoning_payload(row) for row in split_rows]
        for split, split_rows in rows_by_split.items()
    }

    write_dataset(short_output_dir, short_rows)
    write_dataset(long_output_dir, long_rows)
    write_public_grpo_views(short_output_dir, rows_by_split)
    write_public_grpo_views(long_output_dir, rows_by_split)

    audit_rows = audit_sample(short_rows["train"] + short_rows["validation"], per_task=audit_per_task)
    write_jsonl(audit_rows, short_output_dir / "manual_audit_sample.jsonl")
    (short_output_dir / "manual_audit_notes.md").write_text(
        manual_audit_template(audit_rows),
        encoding="utf-8",
    )

    report = {
        "schema": "intersectionqa_native_reasoning_sft_datasets_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_paths": [str(path) for path in input_paths],
        "short_output_dir": str(short_output_dir),
        "long_output_dir": str(long_output_dir),
        "seed": seed,
        "eval_fraction": eval_fraction,
        "split_report": split_report,
        "source": summarize_rows(loaded),
        "short": summarize_dataset(short_rows),
        "long": summarize_dataset(long_rows),
        "audit_sample_path": str(short_output_dir / "manual_audit_sample.jsonl"),
        "manual_audit_notes_path": str(short_output_dir / "manual_audit_notes.md"),
        "shortening_policy": shortening_policy(min_short_tokens, max_short_tokens),
    }
    (short_output_dir / "native_reasoning_sft_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (long_output_dir / "native_reasoning_sft_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def load_accepted_rows(input_paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in input_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = str(row.get("id", ""))
                if not row_id:
                    raise ValueError(f"{path}:{line_number}: missing id")
                if row_id in seen:
                    continue
                seen.add(row_id)
                row["_source_trace_run"] = path.parent.name
                row["_source_trace_path"] = str(path)
                rows.append(row)
    return rows


def public_row_from_payload(payload: dict[str, Any]) -> PublicTaskRow:
    return PublicTaskRow.model_validate({key: value for key, value in payload.items() if key not in EXTRA_SUPERVISION_KEYS and not key.startswith("_")})


def short_reasoning_payload(
    payload: dict[str, Any],
    *,
    min_tokens: int,
    max_tokens: int,
) -> dict[str, Any]:
    source_reasoning, final_answer = extract_target(payload)
    if final_answer != payload["answer"]:
        raise ValueError(f"{payload['id']}: target answer does not match canonical answer")
    reasoning = shorten_reasoning(
        source_reasoning,
        task_type=str(payload["task_type"]),
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    row = clean_payload(payload)
    row["target_text"] = f"<think>{reasoning}</think><answer>{payload['answer']}</answer>"
    row["canonical_answer"] = payload["answer"]
    row["supervision"] = {
        **source_supervision(payload),
        "target_text_format": SHORT_TARGET_FORMAT,
        "target_text_source": "openrouter_native_reasoning_shortened",
        "source_trace_run": payload.get("_source_trace_run"),
        "source_trace_row_id": payload["id"],
        "shortener_model": SHORTENER_ID,
        "shortening_policy": shortening_policy(min_tokens, max_tokens),
    }
    return row


def long_reasoning_payload(payload: dict[str, Any]) -> dict[str, Any]:
    _source_reasoning, final_answer = extract_target(payload)
    if final_answer != payload["answer"]:
        raise ValueError(f"{payload['id']}: target answer does not match canonical answer")
    row = clean_payload(payload)
    row["canonical_answer"] = payload["answer"]
    row["supervision"] = {
        **source_supervision(payload),
        "target_text_format": TARGET_FORMAT,
        "target_text_source": "openrouter_native_reasoning_long",
        "source_trace_run": payload.get("_source_trace_run"),
        "source_trace_row_id": payload["id"],
        "truncation_policy": "measure_and_report_before_medium_or_full_training",
    }
    return row


def extract_target(payload: dict[str, Any]) -> tuple[str, str]:
    match = THINK_RE.fullmatch(str(payload.get("target_text", "")).strip())
    if match is None:
        raise ValueError(f"{payload.get('id')}: target_text is not <think>...</think><answer>...</answer>")
    return match.group("think").strip(), match.group("answer").strip()


def clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def source_supervision(payload: dict[str, Any]) -> dict[str, Any]:
    supervision = payload.get("supervision")
    return dict(supervision) if isinstance(supervision, dict) else {}


def shorten_reasoning(
    reasoning: str,
    *,
    task_type: str,
    min_tokens: int = 128,
    max_tokens: int = 256,
) -> str:
    candidates = candidate_segments(reasoning)
    selected: list[str] = []
    bucket_limits = {"shape": 3, "transform": 4, "decision": 8}
    for bucket in ("shape", "transform", "decision"):
        added = 0
        for segment in ranked_segments(candidates, task_type=task_type, bucket=bucket):
            add_segment(selected, segment, max_tokens=max_tokens)
            if segment in selected:
                added += 1
            if added >= bucket_limits[bucket]:
                break
            if token_count(" ".join(selected)) >= min_tokens and bucket == "decision":
                break
        if token_count(" ".join(selected)) >= min_tokens and any(decision_score(s, task_type) > 0 for s in selected):
            break
    if token_count(" ".join(selected)) < min_tokens:
        for segment in ranked_segments(candidates, task_type=task_type, bucket="all"):
            add_segment(selected, segment, max_tokens=max_tokens)
            if token_count(" ".join(selected)) >= min_tokens:
                break
    if not selected:
        raise ValueError("could not extract any reasoning segments")
    return clamp_to_token_window(" ".join(selected), min_tokens=min_tokens, max_tokens=max_tokens)


def candidate_segments(reasoning: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", reasoning.replace("\r", "\n")).strip()
    rough = re.split(r"(?<=[.!?])\s+|(?:\s+-\s+)|(?:\s+\*\*)", normalized)
    segments: list[str] = []
    for item in rough:
        segment = clean_segment(item)
        words = segment.split()
        if len(words) < 6 or len(words) > 70:
            continue
        lower = segment.lower()
        if any(
            marker in lower
            for marker in (
                "wait,",
                "wait ",
                "double check",
                "final check",
                "one more check",
                "i need",
                "i should",
                "i am",
                "i'm",
                "i must",
                "the user wants",
                "is there any",
                "could ",
                "might ",
                "maybe",
                "assume",
                "suspicious",
            )
        ):
            continue
        if "?" in segment:
            continue
        if "<answer>" in lower or "answer:" in lower:
            continue
        if segment not in segments:
            segments.append(segment)
    return segments


def clean_segment(segment: str) -> str:
    segment = re.sub(r"^[\s>*#`_:-]+", "", segment)
    segment = re.sub(r"^\d+\.\s*", "", segment)
    segment = segment.replace("`", "")
    segment = re.sub(r"\*\*", "", segment)
    return segment.strip(" -:\n\t")


def ranked_segments(candidates: list[str], *, task_type: str, bucket: str) -> list[str]:
    def score(segment: str) -> tuple[int, int]:
        if bucket == "shape":
            primary = shape_score(segment)
        elif bucket == "transform":
            primary = transform_score(segment)
        elif bucket == "decision":
            primary = decision_score(segment, task_type)
        else:
            primary = shape_score(segment) + transform_score(segment) + decision_score(segment, task_type)
        return (primary, -len(segment.split()))

    return [segment for segment in sorted(candidates, key=score, reverse=True) if score(segment)[0] > 0]


def shape_score(segment: str) -> int:
    lower = segment.lower()
    terms = (
        "object_a",
        "object a",
        "object_b",
        "object b",
        "box",
        "cylinder",
        "tube",
        "hollow",
        "sphere",
        "plate",
        "hole",
        "cut",
        "slot",
        "fillet",
        "extent",
        "range",
        "face",
        "radius",
        "dimensions",
    )
    return sum(1 for term in terms if term in lower)


def transform_score(segment: str) -> int:
    lower = segment.lower()
    terms = (
        "transform",
        "translation",
        "rotation",
        "shift",
        "moved",
        "global",
        "center",
        "x range",
        "y range",
        "z range",
        "extent",
        "gap",
        "separated",
        "touch",
    )
    return sum(1 for term in terms if term in lower)


def decision_score(segment: str, task_type: str) -> int:
    lower = segment.lower()
    common = (
        "intersection",
        "overlap",
        "positive-volume",
        "volume",
        "distance",
        "clearance",
        "touching",
        "disjoint",
        "contained",
        "intersecting",
        "near_miss",
        "bucket",
        "normalized",
        "variant",
        "rank",
        "threshold",
        "required",
        "therefore",
        "so ",
        "because",
    )
    task_terms = {
        "binary_interference": ("yes", "no", "interference", "positive-volume"),
        "relation_classification": ("relation", "touching", "disjoint", "contained", "intersecting"),
        "volume_bucket": ("normalized", "bucket", "intersection volume"),
        "clearance_bucket": ("clearance", "bucket", "minimum distance"),
        "pairwise_interference": ("variant", "both variants", "interference"),
        "ranking_normalized_intersection": ("rank", "ranking", "variant", "largest", "smallest"),
        "tolerance_fit": ("required clearance", "satisfy", "fails", "threshold"),
    }
    return sum(1 for term in common + task_terms.get(task_type, ()) if term in lower)


def add_segment(selected: list[str], segment: str, *, max_tokens: int) -> None:
    if segment in selected:
        return
    current = " ".join(selected)
    if token_count(current) >= max_tokens:
        return
    if token_count(f"{current} {segment}") <= max_tokens:
        selected.append(segment)


def clamp_to_token_window(text: str, *, min_tokens: int, max_tokens: int) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    clipped = words[:max_tokens]
    while clipped and not re.search(r"[.!?]$", clipped[-1]):
        clipped.pop()
    if len(clipped) < min_tokens:
        clipped = words[:max_tokens]
    return " ".join(clipped).strip()


def token_count(text: str) -> int:
    return len(text.split())


def write_dataset(output_dir: Path, rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        write_jsonl(rows, output_dir / f"{split}.jsonl")


def write_public_grpo_views(output_dir: Path, rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    public_rows = {
        "grpo_train": [public_row_json(row) for row in rows_by_split["train"]],
        "grpo_validation": [public_row_json(row) for row in rows_by_split["validation"]],
    }
    for split, rows in public_rows.items():
        write_jsonl(rows, output_dir / f"{split}.jsonl")


def public_row_json(payload: dict[str, Any]) -> dict[str, Any]:
    return public_row_from_payload(payload).model_dump(mode="json")


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def audit_sample(rows: list[dict[str, Any]], *, per_task: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_type"]), []).append(row)
    samples: list[dict[str, Any]] = []
    for task_type in sorted(by_task):
        for row in by_task[task_type][:per_task]:
            samples.append(
                {
                    "id": row["id"],
                    "task_type": row["task_type"],
                    "answer": row["answer"],
                    "source_trace_run": row["supervision"].get("source_trace_run"),
                    "target_text": row["target_text"],
                    "audit_status": "",
                    "audit_note": "",
                }
            )
    return samples


def manual_audit_template(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Manual Short-Trace Proof Audit",
        "",
        "Allowed statuses: valid, fallacious, too vague, answer_mismatch, leaks_label_without_reasoning.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['id']} ({row['task_type']})",
                "",
                f"- answer: `{row['answer']}`",
                "- status: ",
                "- note: ",
                "",
                row["target_text"],
                "",
            ]
        )
    return "\n".join(lines)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "task_counts": dict(sorted(Counter(row["task_type"] for row in rows).items())),
        "source_trace_runs": dict(sorted(Counter(row.get("_source_trace_run") for row in rows).items())),
    }


def summarize_dataset(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        split: {
            "row_count": len(rows),
            "task_counts": dict(sorted(Counter(row["task_type"] for row in rows).items())),
            "reasoning_token_lengths": length_summary([target_reasoning_tokens(row) for row in rows]),
            "target_char_lengths": length_summary([len(str(row["target_text"])) for row in rows]),
        }
        for split, rows in rows_by_split.items()
    }


def target_reasoning_tokens(row: dict[str, Any]) -> int:
    reasoning, _answer = extract_target(row)
    return token_count(reasoning)


def length_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def shortening_policy(min_tokens: int, max_tokens: int) -> str:
    return (
        f"Select source reasoning sentences about object features, transforms, and task decision; "
        f"drop self-check/filler; keep {min_tokens}-{max_tokens} whitespace tokens before the answer tag."
    )


if __name__ == "__main__":
    main()
