"""Dataset card generation for exported IntersectionQA datasets."""

from __future__ import annotations

import json
from pathlib import Path

from intersectionqa.export.jsonl import read_metadata, read_source_manifest
from intersectionqa.schema import DatasetMetadata


TASK_DESCRIPTIONS = {
    "binary_interference": (
        "Closed-book yes/no prediction of whether two transformed CAD solids have "
        "positive-volume material overlap. Touching faces, edges, or points are `no`."
    ),
    "relation_classification": (
        "Multiclass relation prediction: `disjoint`, `touching`, `near_miss`, "
        "`intersecting`, `contained`, or `invalid`."
    ),
    "volume_bucket": (
        "Bucket prediction for normalized intersection volume, defined as "
        "`intersection_volume / min(volume_a, volume_b)`."
    ),
    "clearance_bucket": (
        "Bucket prediction for exact minimum solid-to-solid clearance, with explicit "
        "`touching` and `intersecting` cases."
    ),
    "tolerance_fit": (
        "Prediction of whether the exact clearance satisfies a required tolerance."
    ),
    "pairwise_interference": (
        "Counterfactual comparison of two related assemblies: choose which variant has "
        "positive-volume interference."
    ),
    "ranking_normalized_intersection": (
        "Counterfactual ranking of several assembly variants from highest to lowest "
        "normalized intersection volume."
    ),
    "repair_direction": (
        "Reserved task family for predicting a simple transform direction that would "
        "remove or reduce interference."
    ),
}

FIELD_DESCRIPTIONS = [
    ("`id`", "Stable public row identifier."),
    ("`split`", "Dataset split name."),
    ("`task_type`", "One of the task families listed above."),
    (
        "`prompt`",
        "The closed-book model input. It contains CadQuery code, transforms, task instructions, and answer format.",
    ),
    ("`answer`", "Canonical target string for the task."),
    ("`script`", "Executable reconstruction script for the two placed objects."),
    ("`source`", "Source corpus. Current public releases are CADEvolve-derived."),
    ("`generator_id`", "Best available source family or generator grouping identifier."),
    ("`base_object_pair_id`", "Stable identifier for the object pair before transform variants."),
    ("`assembly_group_id`", "Grouping identifier for related assemblies and split leakage checks."),
    (
        "`counterfactual_group_id`",
        "Identifier shared by variants that differ by one controlled parameter, when present.",
    ),
    (
        "`changed_parameter` / `changed_value`",
        "The controlled transform or parameter change for the row, when applicable.",
    ),
    (
        "`labels`",
        "Precomputed geometry labels: volumes, intersection volume, normalized intersection, minimum distance, relation, and containment flags.",
    ),
    (
        "`diagnostics`",
        "Exact/bounding-box diagnostics and label status. These are for analysis, not prompt input.",
    ),
    (
        "`difficulty_tags`",
        "Diagnostic tags such as `near_boundary`, `tiny_overlap`, `cavity_targeted`, or `aabb_exact_disagreement`.",
    ),
    ("`label_policy`", "Thresholds used to derive labels from raw geometry values."),
    ("`hashes`", "Content hashes for source code, objects, transforms, geometry, config, and prompt."),
    (
        "`metadata`",
        "Prompt template version, split group, source subtrees, generator IDs, artifact IDs, and task-specific metadata.",
    ),
]


def build_dataset_card(dataset_dir: Path) -> str:
    metadata = read_metadata(dataset_dir / "metadata.json")
    if metadata is None:
        raise ValueError(f"{dataset_dir}: missing metadata.json")
    source_manifest = read_source_manifest(dataset_dir / "source_manifest.json")
    release_name = _release_name(dataset_dir, metadata)

    source_lines = []
    if source_manifest is not None:
        for source in source_manifest.sources:
            record_count = source.source_records_loaded or source.fixture_count or 0
            source_lines.append(
                f"- `{source.source}`: {record_count:,} records, "
                f"purpose `{source.purpose or source.execution_policy or 'source corpus'}`."
            )
    if not source_lines:
        source_lines.append("- Source manifest unavailable.")

    split_rows = [
        "| Split | Rows | Holdout rule | Task counts |",
        "| --- | ---: | --- | --- |",
        *[
            f"| `{name}` | {summary.row_count:,} | `{summary.holdout_rule}` | "
            f"{_format_counts(summary.task_counts)} |"
            for name, summary in sorted(metadata.splits.items())
        ],
    ]
    task_rows = [
        "| Task | Rows | What it asks |",
        "| --- | ---: | --- |",
        *[
            f"| `{task_type}` | {metadata.counts.by_task.get(task_type, 0):,} | "
            f"{TASK_DESCRIPTIONS.get(task_type, 'Task family reserved by the schema.')} |"
            for task_type in metadata.task_types
        ],
    ]
    relation_rows = [
        "| Relation | Rows | Meaning |",
        "| --- | ---: | --- |",
        *[
            f"| `{relation}` | {count:,} | {_relation_meaning(relation)} |"
            for relation, count in sorted(metadata.counts.by_relation.items())
        ],
    ]
    field_rows = [
        "| Field | Description |",
        "| --- | --- |",
        *[f"| {field} | {description} |" for field, description in FIELD_DESCRIPTIONS],
    ]
    limitation_lines = [
        f"- {item}" for item in metadata.known_limitations
    ] or ["- No known limitations recorded in `metadata.json`."]
    yaml_lines = [
        "---",
        f"license: {metadata.license}",
        "language:",
        "- en",
        "task_categories:",
        "- question-answering",
        "- text-classification",
        "annotations_creators:",
        "- machine-generated",
        "source_datasets:",
        "- original",
        "size_categories:",
        f"- {_size_category(metadata.counts.total_rows)}",
        "tags:",
        "- cad",
        "- cadquery",
        "- computational-geometry",
        "- spatial-reasoning",
        "- code",
        "- benchmark",
        f"pretty_name: {release_name}",
        "configs:",
        "- config_name: default",
        "  data_files:",
        *_data_file_config(dataset_dir, metadata),
        "---",
        "",
    ]
    sibling_note = (
        "IntersectionQA is published in two public sizes: `IntersectionQA-15K` for "
        "quick inspection and smoke experiments, and `IntersectionQA-90K` for the "
        "larger benchmark/training release."
    )

    return "\n".join(
        [
            *yaml_lines,
            f"# {release_name}",
            "",
            "## Dataset Summary",
            "IntersectionQA is a code-only CAD spatial-reasoning benchmark. Each example gives a "
            "model two executable CadQuery object-construction functions plus assembly transforms, "
            "then asks it to infer the geometric relation induced by that code. The central question "
            "is whether a code model can mentally track the spatial consequences of CAD programs: "
            "positive-volume interference, contact, near misses, clearance, containment, and overlap "
            "magnitude.",
            "",
            "The dataset is not intended to replace a CAD kernel. Official labels are produced "
            "offline with CadQuery/OpenCASCADE Boolean and distance queries, then stored in the rows "
            "so training and evaluation can run without executing CAD code. The benchmark measures "
            "closed-book geometric grounding from code; tool-assisted CAD agents can be evaluated "
            "separately as an upper-bound setting.",
            "",
            "This is a benchmark-first release rather than a conventional train-heavy supervised "
            "dataset. Interpret `train`, `validation`, and `test_random` as the simple "
            "train/dev/test path. The other `test_*` splits are named diagnostic challenge suites. "
            "In particular, `test_near_boundary` is intentionally large because it is the main "
            "stress set for touching, near-miss, tiny-overlap, and counterfactual cases; it should "
            "be reported separately from the primary random test score.",
            "",
            sibling_note,
            "",
            f"This repository contains `{metadata.counts.total_rows:,}` public task rows from "
            f"IntersectionQA `{metadata.dataset_version}`.",
            "",
            "## Motivation",
            "Text-to-CAD and CAD-code systems are often evaluated by final shape similarity or by "
            "whether generated code executes. IntersectionQA isolates a narrower capability that is "
            "important for CAD agents: given existing CAD code and placement transforms, can the "
            "model reason about the actual assembly geometry before running a verifier?",
            "",
            "The benchmark is built around controlled geometry questions rather than free-form design "
            "quality. This makes it useful for comparing closed-book model predictions, fine-tuned "
            "code models, shortcut baselines such as AABB overlap, and tool-assisted CAD-kernel agents.",
            "",
            "## Task Descriptions",
            *task_rows,
            "",
            "## Label Definitions",
            "- `interference` means positive-volume material overlap. Merely touching at a face, edge, or point is not interference.",
            "- `binary_interference` answers `yes` for `intersecting` and `contained`; it answers `no` for `disjoint`, `touching`, and `near_miss`.",
            "- `relation_classification` uses the precedence `invalid`, `contained`, `intersecting`, `touching`, `near_miss`, then `disjoint`.",
            "- `volume_bucket` uses `intersection_volume / min(volume_a, volume_b)` with buckets `0`, `(0, 0.01]`, `(0.01, 0.05]`, `(0.05, 0.20]`, `(0.20, 0.50]`, and `>0.50`.",
            "- `clearance_bucket` uses exact minimum distance with buckets `touching`, `(0, 0.1]`, `(0.1, 1]`, `(1, 5]`, `>5`, plus `intersecting` for positive-overlap rows.",
            "- `tolerance_fit` is `yes` only when the stored exact clearance satisfies the row's required threshold.",
            "- Counterfactual pairwise and ranking rows derive their answers from exact geometry metadata for the compared variants.",
            "",
            "## Relation Distribution",
            *relation_rows,
            "",
            "## Data Sources",
            *source_lines,
            "",
            "CADEvolve is the primary source corpus because it provides executable CadQuery programs "
            "at scale. For IntersectionQA, CADEvolve programs are normalized into `object_a()` and "
            "`object_b()` functions and paired with deterministic transforms. Synthetic primitives are "
            "reserved for golden tests and smoke/debug fixtures; the public 15K and 90K releases report "
            "their actual source counts in `metadata.json`.",
            "",
            "## Generation Process",
            "The release pipeline follows this high-level process:",
            "",
            "1. Load executable CADEvolve CadQuery programs from a bounded, deterministic source subset.",
            "2. Execute each source program in isolated workers with timeouts and validate that it produces finite, non-degenerate solid geometry.",
            "3. Normalize accepted programs into self-contained object functions.",
            "4. Propose two-object assemblies with deterministic object pairing and bbox-guided, contact-targeted, near-boundary, cavity-targeted, and counterfactual transform strategies.",
            "5. Label accepted assemblies using exact CadQuery/OpenCASCADE Boolean intersection volume and solid-to-solid distance queries.",
            "6. Materialize multiple task rows from the same geometry record while keeping labels, diagnostics, thresholds, provenance, and split groups outside the prompt text.",
            "",
            "Public prompts deliberately omit official labels, raw geometry values, diagnostics, and label-policy thresholds. Those fields remain in the row for evaluation and dataset slicing.",
            "",
            "## Splits",
            *split_rows,
            "",
            "Splits are designed for different evaluation questions. `train` and `validation` support "
            "model development. `test_random` is a sanity-check test split. `test_object_pair_heldout` "
            "keeps object-pair/assembly groups out of training. `test_near_boundary` emphasizes "
            "touching, tiny-overlap, near-miss, counterfactual, and other hard geometry cases. "
            "`test_generator_heldout` is reserved for generator-family holdout and may be empty when "
            "the current source window does not provide a reliable generator-family split.",
            "",
            "Because this release deliberately oversamples hard boundary cases, `test_near_boundary` "
            "can be larger than `train`. Treat it as a diagnostic benchmark pool, not as a small "
            "held-out test split from an IID training distribution.",
            "",
            "Rows from the same object-pair, assembly, and counterfactual groups are kept split-safe "
            "according to the exported group metadata.",
            "",
            "## Data Fields",
            *field_rows,
            "",
            "## Label Policy",
            f"- `epsilon_volume_ratio`: `{metadata.label_policy.epsilon_volume_ratio}`",
            f"- `epsilon_distance_mm`: `{metadata.label_policy.epsilon_distance_mm}`",
            f"- `near_miss_threshold_mm`: `{metadata.label_policy.near_miss_threshold_mm}`",
            "",
            "The policy-positive overlap threshold is record-dependent: "
            "`epsilon_volume = epsilon_volume_ratio * min(volume_a, volume_b)`. Changing these "
            "thresholds changes derived labels and should be treated as a new dataset version or a "
            "formal relabeling audit.",
            "",
            "## How to Use",
            "Load a split with the Hugging Face `datasets` library:",
            "",
            "```python",
            "from datasets import load_dataset",
            "",
            f'dataset = load_dataset("{release_name}", split="train")',
            "row = dataset[0]",
            "prompt = row[\"prompt\"]",
            "answer = row[\"answer\"]",
            "```",
            "",
            "For closed-book evaluation, provide only `prompt` to the model and compare the parsed "
            "model output with `answer`. Do not include `labels`, `diagnostics`, or `metadata` in the "
            "model input unless you are intentionally running an analysis or tool-assisted setting.",
            "",
            "## Intended Uses",
            "- Benchmark closed-book CAD-code spatial reasoning.",
            "- Fine-tune or evaluate code models on answer-only geometry QA.",
            "- Analyze model behavior on contact-vs-interference, near-boundary, containment, and AABB-failing cases.",
            "- Compare model predictions with shortcut baselines and tool-assisted CAD-kernel upper bounds.",
            "- Build diagnostic slices using `difficulty_tags`, relation labels, source subtrees, and split groups.",
            "",
            "## Out-of-Scope Uses",
            "- Treating labels as legal, manufacturing, safety, or tolerance certification.",
            "- Treating closed-book model accuracy as a substitute for CAD verification in production workflows.",
            "- Training models to execute untrusted CAD code outside a sandbox.",
            "- Using prompt-visible labels, diagnostics, or geometry values during closed-book evaluation.",
            "- Claiming any bounded public release represents the full distribution of all possible CAD assemblies.",
            "",
            "## Known Limitations",
            *limitation_lines,
            "- The paper is not yet published; this card describes the public dataset and benchmark intent, and a formal citation will be added when available.",
            "- Current rows are code-only two-object assemblies. Rendered images, STEP/mesh artifacts, multi-part repair tasks, and full CAD-agent workflows are outside this default public JSONL release.",
            "- Labels depend on the stored CadQuery/OpenCASCADE geometry computations and threshold policy. Small kernel or tolerance changes can matter near decision boundaries.",
            "- `test_generator_heldout` can be empty in the current exports; use the non-empty held-out and near-boundary splits for reported evaluation unless a later release fills that split.",
            "- CADEvolve-derived examples may share broad source-family patterns even when object-pair and assembly-group leakage controls are applied. Use hashes and group fields for additional audits.",
            "- AABB overlap is included only as a diagnostic shortcut baseline. It is not an official label source.",
            "",
            "## License",
            f"The IntersectionQA release license is `{metadata.license}`. Source-specific provenance is "
            "stored in the source manifest and row metadata. CADEvolve source programs are treated as "
            "untrusted Python during generation and are executed only in isolated generation workers.",
            "",
            "## Reproducibility",
            f"- Dataset version: `{metadata.dataset_version}`",
            f"- Created from commit: `{metadata.created_from_commit}`",
            f"- Config hash: `{metadata.config_hash}`",
            f"- Source manifest hash: `{metadata.source_manifest_hash}`",
            f"- Total rows: `{metadata.counts.total_rows:,}`",
            f"- Source counts: {_format_counts(metadata.counts.by_source)}",
            f"- CadQuery version recorded by export: `{_optional_value(metadata.cadquery_version)}`",
            f"- OCP version recorded by export: `{_optional_value(metadata.ocp_version)}`",
            "",
            "The public rows are self-contained for ordinary training and evaluation: official answers, "
            "geometry labels, diagnostics, provenance, split metadata, and hashes are present in the "
            "JSONL/Parquet rows. CadQuery execution is not required to load the dataset or score model "
            "predictions.",
            "",
            "## Citation",
            "A paper citation will be added after publication. Until then, cite the dataset release "
            "and include the dataset name, version, and Hugging Face repository used.",
            "",
        ]
    )


def write_dataset_card(dataset_dir: Path, path: Path | None = None) -> Path:
    output_path = path or dataset_dir / "DATASET_CARD.md"
    output_path.write_text(build_dataset_card(dataset_dir), encoding="utf-8")
    return output_path


def _release_name(dataset_dir: Path, metadata: DatasetMetadata) -> str:
    name = dataset_dir.name
    if name.startswith("IntersectionQA"):
        return name
    parent_name = dataset_dir.parent.name
    if parent_name.startswith("IntersectionQA"):
        return parent_name
    return metadata.dataset_name


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "{}"
    return "`" + json.dumps(dict(sorted(counts.items())), sort_keys=True) + "`"


def _data_file_config(dataset_dir: Path, metadata: DatasetMetadata) -> list[str]:
    parquet_paths = {
        split_name: dataset_dir / "parquet" / f"{split_name}.parquet"
        for split_name in metadata.splits
    }
    if all(path.exists() for path in parquet_paths.values()):
        paths = {split_name: f"parquet/{split_name}.parquet" for split_name in metadata.splits}
    else:
        paths = {split_name: summary.path for split_name, summary in metadata.splits.items()}

    lines: list[str] = []
    for split_name in metadata.splits:
        lines.extend(
            [
                f"  - split: {split_name}",
                f"    path: {paths[split_name]}",
            ]
        )
    return lines


def _relation_meaning(relation: str) -> str:
    meanings = {
        "disjoint": "No positive overlap, no contact, and clearance greater than the near-miss threshold.",
        "touching": "Zero policy-positive overlap and contact within distance tolerance.",
        "near_miss": "Small positive clearance above touching tolerance and at or below the near-miss threshold.",
        "intersecting": "Positive-volume material overlap without containment precedence.",
        "contained": "One solid is fully inside the other with positive-volume overlap.",
        "invalid": "Geometry or label computation failed; excluded from normal task rows unless explicitly included.",
    }
    return meanings.get(relation, "Relation label recorded by the dataset.")


def _size_category(rows: int) -> str:
    if rows < 1_000:
        return "n<1K"
    if rows < 10_000:
        return "1K<n<10K"
    if rows < 100_000:
        return "10K<n<100K"
    if rows < 1_000_000:
        return "100K<n<1M"
    return "n>1M"


def _optional_value(value: str | None) -> str:
    return value if value is not None else "not recorded"
