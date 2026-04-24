"""Dataset card generation for exported IntersectionQA datasets."""

from __future__ import annotations

from pathlib import Path

from intersectionqa.export.jsonl import read_metadata, read_source_manifest


def build_dataset_card(dataset_dir: Path) -> str:
    metadata = read_metadata(dataset_dir / "metadata.json")
    if metadata is None:
        raise ValueError(f"{dataset_dir}: missing metadata.json")
    source_manifest = read_source_manifest(dataset_dir / "source_manifest.json")
    source_lines = []
    if source_manifest is not None:
        for source in source_manifest.sources:
            source_lines.append(
                f"- `{source.source}`: {source.source_records_loaded or source.fixture_count or 0} records, "
                f"purpose `{source.purpose or source.execution_policy or 'source corpus'}`."
            )
    if not source_lines:
        source_lines.append("- Source manifest unavailable.")

    split_lines = [
        f"- `{name}`: {summary.row_count} rows; tasks {summary.task_counts}; rule `{summary.holdout_rule}`."
        for name, summary in sorted(metadata.splits.items())
    ]
    task_lines = [f"- `{task_type}`" for task_type in metadata.task_types]
    limitation_lines = [f"- {item}" for item in metadata.known_limitations]
    label_policy = metadata.label_policy
    return "\n".join(
        [
            f"# {metadata.dataset_name} Dataset Card",
            "",
            "## Motivation",
            "IntersectionQA evaluates spatial reasoning over executable CadQuery assemblies. "
            "The benchmark asks models to infer exact interference, contact, clearance, and related "
            "geometry labels from code and transforms.",
            "",
            "## Task Descriptions",
            *task_lines,
            "",
            "## Label Definitions",
            "- `binary_interference`: `yes` iff exact positive-volume overlap is present.",
            "- `relation_classification`: one of disjoint, touching, near_miss, intersecting, contained, or invalid.",
            "- `volume_bucket`: bucketed normalized intersection volume.",
            "- `clearance_bucket`: bucketed exact minimum clearance or explicit intersecting/touching status.",
            "- `tolerance_fit`: `yes` iff exact clearance satisfies the stated threshold.",
            "- Counterfactual pairwise and ranking rows derive labels from exact variant geometry metadata.",
            "",
            "## Data Sources",
            *source_lines,
            "",
            "## Generation Process",
            "Objects are normalized into CadQuery functions, validated with CadQuery/OpenCASCADE, "
            "assembled with deterministic transforms, and labelled with exact CAD-kernel Boolean "
            "and distance queries. Public prompts omit labels and diagnostics.",
            "",
            "## Splits",
            *split_lines,
            "",
            "## Label Policy",
            f"- epsilon_volume_ratio: `{label_policy.epsilon_volume_ratio}`",
            f"- epsilon_distance_mm: `{label_policy.epsilon_distance_mm}`",
            f"- near_miss_threshold_mm: `{label_policy.near_miss_threshold_mm}`",
            "",
            "## Known Limitations",
            *(limitation_lines or ["- No known limitations recorded."]),
            "",
            "## Intended Uses",
            "- Benchmark closed-book CAD-code spatial reasoning.",
            "- Compare shortcut baselines, model predictions, and tool-assisted upper bounds.",
            "- Audit model behavior on near-boundary and counterfactual geometry cases.",
            "",
            "## Out-of-Scope Uses",
            "- Treating labels as legal, manufacturing, or safety certification.",
            "- Using synthetic smoke rows as a claim about full release-scale distribution.",
            "- Treating tool-assisted upper-bound results as closed-book model reasoning.",
            "",
            "## License",
            f"Dataset release license: `{metadata.license}`. Source-specific license and provenance "
            "are recorded in source metadata and manifests.",
            "",
            "## Reproducibility",
            f"- Dataset version: `{metadata.dataset_version}`",
            f"- Created from commit: `{metadata.created_from_commit}`",
            f"- Config hash: `{metadata.config_hash}`",
            f"- Source manifest hash: `{metadata.source_manifest_hash}`",
            "",
        ]
    )


def write_dataset_card(dataset_dir: Path, path: Path | None = None) -> Path:
    output_path = path or dataset_dir / "DATASET_CARD.md"
    output_path.write_text(build_dataset_card(dataset_dir), encoding="utf-8")
    return output_path
