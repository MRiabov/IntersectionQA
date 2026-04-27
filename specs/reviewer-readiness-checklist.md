# IntersectionQA v0.1 Reviewer-Readiness Checklist

Use this checklist before sharing a dataset build, paper draft, or review
artifact. It is intentionally operational: every item should either have a
passing command, a generated artifact path, or a recorded exception.

## Dataset Build

- Generate the intended dataset from the checked-in config and record the exact command.
- Confirm CADEvolve source provenance is preserved through archive member paths, not local cache paths.
- Confirm synthetic fixtures are used only for golden, smoke, debug, or fallback rows.
- Validate the exported dataset directory:

```bash
rtk uv run python -m scripts.dataset.validate_dataset <dataset_dir>
```

- Confirm Parquet release files exist under `<dataset_dir>/parquet/` and match
  the public split row counts in `parquet_manifest.json`.
- Record `smoke_report.json` or the full generation summary, including row counts, task counts, relation counts, split counts, source counts, and failure counts.

## Reproducibility

- Run the full test suite:

```bash
rtk uv run pytest -q
```

- Compile Python modules:

```bash
rtk uv run python -m compileall -q intersectionqa scripts
```

- Compare two independently generated dataset directories:

```bash
rtk uv run python -m scripts.dataset.audit_reproducibility <run_a> <run_b>
```

- Confirm local-only cache paths and `data/cadevolve.tar` are not included in release artifacts.

## Label Quality

- Check leakage audit status in the generation report.
- Inspect failure manifests for repeated failure modes that would bias the dataset.
- Render and manually inspect a small sample covering:
  - positive overlap
  - disjoint
  - touching or near-miss
  - boundary/tiny-overlap candidates

```bash
rtk uv run python -m scripts.dataset.export_row_artifacts <dataset_dir> <row_id> --output-dir <debug_dir>
rtk uv run python -m scripts.dataset.render_row_artifacts <dataset_dir> <row_id> --output-dir <debug_dir>
```

## Evaluation

- Run the AABB baseline:

```bash
rtk uv run python -m scripts.evaluation.evaluate_baseline <dataset_dir>
```

- For saved model predictions, evaluate strict exact-answer metrics:

```bash
rtk uv run python -m scripts.evaluation.evaluate_predictions <dataset_dir> <predictions.jsonl>
```

- Build a comparison table for paper/reporting:

```bash
rtk uv run python -m scripts.evaluation.baseline_comparison_table \
  <dataset_dir> \
  --prediction <system>=<predictions.jsonl> \
  --markdown-output <comparison.md>
```

- Summarize generation and prediction failure cases:

```bash
rtk uv run python -m scripts.evaluation.failure_case_analysis \
  <dataset_dir> \
  --predictions-jsonl <predictions.jsonl> \
  --output <failure_analysis.json>
```

## Release Notes

- State dataset version, config hash, label policy, task families, and split policy.
- State that public rows are closed-book code-only prompts and do not require CadQuery execution to evaluate.
- State known limitations: no rendering/multimodal prompts in v0.1, no full-scale OBB/convex-hull baseline by default, and relation balancing is target-based rather than guaranteed.
- Include CADEvolve licensing/provenance and clarify that local caches are not release dependencies.
