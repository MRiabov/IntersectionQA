# IntersectionQA

IntersectionQA is a benchmark and dataset plan for CAD-code spatial reasoning over
CadQuery objects and assembly transforms. The released v0.1 target is a
code-only, two-object benchmark derived primarily from CADEvolve programs, with
small synthetic fixtures used only for golden tests, smoke/debug cases, and
local fallback.

## Canonical Specification Map

Use these documents in `specs/` as the source of truth:

| Document | Authority |
| --- | --- |
| `specs/benchmark-task-spec.md` | Task semantics, prompt contracts, answer formats, parser behavior, and MVP task scope. |
| `specs/label_rules.md` | Official geometry-label precedence, thresholds, bucket boundaries, failure reasons, and golden cases. |
| `specs/schema.md` | Canonical internal records, public JSONL row schema, metadata files, manifests, hashes, IDs, and validation requirements. |
| `specs/generation_policy.md` | Candidate generation, CADEvolve source policy, sampling, balancing, split grouping, diagnostics, and anti-patterns. |
| `specs/using-cadevolve-dataset-export.md` | Practical CADEvolve archive usage notes and source-layout details. |
| `specs/useful-implementation-optimization.md` | Implementation and scaling guidance; informative unless it conflicts with the canonical specs above. |
| `specs/reviewer-readiness-checklist.md` | Operational checklist for validating a dataset build before review or release. |
| `specs/paper-spec.md` | Research framing and paper ambition; informative unless it conflicts with the canonical specs above. |
| `epics-and-stories.md` and `epics-and-stories.yaml` | Planning backlog generated from the specs and implementation matrix. |
| `implementation-complexity-priority-matrix.md` | Build/defer decisions, suggested module map, and risk-ranked implementation order. |

If two documents disagree, reconcile in this order:

1. `specs/label_rules.md` controls labels, thresholds, relation precedence, and buckets.
2. `specs/schema.md` controls record fields, enum values, metadata, hashes, IDs, and validation.
3. `specs/benchmark-task-spec.md` controls task-facing semantics and answer parsing.
4. `specs/generation_policy.md` controls what examples should be generated and accepted.
5. Planning and paper documents should be updated to match the canonical specs.

## v0.1 MVP Scope

The first implementation target is intentionally narrow:

- CADEvolve-derived two-object assemblies.
- Closed-book code-only prompts.
- MVP tasks:
  - `binary_interference`
  - `relation_classification`
  - `volume_bucket`
- Group-safe splits including random, generator-held-out,
  object-pair/assembly-held-out, and near-boundary hard subsets.
- Precomputed labels and diagnostics in JSONL rows; no CadQuery execution needed
  to load answers or evaluate model outputs.

The schema reserves later task types such as `clearance_bucket`,
`pairwise_interference`, `ranking_normalized_intersection`, `repair_direction`,
and `tolerance_fit`, but those are not first-class v0.1 MVP tasks.

## Release-Candidate Builds

Dataset generation writes public split JSONL and compressed Parquet files under
`parquet/`. Use the release-candidate builder when preparing a sizeable dataset
run because it validates the export and writes stats, AABB baseline, failure
analysis, and comparison-table reports in one place:

```bash
CACHE_ROOT=$(find .cache/intersectionqa/cadevolve_sources -name extraction_manifest.json -printf '%h\n' | sort | head -1)
rtk uv run python -m scripts.build_release_candidate \
  --config configs/smoke.yaml \
  --cadevolve-source-cache-root "$CACHE_ROOT" \
  --output-dir /tmp/intersectionqa_rc
```

For source-window sharding:

```bash
CACHE_ROOT=$(find .cache/intersectionqa/cadevolve_sources -name extraction_manifest.json -printf '%h\n' | sort | head -1)
rtk uv run python -m scripts.build_release_candidate \
  --config configs/smoke.yaml \
  --cadevolve-source-cache-root "$CACHE_ROOT" \
  --output-dir /tmp/intersectionqa_rc_sharded \
  --shard-count 10 \
  --source-shard-size 250
```

## CADEvolve Source Cache

Keep `data/cadevolve.tar` as the canonical upstream artifact, but do not use the
tar archive as the repeated local hot path. Smoke/full generation now materializes
the configured executable CADEvolve source subset into:

```text
.cache/intersectionqa/cadevolve_sources/
```

The cache is keyed by the archive fingerprint and preserves each original archive
member path in the generated provenance. Local cache paths are not required by
public rows and are not part of the dataset release.

Once the source cache is prepared, generation does not need the tar on the hot
path. Pass `--cadevolve-source-cache-root` to use a specific prepared cache
root; default generation does not auto-discover local caches because that would
make tests and runs depend on machine-local artifacts.

For space, extract only the configured subset or shards needed for the intended
run. The current v0.1 target should stay bounded; generating more than roughly
100k public rows is out of scope until storage and runtime are re-evaluated.

To prewarm a local source prefix explicitly:

```bash
rtk uv run python -m scripts.prepare_cadevolve_sources \
  --cadevolve-archive data/cadevolve.tar \
  --limit 100000
```

This extracts the first `100000` sorted executable CADEvolve source programs,
not `100000` final public task rows. The current MVP generator can produce
multiple public task rows per accepted geometry.

## Inspecting Rows Locally

Exported datasets are split JSONL files. Use the inspect utility for prompt and
label review:

```bash
rtk uv run python -m scripts.inspect_example /tmp/intersectionqa_smoke_cadevolve intersectionqa_binary_000001 --show-prompt
```

For local CAD audit artifacts, export one row to a debug directory:

```bash
rtk uv run python -m scripts.export_row_artifacts \
  /tmp/intersectionqa_smoke_cadevolve \
  intersectionqa_binary_000001 \
  --output-dir /tmp/intersectionqa_row_debug
```

This writes `prompt.txt`, `row.json`, `assembly.py`, `object_a.step`,
`object_b.step`, `assembly.step`, and `intersection.step` when the row has
policy-positive overlap. These files are dev/debug artifacts and are not part
of the default public JSONL export.

To render PNG previews for a row, use the PyVista-backed renderer:

```bash
rtk uv run python -m scripts.render_row_artifacts \
  /tmp/intersectionqa_smoke_cadevolve \
  intersectionqa_binary_000001 \
  --output-dir /tmp/intersectionqa_row_debug \
  --image-size 1200x900
```

This writes `renders/object_a.png`, `renders/object_b.png`,
`renders/assembly.png`, `renders/intersection.png` when applicable, and a
`renders/contact_sheet.png` overview.

## Zero-Shot Evaluation

The zero-shot runner uses the exported public rows directly, wraps each prompt in
the versioned closed-book evaluation prompt, records decoding settings, writes
raw predictions, and reports the strict parser invalid-output rate:

```bash
rtk uv run python -m scripts.evaluate_zero_shot \
  /tmp/intersectionqa_smoke_cadevolve \
  --provider openai-chat \
  --model gpt-5.4 \
  --limit 25 \
  --requests-jsonl /tmp/intersectionqa_zero_shot_requests.jsonl
```

For open code models served through Hugging Face Inference Providers, use
`--provider huggingface-chat --model <repo-or-provider-model>`. Add
`--export-requests-only` to write the fixed request JSONL without making model
calls.

To build a compact baseline/model comparison table from saved prediction JSONL:

```bash
rtk uv run python -m scripts.baseline_comparison_table \
  /tmp/intersectionqa_smoke_cadevolve \
  --prediction gpt_5_4=/tmp/intersectionqa_zero_shot_predictions.jsonl \
  --markdown-output /tmp/intersectionqa_comparison.md
```

To summarize generation failures and optional model failure cases:

```bash
rtk uv run python -m scripts.failure_case_analysis \
  /tmp/intersectionqa_smoke_cadevolve \
  --predictions-jsonl /tmp/intersectionqa_zero_shot_predictions.jsonl \
  --output /tmp/intersectionqa_failure_analysis.json
```

## Reproducibility Checks

To compare two generated dataset directories for byte-identical release
artifacts, run:

```bash
rtk uv run python -m scripts.audit_reproducibility \
  /tmp/intersectionqa_run_a \
  /tmp/intersectionqa_run_b
```

The audit validates both directories and compares the public split JSONL files,
metadata, schema, source manifest, split manifest, object-validation manifest,
and failure manifest. It intentionally ignores `smoke_report.json`, which
contains run-local reporting such as the output directory.
