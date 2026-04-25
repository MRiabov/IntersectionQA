# IntersectionQA

IntersectionQA is a benchmark and dataset plan for CAD-code spatial reasoning over
CadQuery objects and assembly transforms. The released v0.1 target is a
code-only, two-object benchmark derived primarily from CADEvolve programs, with
small synthetic fixtures used only for golden tests, smoke/debug cases, and
local fallback.

This repository is currently IntersectionQA-first, but is evolving into shared
benchmark tooling for IntersectionQA and IntersectionEdit task families.
IntersectionQA covers diagnostic and question-answering tasks over CAD
intersections. IntersectionEdit covers edit, repair, and action tasks over the
same geometry and verifier infrastructure; the first implemented slice is a
conservative `repair_direction` and `repair_translation` tasks. The shared core is expected to own
geometry, CadQuery execution, transforms, grouping, splitting, leakage audits,
export, evaluation, and verifier/reward logic.

## Canonical Specification Map

Use these documents in `specs/` as the source of truth:

| Document | Authority |
| --- | --- |
| `specs/benchmark-family.md` | Umbrella direction for the IntersectionQA-first shared benchmark codebase and future IntersectionEdit sibling task family. |
| `specs/benchmark-task-spec.md` | Task semantics, prompt contracts, answer formats, parser behavior, and MVP task scope. |
| `specs/label_rules.md` | Official geometry-label precedence, thresholds, bucket boundaries, failure reasons, and golden cases. |
| `specs/schema.md` | Canonical internal records, public JSONL row schema, metadata files, manifests, hashes, IDs, and validation requirements. |
| `specs/generation_policy.md` | Candidate generation, CADEvolve source policy, sampling, balancing, split grouping, diagnostics, and anti-patterns. |
| `specs/intersectionedit-task-spec.md` | Draft specification for IntersectionEdit edit/repair/action tasks and the first conservative repair slices; not part of the frozen v0.1 contract. |
| `specs/using-cadevolve-dataset-export.md` | Practical CADEvolve archive usage notes and source-layout details. |
| `specs/useful-implementation-optimization.md` | Implementation and scaling guidance; informative unless it conflicts with the canonical specs above. |
| `specs/reviewer-readiness-checklist.md` | Operational checklist for validating a dataset build before review or release. |
| `specs/paper-spec.md` | Research framing and paper ambition; informative unless it conflicts with the canonical specs above. |
| `epics-and-stories.md` and `epics-and-stories.yaml` | Planning backlog generated from the specs and implementation matrix. |
| `implementation-complexity-priority-matrix.md` | Build/defer decisions, suggested module map, and risk-ranked implementation order. |
| `docs/text_finetune_runbook.md` | Current text-only dataset, HF upload, Vast/A100, Qwen3.5 SFT commands, and known training failure modes. |
| `docs/dataset_split_framing.md` | Practical interpretation of `train`, `validation`, `test_random`, and the named challenge splits. |

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

The schema supports later task types such as `clearance_bucket`,
`pairwise_interference`, `ranking_normalized_intersection`, `repair_direction`,
`repair_translation`, and `tolerance_fit`, but those are not first-class frozen
v0.1 MVP tasks.

## Release-Candidate Builds

Dataset generation writes public split JSONL and compressed Parquet files under
`parquet/`. Use the release-candidate builder when preparing a sizeable dataset
run because it validates the export and writes stats, AABB baseline, failure
analysis, and comparison-table reports in one place:

```bash
SOURCE_DIR=$(find data/cadevolve_sources -name extraction_manifest.json -printf '%h\n' | sort | head -1)
rtk uv run python -m scripts.build_release_candidate \
  --config configs/smoke.yaml \
  --cadevolve-source-dir "$SOURCE_DIR" \
  --output-dir data/intersectionqa_rc
```

For source-window sharding:

```bash
SOURCE_DIR=$(find data/cadevolve_sources -name extraction_manifest.json -printf '%h\n' | sort | head -1)
rtk uv run python -m scripts.build_release_candidate \
  --config configs/smoke.yaml \
  --cadevolve-source-dir "$SOURCE_DIR" \
  --output-dir data/intersectionqa_rc_sharded \
  --shard-count 10 \
  --source-shard-size 250
```

For the opt-in first IntersectionEdit repair slice, use the separate smoke
config. This keeps the frozen IntersectionQA v0.1 defaults unchanged while
exercising `repair_direction`/`repair_translation` export, validation, and verifier reports:

```bash
rtk uv run python -m scripts.build_release_candidate \
  --config configs/repair_smoke.yaml \
  --output-dir data/intersectionedit_repair_smoke
```

Repair release-candidate builds write `reports/edit_verifier.json` for stored
edit-row verification, fail by default if any stored repair row does not
exact-verify, write `reports/repair_verifier.json` for direction-prediction
verification, and add repair-success rows to `reports/baseline_comparison.json`
and `reports/baseline_comparison.md`. The standard
`reports/failure_analysis.json` also includes a `repair_prediction_verifier`
block when repair rows and predictions are available. `reports/dataset_stats.json`
and `scripts.dataset_stats` include repair summaries with policy,
selected-direction, and selected-magnitude counts.

The exact repair verifier executes trusted dataset row scripts to reconstruct
CadQuery geometry. Do not run verifier commands on untrusted JSONL rows without
external sandboxing.

Repair predictions can also be verifier-checked by applying the predicted move
to `object_b` and remeasuring exact geometry:

```bash
rtk uv run python -m scripts.evaluate_repair_predictions \
  data/intersectionedit_repair_smoke \
  predictions.jsonl
```

## CADEvolve Source Directory

Keep `data/cadevolve.tar` only as an upstream bootstrap artifact if you need to
create the extracted source tree. Smoke/full generation runs from the extracted
CADEvolve source directory:

```text
data/cadevolve_sources/
```

The extracted source tree preserves each original archive member path in the
generated provenance. Local filesystem paths are not required by public rows and
are not part of the dataset release.

Once the source tree is prepared, generation does not need the tar. Pass
`--cadevolve-source-dir` to use a specific extracted directory. The checked-in
local configs use `data/cadevolve_sources` and `data/cache` so generated
artifacts stay under the ignored `data/` tree.

For space, extract only the configured subset or shards needed for the intended
run. The current v0.1 target should stay bounded; generating more than roughly
100k public rows is out of scope until storage and runtime are re-evaluated.

To extract a local source prefix explicitly:

```bash
rtk uv run python -m scripts.prepare_cadevolve_sources \
  --cadevolve-archive data/cadevolve.tar \
  --limit 100000
```

This extracts the first `100000` sorted executable CADEvolve source programs
into the extracted source tree, not `100000` final public task rows. The
current MVP generator can produce multiple public task rows per accepted
geometry.

## Inspecting Rows Locally

Exported datasets are split JSONL files. Use the inspect utility for prompt and
label review:

```bash
rtk uv run python -m scripts.inspect_example data/intersectionqa_v0_1 intersectionqa_binary_000001 --show-prompt
```

For local CAD audit artifacts, export one row to a debug directory:

```bash
rtk uv run python -m scripts.export_row_artifacts \
  data/intersectionqa_v0_1 \
  intersectionqa_binary_000001 \
  --output-dir data/debug/intersectionqa_row_debug
```

This writes `prompt.txt`, `row.json`, `assembly.py`, `object_a.step`,
`object_b.step`, `assembly.step`, and `intersection.step` when the row has
policy-positive overlap. These files are dev/debug artifacts and are not part
of the default public JSONL export.

To render PNG previews for a row, use the PyVista-backed renderer:

```bash
rtk uv run python -m scripts.render_row_artifacts \
  data/intersectionqa_v0_1 \
  intersectionqa_binary_000001 \
  --output-dir data/debug/intersectionqa_row_debug \
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
  data/intersectionqa_v0_1 \
  --provider openai-chat \
  --model gpt-5.4 \
  --limit 25 \
  --requests-jsonl data/eval/intersectionqa_zero_shot_requests.jsonl
```

For open code models served through Hugging Face Inference Providers, use
`--provider huggingface-chat --model <repo-or-provider-model>`. Add
`--export-requests-only` to write the fixed request JSONL without making model
calls.

To build a compact baseline/model comparison table from saved prediction JSONL:

```bash
rtk uv run python -m scripts.baseline_comparison_table \
  data/intersectionqa_v0_1 \
  --prediction gpt_5_4=data/eval/intersectionqa_zero_shot_predictions.jsonl \
  --markdown-output data/eval/intersectionqa_comparison.md
```

To summarize generation failures and optional model failure cases:

```bash
rtk uv run python -m scripts.failure_case_analysis \
  data/intersectionqa_v0_1 \
  --predictions-jsonl data/eval/intersectionqa_zero_shot_predictions.jsonl \
  --output data/eval/intersectionqa_failure_analysis.json
```

## Reproducibility Checks

To compare two generated dataset directories for byte-identical release
artifacts, run:

```bash
rtk uv run python -m scripts.audit_reproducibility \
  data/intersectionqa_run_a \
  data/intersectionqa_run_b
```

The audit validates both directories and compares the public split JSONL files,
metadata, schema, source manifest, split manifest, object-validation manifest,
and failure manifest. It intentionally ignores `smoke_report.json`, which
contains run-local reporting such as the output directory.
