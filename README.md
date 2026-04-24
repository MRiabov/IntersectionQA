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
