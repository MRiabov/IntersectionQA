# IntersectionQA Generation Policy

## 1. Purpose

This file defines dataset generation policy for IntersectionQA. It specifies what examples should be proposed, accepted, rejected, balanced, tagged, grouped, and recorded before implementation starts. It is not a code architecture, module layout, or execution framework.

CADEvolve is the primary source corpus for real IntersectionQA dataset examples. The real benchmark should be built from executable CADEvolve CadQuery programs, not from a separate full synthetic CAD corpus.

Generation proposes candidate object pairs, transforms, and task rows. Exact labels come from the geometry labeling pipeline and the threshold semantics in `specs/label_rules.md`. For any candidate, exact CadQuery/OpenCASCADE-derived labels are final. Candidate strategy labels, bounding-box diagnostics, or intended perturbation class are never ground truth when they disagree with exact labels.

Prompt/task semantics must remain consistent with `specs/benchmark-task-spec.md`: code-only closed-book prompts expose object code and transforms, while official labels, diagnostics, thresholds, and failure reasons are stored outside the prompt text.

## 2. MVP Generation Scope

The first implementation target is deliberately narrow:

- Two-object assemblies only.
- Code-only closed-book prompts only.
- CADEvolve CadQuery programs as the primary object source.
- Minimal synthetic primitives only for golden tests, smoke/debug fixtures, and local fallback when CADEvolve is unavailable.
- MVP task rows derived from accepted geometry records for:
  - `binary_interference`
  - `relation_classification`
  - `volume_bucket`

MVP generation should not start by building a large standalone synthetic CAD corpus. Synthetic fixtures are allowed to make validation deterministic, but accepted released benchmark examples should come primarily from CADEvolve once CADEvolve validation works.

## 3. CADEvolve Source Policy

Use the downloaded `cadevolve.tar` archive as the CADEvolve input. Iterate archive members directly; permanent extraction is optional and must not be required for provenance.

Prefer executable CadQuery program trees:

- `CADEvolve-P/`
- `CADEvolve-C/`

Treat `CADEvolve-G/` as generator metadata, embeddings, or family context. Do not treat it as an executable object source unless a later policy explicitly defines that path.

For every source object record, store at least:

- `source: "cadevolve"`
- deterministic `source_id`
- archive member path exactly as seen in `cadevolve.tar`
- normalized archive member path with any leading `./` removed
- source split or subtree, such as `CADEvolve-P/ABC-P`, `CADEvolve-P/CADEvolve-P-core`, `CADEvolve-P/ShapeNet-P`, `CADEvolve-C/ABC-C`, `CADEvolve-C/CADEvolve-C-core`, or `CADEvolve-C/ShapeNet-C`
- generator or family ID if derivable from path, metadata, or `CADEvolve-G/parametric_generators.json`
- license metadata, expected initially to include CADEvolve's `apache-2.0` license and source dataset reference
- raw source code hash
- normalized object code hash

All CADEvolve programs are untrusted Python. They must execute only inside isolated workers with timeouts and failure reporting. They must not run in the main generation process.

The source loader must normalize common output conventions. A source program may expose its final shape as `result`, `shape`, `solid`, `part`, or another auditable CadQuery/CQ-compatible top-level value. If no usable output can be identified, reject the source object with `failure_reason: "missing_result_object"` or a more specific reason from `specs/label_rules.md`.

## 4. CADEvolve Sampling Strategy

Sampling must be deterministic from the generation config and seed.

Start with a small deterministic CADEvolve subset for smoke runs. The subset should be selected by sorted archive member path and a fixed seed, with a config-controlled limit. It should include both `CADEvolve-P/` and `CADEvolve-C/` when available.

Expand only after object validation works for the current subset. Expansion order should be by archive subtree, for example:

1. one small core subtree
2. the rest of that tree
3. another CADEvolve executable tree
4. all configured executable trees

Where metadata is available, stratify object selection by:

- source tree and subtree
- generator or family ID
- operation signature, such as use of `box`, `extrude`, `cut`, `union`, `loft`, `sweep`, `fillet`, `chamfer`, or `shell`
- object complexity, such as operation count, source token count, number of solids in a compound, bounding-box aspect ratio, or volume-to-bounding-box-volume ratio
- validation status and failure category for reporting, not for accepted-example balancing

Avoid random script-level train/test leakage when generator or family IDs are available. Examples derived from the same generator/family must be eligible for generator-held-out grouping. If generator/family ID is unavailable, store enough metadata to support conservative fallback grouping by source subtree, source path prefix, object ID, object pair ID, assembly group ID, and counterfactual group ID.

Accepted geometry records must store enough metadata to support:

- random split sanity checks
- generator-held-out splits
- object-pair-held-out splits
- assembly-group-held-out splits
- counterfactual-group-held-out splits
- near-boundary hard subsets

## 5. Minimal Synthetic Fixtures

Synthetic objects are allowed only for:

- unit tests
- golden geometry cases
- transform convention tests
- relation-label edge cases
- smoke tests when CADEvolve is unavailable locally

Synthetic fixtures must be tagged with `primitive_fixture` and must not dominate released benchmark counts. They are not a substitute for CADEvolve ingestion.

Required fixture examples:

- separated boxes
- touching boxes
- tiny-overlap boxes
- near-miss boxes
- contained boxes
- simple ring or plate-with-hole where AABBs overlap but exact solids do not

Fixture geometry should use simple, auditable dimensions in millimetres and should be traceable to expected labels in `specs/label_rules.md`.

## 6. Object Validation Policy

A source object is accepted for candidate generation only if all of these are true:

- the script executes inside an isolated worker
- execution completes before the configured timeout
- the selected output is a CadQuery/CQ solid, a supported OpenCASCADE solid, or a convertible object that produces a valid solid or valid compound for two-solid labeling
- volume is finite and positive
- bounding box coordinates are finite
- the object is not degenerate under the active validation thresholds
- object-level diagnostics can be serialized deterministically

Rejected source objects must be recorded in a source validation manifest with a machine-readable failure reason. Use the failure categories in `specs/label_rules.md` where possible, including:

- `source_parse_error`
- `source_exec_error`
- `missing_result_object`
- `invalid_cadquery_type`
- `non_solid_result`
- `zero_or_negative_volume`
- `non_finite_bbox`
- `timeout`
- `worker_crash`
- `unknown_error`

Rejected source objects are not eligible for normal MVP task rows.

## 7. Assembly Candidate Strategies

All candidate strategies propose object pairs and transforms. Exact labels from the geometry labeling pipeline decide final acceptance, relation, answer, and bucket.

| Strategy | Goal | Inputs required | Expected labels | Failure/rejection conditions | MVP status |
| --- | --- | --- | --- | --- | --- |
| Broad random CADEvolve object pairing | Provide source diversity and avoid overfitting to a narrow geometry motif. | Validated CADEvolve objects, deterministic pair sampler, transform sampler, object metadata. | Mostly `disjoint`, with some `intersecting` and occasional `near_miss` or `touching`. | Invalid object, prompt too long, duplicate object pair/transform hash, label failure, severe class imbalance after sampling. | Required, but not sufficient alone. |
| Bbox-guided near-overlap placement | Efficiently propose candidates near potential contact or interference boundaries. | Validated object AABBs, object scales, deterministic perturbation sampler. | `touching`, `near_miss`, `tiny_overlap`, `intersecting`, and `disjoint`. | AABB placement cannot be computed, exact label unstable, repeated duplicates, candidate outside configured scale limits. | Required. |
| Contact-targeted placement | Produce examples that separate contact from positive-volume interference. | Surface or bbox face placement heuristic, small perturbation values, exact distance labels. | `touching` when exact distance is within `epsilon_distance_mm`; nearby variants may be `near_miss` or `intersecting`. | Exact distance unavailable, contact not achieved within tolerance, Boolean/distance inconsistency, unstable relation. | Required for fixtures and preferred for CADEvolve where reliable. |
| Near-miss perturbation | Produce positive-clearance examples below `near_miss_threshold_mm`. | A boundary candidate, perturbation direction, configured clearance range, exact distance labels. | `near_miss`. | Minimum distance unavailable, clearance is zero/touching, clearance exceeds threshold, exact overlap appears. | Required. |
| Tiny-overlap perturbation | Produce small positive intersections above `epsilon_volume` but close to the decision boundary. | A boundary candidate, perturbation direction, configured overlap range, exact Boolean labels. | `intersecting`, usually with `tiny_overlap` and `near_boundary` tags. | Intersection volume is at or below `epsilon_volume`, overlap is too large for tiny-overlap target, unstable Boolean result. | Required. |
| Cavity/concavity-targeted placement | Produce examples where AABB overlap is misleading and exact geometry matters. | Objects with holes, rings, slots, shells, cavities, brackets, high bbox-volume slack, or operation signatures containing cuts/shells/booleans. | Often `disjoint` or `near_miss` with `aabb_exact_disagreement`; may also produce `intersecting`. | No useful cavity/concavity diagnostics, exact labels unavailable, placement cannot enter void, prompt too long. | Optional for first smoke run; P1 for richer MVP. |
| Counterfactual transform sweep | Generate variants that share an object pair and script structure while one transform parameter changes. | Base object pair, base transform, one varied transform parameter, deterministic sweep values, exact labels for each variant. | At least two labels when possible, such as `near_miss` and `intersecting`, or `touching` and `intersecting`. | Group has only one accepted variant, no label diversity after configured attempts, multiple parameters changed unintentionally, group metadata missing. | Required once base labeling is stable. |

Candidate generation should keep failed attempts in diagnostics or generation reports. Silent dropping is not allowed.

## 8. Transform Sampling

Translation units are millimetres. Rotation values are degrees.

The transform schema is:

```json
{
  "translation": [0.0, 0.0, 0.0],
  "rotation_xyz_deg": [0.0, 0.0, 0.0],
  "rotation_order": "XYZ"
}
```

MVP transforms must include:

- axis-aligned examples with zero rotation
- single-axis rotations around X, Y, or Z
- selected multi-axis rotations using XYZ Euler order

Sampling must be deterministic from seed and config. Do not depend on Python process order, worker completion order, filesystem iteration order, or tar iteration order without deterministic sorting.

Transform values must be stored exactly as used in:

- prompt text
- assembly code or transform metadata
- geometry label records
- task rows
- counterfactual variant metadata

If values are rounded for prompt readability, the rounded values must be the values actually used for labeling, or the unrounded values must be exposed consistently in both prompt and metadata. Prompt transforms and stored transforms must not disagree.

## 9. Boundary and Epsilon Policy

The active threshold policy comes from `specs/label_rules.md` and must be stored with every dataset version. Generation may target boundary classes, but final relations are derived only from exact geometry fields and configured thresholds.

Candidate perturbation classes:

- `touching`: exact or near-exact touching where controllable; no policy-positive intersection volume and `minimum_distance <= epsilon_distance_mm`
- `near_miss`: positive clearance with `epsilon_distance_mm < minimum_distance <= near_miss_threshold_mm`
- `tiny_overlap`: positive intersection above `epsilon_volume`, preferably close to the threshold
- `clear_overlap`: positive intersection comfortably above `epsilon_volume`
- `clear_disjoint`: positive clearance comfortably above `near_miss_threshold_mm`

Numerically unstable examples may be rejected or accepted with an explicit stability tag. Reject by default when repeated labeling, threshold audits, or consistency checks disagree on the relation needed by an MVP task.

Near-boundary examples should store raw `intersection_volume`,
`normalized_intersection`, and `minimum_distance` through normal label metadata,
and should store `epsilon_volume_ratio`, `epsilon_distance_mm`, and
`near_miss_threshold_mm` through `label_policy`. Validators derive the
record-specific `epsilon_volume` from the stored ratio and object volumes.

## 10. Counterfactual Groups

A counterfactual group shares:

- object pair
- source scripts or normalized object functions
- assembly script structure
- all non-varied transforms and object parameters

Only one transform or object parameter should vary within a group. For the CADEvolve MVP, prioritize transform counterfactuals because editing arbitrary CADEvolve source parameters may be harder and less reliable than varying assembly transforms.

Each group should contain at least two different derived labels when possible. If a group cannot achieve label diversity after the configured sweep, keep the failure summary for diagnostics and do not count the group as a successful counterfactual group.

Group members are split-inseparable. No normal train/validation/test split may place variants from the same `counterfactual_group_id` in different splits.

Every group member must store:

- `counterfactual_group_id`
- `variant_id`
- `base_object_pair_id`
- `assembly_group_id`
- `changed_parameter`
- `changed_value`

Recommended MVP varied parameters:

- `translation_x`
- `translation_y`
- `translation_z`
- `rotation_x_deg`
- `rotation_y_deg`
- `rotation_z_deg`

## 11. Difficulty Tags

Initial difficulty and diagnostic tags:

- `axis_aligned`
- `cadevolve_simple`
- `cadevolve_compound`
- `primitive_fixture`
- `rotated`
- `compound_boolean`
- `cavity_targeted`
- `contact_vs_interference`
- `near_boundary`
- `tiny_overlap`
- `near_miss`
- `aabb_exact_disagreement`
- `contained`
- `invalid`

Tags are multi-label diagnostics, not mutually exclusive classes. For example, one record may be tagged `cadevolve_compound`, `rotated`, `near_boundary`, and `tiny_overlap`.

Use `invalid` only for internal manifests, failure reports, audit data, or explicit invalid-task variants. Normal MVP task exports should prefer `label_status == "ok"` rows.

## 12. Class Balance Targets

Provisional accepted-example targets by relation class:

| Relation target | Target share |
| --- | --- |
| `intersecting` | 40% |
| `disjoint` | 30% |
| `touching` | 15% |
| `near_miss` | 15% |

These are targets, not hard guarantees. CADEvolve object availability, source validation failures, OpenCASCADE kernel failures, boundary-generation reliability, and prompt-length limits may affect actual distributions.

When `contained` examples are generated, report them separately. They may be included in the binary-interference positive class, but relation-classification reports must distinguish `contained` from ordinary `intersecting`.

Balance must be reported per split and per task type. Reports should include at least counts by:

- relation
- binary answer
- volume bucket
- source tree/subtree
- candidate strategy
- difficulty tag
- split

## 13. Candidate Rejection Rules

Reject a candidate geometry record or task row when any required condition fails:

- invalid source object
- zero or negative volume
- non-finite bounding box
- geometry label failure for required fields
- unstable relation under configured thresholds
- duplicate geometry/transform hash already accepted
- prompt too long for configured limits
- missing required provenance, object-pair, assembly, or group IDs
- missing or non-deterministic transform metadata
- source code cannot be normalized into the prompt/task format
- required exact label contradicts label consistency rules in `specs/label_rules.md`

Rejected candidates must produce a failure reason and enough context to audit source, strategy, and attempted transform. Do not silently drop rejected candidates.

## 14. Smoke Dataset Requirements

The first smoke target should use CADEvolve where possible:

- 100 accepted geometry records
- at least 20 validated CADEvolve source objects if available locally
- at least one accepted example each for:
  - clear disjoint
  - touching
  - near-miss
  - tiny overlap
  - clear overlap
  - rotated examples
- counterfactual groups if the counterfactual generator is implemented

The smoke dataset may use synthetic fixtures only to fill golden edge cases that CADEvolve sampling does not reliably produce in the first smoke run. Synthetic fill-ins must be counted and tagged separately.

Smoke generation must emit a compact report with:

- number of CADEvolve archive members scanned
- number of source objects validated and rejected
- accepted geometry-record counts by relation
- accepted task-row counts by task type
- synthetic fixture count
- candidate rejection counts by failure reason
- seed, config hash, source manifest hash, and label policy

## 15. Full Dataset Policy

Full dataset size must be config-controlled. Exact target sizes can be updated after CadQuery execution throughput, source validation yield, exact labeling throughput, and prompt-materialization cost are measured.

Full generation should be shardable. A shard must be reproducible from:

- global config
- shard ID
- shard seed or deterministic seed derivation
- source manifest hash
- label policy

Accepted examples should come primarily from CADEvolve. Synthetic fixtures may remain in golden/audit/smoke subsets, but they should not dominate released benchmark counts and should not be presented as the main real-data source.

Full exports should derive all MVP task rows from accepted geometry records. Do not recompute geometry separately for each task format.

## 16. Determinism and Reproducibility

All generation uses explicit seeds. Every stochastic choice must be derived from the configured seed and stable candidate keys.

IDs must be deterministic. Stable IDs should be derived from canonical content or deterministic counters after stable sorting, not from wall-clock time, process ID, worker ID, or parallel completion order.

Parallel completion order must not affect:

- accepted source ordering
- object pair ordering
- geometry IDs
- task row IDs
- split assignment
- class balance decisions
- exported file content

Every generation run must store:

- generation config hash
- source manifest hash
- code commit hash
- dataset version
- label policy name/version and threshold values
- CadQuery version
- OCP/OpenCASCADE version when available
- Python version
- CADEvolve archive filename
- CADEvolve archive hash or version metadata
- CADEvolve license metadata

Changing label thresholds, source manifests, transform sampling, prompt templates, or normalization rules must change the relevant reproducibility hash and should trigger a new dataset version or explicit relabeling audit.

## 17. Diagnostics to Store

Store these diagnostics for accepted examples and, where applicable, rejected candidates:

- CADEvolve archive member path
- normalized source path
- source tree and subtree
- candidate strategy
- difficulty tags
- operation signature if available
- object family or topology tags if derivable
- transform family
- raw transform values
- label policy
- AABB overlap
- exact overlap
- raw `intersection_volume`
- raw `normalized_intersection`
- raw `minimum_distance`
- `boolean_status`
- `distance_status`
- `label_status`
- rejection or failure reason for skipped candidates

Diagnostics may be kept outside public prompt text, but they must be available for dataset validation, split audits, baseline analysis, and paper statistics.

## 18. Anti-Patterns

Do not build a separate full synthetic CAD dataset before using CADEvolve.

Do not rely on purely random placement. Random placement is allowed for diversity, but boundary-targeted, contact-targeted, near-miss, tiny-overlap, and counterfactual strategies are required for useful MVP data.

Do not use AABB overlap as ground truth when boxes overlap. AABB non-overlap may safely rule out positive-volume intersection, but AABB overlap is only a diagnostic and candidate-generation signal.

Do not split counterfactual variants across main splits.

Do not silently drop failures. Record source failures, candidate rejections, geometry failures, prompt-length exclusions, and duplicate exclusions with machine-readable reasons.

Do not execute CADEvolve scripts in the main process.

Do not recompute geometry separately for every prompt row. Validate source objects once, label accepted assembly variants once, and derive task rows from stored geometry labels.
