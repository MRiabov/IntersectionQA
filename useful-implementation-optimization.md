# Useful Implementation Optimization Notes

This document records implementation choices that should make IntersectionQA easier to build, optimize, debug, and release. It is written for a CAD/3D workload where the expensive work is geometry execution, Boolean intersection, distance queries, meshing, rendering, and validation.

The main design assumption is:

> Dataset generation is offline. The released dataset should usually contain CadQuery code, transforms, prompts, answers, precomputed labels, diagnostics, provenance, and split metadata. It should not embed mesh files, STEP files, or other heavyweight geometry artifacts unless a separate optional artifact bundle is explicitly created.

## Core Release Decision

The default dataset format should be code-plus-label, not geometry-plus-label.

Store in public JSONL records:

- object source code or normalized object functions
- assembly code or transform metadata sufficient to reconstruct the assembly
- task prompt
- final answer
- raw labels such as intersection volume, normalized intersection, minimum distance, and relation
- diagnostic labels such as AABB overlap, OBB overlap if available, failure reason, difficulty tag, and operation signature
- provenance such as source ID, generator ID, object-pair ID, assembly-group ID, counterfactual-group ID, config hash, code hash, and dataset version

Do not store by default:

- STEP files
- STL/OBJ/glTF meshes
- B-Rep dumps
- rendered images
- local cache paths
- serialized CadQuery/OpenCASCADE objects

Allowed exceptions:

- a small audit bundle for manual inspection
- optional render bundles for multimodal experiments
- optional geometry-cache artifacts used only inside local generation runs
- failure reproducer artifacts stored outside the main dataset export

This keeps the dataset smaller, easier to host, easier to version, and less likely to be blocked by artifact storage or geometry serialization problems.

## Pipeline Shape

Separate the system into four stages:

1. Source normalization
2. Geometry labeling
3. Task materialization
4. Export and validation

The expensive CadQuery/OpenCASCADE work should happen only in stage 2. Stages 3 and 4 should consume precomputed metadata and avoid loading CadQuery entirely where possible.

Recommended data flow:

```text
source CadQuery programs
  -> normalized object records
  -> validated object cache
  -> generated assembly candidates
  -> offline exact labels and diagnostics
  -> task rows and prompts
  -> split-safe JSONL exports
```

Important rule:

> Do not recompute geometry labels separately for every task format.

Compute geometry once per object pair or assembly variant, then derive binary, relation, volume-bucket, clearance-bucket, ranking, pairwise, and repair prompts from the same label record.

## Artifact Policy

Use local caches during generation, but keep them out of the public dataset.

Suggested local directories:

```text
.cache/intersectionqa/objects/
.cache/intersectionqa/assemblies/
.cache/intersectionqa/labels/
.cache/intersectionqa/renders/
.cache/intersectionqa/failures/
```

Suggested release directories:

```text
data/intersectionqa_v0_1/train.jsonl
data/intersectionqa_v0_1/validation.jsonl
data/intersectionqa_v0_1/test_random.jsonl
data/intersectionqa_v0_1/test_generator_heldout.jsonl
data/intersectionqa_v0_1/test_object_pair_heldout.jsonl
data/intersectionqa_v0_1/test_near_boundary.jsonl
data/intersectionqa_v0_1/metadata.json
data/intersectionqa_v0_1/schema.json
```

If optional artifacts are produced, link them by stable content IDs, not local absolute paths.

```json
{
  "artifact_ids": {
    "debug_step": null,
    "render_iso": "render_sha256_..."
  }
}
```

## Record Granularity

Use two record layers internally:

1. Geometry label record
2. Task prompt record

The geometry label record is the expensive unit. It describes one assembled object pair or one multi-object assembly with all raw labels.

The task prompt record is cheap. It references one or more geometry label records and defines a prompt-answer pair.

Example internal geometry record:

```json
{
  "geometry_id": "geom_000001",
  "source": "synthetic",
  "object_a_id": "obj_000013",
  "object_b_id": "obj_000097",
  "base_object_pair_id": "pair_000021",
  "assembly_group_id": "asmgrp_000045",
  "counterfactual_group_id": "cfg_000008",
  "variant_id": "cfg_000008_v03",
  "transform_a": {
    "translation": [0.0, 0.0, 0.0],
    "rotation_xyz_deg": [0.0, 0.0, 0.0]
  },
  "transform_b": {
    "translation": [9.95, 0.0, 0.0],
    "rotation_xyz_deg": [0.0, 0.0, 0.0]
  },
  "labels": {
    "volume_a": 1000.0,
    "volume_b": 1000.0,
    "intersection_volume": 5.0,
    "normalized_intersection": 0.005,
    "minimum_distance": 0.0,
    "relation": "intersecting"
  },
  "diagnostics": {
    "aabb_overlap": true,
    "exact_overlap": true,
    "label_status": "ok",
    "failure_reason": null
  }
}
```

Example public task row:

```json
{
  "id": "intersectionqa_binary_000001",
  "geometry_ids": ["geom_000001"],
  "task_type": "binary_interference",
  "prompt": "...",
  "answer": "yes",
  "script": "...",
  "labels": {
    "relation": "intersecting",
    "intersection_volume": 5.0,
    "normalized_intersection": 0.005
  },
  "diagnostics": {
    "aabb_overlap": true,
    "difficulty": "near_boundary"
  },
  "split": "train"
}
```

## Caching Strategy

Cache by stable hashes instead of by incidental file names.

Recommended hashes:

- `source_code_hash`: normalized source text
- `object_hash`: normalized object code plus source metadata
- `transform_hash`: canonical transform JSON
- `geometry_hash`: object hashes plus transform hashes plus label thresholds
- `config_hash`: full generation config
- `prompt_hash`: prompt template version plus geometry IDs

The geometry hash should include tolerance settings. A change to `epsilon_volume`, `epsilon_distance`, or near-miss thresholds can change relation labels and must invalidate affected cache entries.

Do not cache only final task rows. Cache the expensive object validation and geometry label records so prompt formats can be regenerated cheaply.

## Offline Worker Model

Run CadQuery/OpenCASCADE work in isolated worker processes.

Reasons:

- CADEvolve programs are arbitrary Python and should be treated as untrusted input.
- Geometry kernels can leak memory or leave problematic global state.
- Some Boolean and distance operations can hang or become very slow.
- A failing source script should not poison the main generation process.

Recommended worker behavior:

- one job per object validation or geometry labeling unit
- subprocess timeout per job
- maximum memory limit where practical
- structured JSON result on success or failure
- explicit failure reason categories
- no access to network
- temporary working directory per job

Useful failure categories:

```text
source_parse_error
source_exec_error
missing_result_object
invalid_cadquery_type
non_solid_result
zero_or_negative_volume
non_finite_bbox
boolean_intersection_failed
distance_query_failed
timeout
worker_crash
unknown_error
```

## Broad Phase Before Exact Geometry

Use cheap diagnostics before expensive exact queries.

For binary interference:

- If transformed AABBs do not overlap, positive-volume intersection is impossible.
- This can safely label binary interference as `no`.
- If the task also needs exact clearance distance, still run the distance query or mark distance as unavailable.

For relation classification:

- AABB non-overlap can establish `disjoint`, but not an exact minimum distance.
- AABB overlap cannot prove intersection because holes, cavities, rotations, and concavities can fool it.
- Touching and near-miss labels need exact distance or a controlled generator construction.

For multi-object assemblies:

- Build candidate pairs using AABB overlap or spatial indexing.
- Run exact pair labels only for plausible candidates and diagnostic negatives.
- Keep sampled AABB-non-overlap negatives so the dataset remains balanced.

The broad phase is an optimization and a diagnostic baseline. It should not replace exact labels for hard examples.

## Geometry Labeling Rules

Keep raw values and derived labels separate.

Raw values:

- volume A
- volume B
- intersection volume
- normalized intersection
- minimum distance
- bounding boxes
- Boolean status
- distance status

Derived labels:

- binary interference
- relation class
- volume bucket
- clearance bucket
- contained flag
- near-boundary flag

Recommended relation rule:

```text
if invalid:
    relation = invalid
elif intersection_volume > epsilon_volume:
    relation = intersecting or contained
elif minimum_distance <= epsilon_distance:
    relation = touching
elif minimum_distance <= near_miss_threshold:
    relation = near_miss
else:
    relation = disjoint
```

Always store the thresholds used to derive labels.

```json
{
  "label_policy": {
    "epsilon_volume_ratio": 1e-6,
    "epsilon_distance_mm": 1e-4,
    "near_miss_threshold_mm": 1.0
  }
}
```

## Avoid Repeated CadQuery Execution

CadQuery execution should be paid for once per normalized object and once per assembly variant.

Recommended approach:

- validate each source object once
- store object-level metadata: volume, bbox, operation signature, validity, failure reason
- generate many candidate transforms using object metadata
- label selected assembly variants exactly
- derive multiple tasks from the same geometry labels

Avoid this pattern:

```text
for each task row:
    execute object_a code
    execute object_b code
    build assembly
    compute Boolean
    generate prompt
```

Prefer this pattern:

```text
for each source object:
    execute and validate once

for each assembly variant:
    execute/place objects and compute labels once

for each task type:
    generate prompt rows from stored labels
```

## Candidate Generation Optimization

Random SE(3) sampling will waste many exact label calls. Use cheap object metadata to generate useful candidates.

Recommended candidate strategies:

- broad random placements for diversity
- bbox-guided placements near overlap boundaries
- controlled surface-to-surface contact placements for touching cases
- epsilon perturbations for near-miss and tiny-overlap cases
- cavity-targeted placements for AABB-overlap-but-exact-disjoint examples
- counterfactual sweeps over one parameter at a time

Use exact CAD labels as the final accept/reject authority, but use bbox and known primitive parameters to propose candidates efficiently.

## Counterfactual Groups

Counterfactual generation is both a benchmark feature and an optimization feature.

It is efficient because one base object pair can produce several useful variants:

- clearly disjoint
- near-miss
- touching
- tiny overlap
- larger overlap

Do not split variants from the same counterfactual group across train, validation, or test unless the experiment is explicitly an interpolation test.

For storage efficiency, task rows can repeat full scripts for standalone usability, but internal generation should reference shared object and geometry records to avoid repeated work.

## Dataset Export Principles

Make exported examples self-contained enough for model training and evaluation, while avoiding heavy geometry artifacts.

Each JSONL row should include:

- prompt
- answer
- task type
- split
- object or assembly code shown to the model
- enough labels for evaluation and slicing
- enough provenance for leakage audits
- enough diagnostics for paper tables

Each JSONL row should not require:

- CadQuery installation to read the dataset
- mesh or STEP downloads
- path access to local caches
- execution just to obtain the official label

This allows training pipelines to treat IntersectionQA as a normal text/code dataset.

## Storage Format

Use JSONL for released task rows. Use Parquet, SQLite, DuckDB, or JSONL for internal manifests depending on scale.

Recommended split:

- JSONL for final public examples
- Parquet or DuckDB for internal geometry records and statistics
- SQLite for local job queue state if generation becomes long-running
- plain JSON for config, metadata, and schema snapshots

Large scale generation benefits from append-only writes:

- write completed job results incrementally
- validate each shard independently
- merge only validated shards
- keep failed jobs in a separate failure manifest

## Parallelism

Use process-level parallelism, not thread-level parallelism, for geometry work.

Reasons:

- CadQuery/OpenCASCADE work may not behave well under Python threads.
- Python execution of arbitrary source code needs isolation.
- Process restarts are useful after memory growth or kernel errors.

Recommended knobs:

```yaml
labeling:
  worker_count: 4
  worker_timeout_s: 30
  max_jobs_per_worker: 100
  cache_dir: .cache/intersectionqa/labels
  retry_count: 1
```

Keep worker count configurable. Geometry kernels can be CPU-heavy and memory-heavy; a high worker count may be slower if it causes memory pressure.

## Determinism

Every generation run should be reproducible from:

- dataset version
- code commit hash
- config hash
- source manifest hash
- random seed
- label policy
- CadQuery/OpenCASCADE version if available

Stable IDs should be generated from content and deterministic counters, not from wall-clock order. Parallel job completion order should not affect final IDs.

## Validation Before Export

Validation should be cheap and run often.

Required checks:

- schema validity for every row
- answer matches labels
- normalized intersection is within expected range
- positive intersection does not coexist with positive minimum distance
- relation label agrees with threshold policy
- invalid examples are either excluded or explicitly tagged
- all required group IDs exist for split-sensitive examples
- no generator, object-pair, assembly, or counterfactual group crosses forbidden split boundaries
- task distributions and label distributions are reported
- every exported row has a prompt, answer, task type, split, and dataset version

For a smoke dataset, generate a small fixed set with:

- disjoint
- touching
- near-miss
- tiny overlap
- clear overlap
- contained if implemented
- AABB-overlap but exact-disjoint if a cavity motif exists

## Geometry Artifact Debugging

When debugging labels, it is useful to write temporary STEP or mesh files. Keep this outside the normal dataset path.

Suggested debug command behavior:

```text
inspect_example.py --id intersectionqa_binary_000001 --write-debug-artifacts .cache/intersectionqa/debug/000001
```

Debug artifacts should be reproducible from code and metadata. The dataset should not depend on them.

## Rendering

Rendering should be optional and downstream of labels.

Do not make rendering a dependency for:

- source validation
- exact labeling
- prompt generation
- code-only model evaluation
- dataset export

If multimodal experiments are added, render from already accepted geometry records and store render links in a separate artifact manifest.

## Baseline Efficiency

Precompute diagnostics needed by non-LLM baselines.

At label time, store:

- AABB overlap
- bbox coordinates
- exact overlap
- relation
- difficulty tags
- operation signature

Then AABB baseline evaluation can run without CadQuery.

OBB and convex-hull baselines may require additional geometry computation. Treat them as optional offline jobs with their own cached diagnostics, not as work done during every prompt export.

## CADEvolve Handling

CADEvolve source programs should be processed as untrusted Python.

Recommended approach:

- iterate tar members without extracting everything permanently
- filter for executable CadQuery program paths
- record archive member name as source provenance
- parse operation signatures cheaply where possible
- execute only inside isolated workers
- normalize output object conventions such as `result`, `shape`, `solid`, or `part`
- store execution failures with source IDs
- validate volume and finite bounding box before using any object in pair generation

Do not export CADEvolve mesh or STEP conversions by default. Export the source code or normalized object function plus labels.

## Prompt Generation Efficiency

Prompt generation should be a pure transformation:

```text
geometry record + prompt template version -> task row
```

It should not:

- call CadQuery
- compute Booleans
- compute distances
- write mesh files
- depend on local geometry caches

This makes prompt iteration cheap. It also makes it safe to update answer formats, task wording, or dataset splits without repeating expensive labels unless the geometry selection changes.

## Implementation Order

Build in this order:

1. Define schema for object records, geometry label records, and task rows.
2. Implement synthetic primitive source and validation.
3. Implement isolated CadQuery worker and exact labels.
4. Add object-level and geometry-level caches.
5. Implement bbox-guided candidate generation.
6. Generate binary and relation task rows from stored labels.
7. Add split-safe export and validation.
8. Add counterfactual groups.
9. Add volume and clearance buckets.
10. Add CADEvolve ingestion after the synthetic path is stable.
11. Add optional rendering and artifact bundles only after code-only release flow works.

## Optimization Priorities

Highest value:

- cache object validation and geometry labels
- isolate CadQuery execution in workers
- short-circuit safe AABB-disjoint binary cases
- generate candidates near useful boundaries instead of random-only sampling
- derive many task rows from one label record
- keep public exports geometry-free

Medium value:

- use DuckDB/Parquet for internal analysis at scale
- shard generation and validation
- add retry logic for transient geometry failures
- precompute operation signatures for dataset slicing
- add bbox or spatial-index pair filtering for multi-object assemblies

Lower value for v1:

- global point-cloud duplicate filtering
- Chamfer-distance leakage filtering
- always-on rendering
- storing STEP/STL artifacts for every example
- dense mesh-based baselines in the core pipeline

## Anti-Patterns To Avoid

Avoid:

- embedding heavy geometry files in every dataset row
- making CadQuery required for normal dataset loading
- recomputing exact labels during prompt export
- using random script-level splits as the primary benchmark split
- treating AABB overlap as a ground-truth intersection label when boxes overlap
- hiding Boolean or distance failures by silently dropping rows
- storing only bucket labels and discarding raw values
- mixing label policy changes into the same dataset version without a version bump
- allowing parallel job order to define example IDs

## Practical Default

The practical default should be:

```yaml
export:
  include_step_files: false
  include_mesh_files: false
  include_renders: false
  include_code: true
  include_precomputed_labels: true
  include_diagnostics: true

labeling:
  mode: offline
  cache_geometry_labels: true
  use_isolated_workers: true

tasks:
  materialize_from_labels_only: true
```

This matches the intended IntersectionQA use case: train and evaluate models on CAD-code spatial reasoning while keeping the expensive, brittle, kernel-dependent computation in an offline data-construction pipeline.
