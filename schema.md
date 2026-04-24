# IntersectionQA Schema Specification

## 1. Purpose

This file defines the canonical data model for IntersectionQA. It is the source of truth for future Pydantic models, JSON Schema export, JSONL writers, validators, split audits, and dataset packaging.

IntersectionQA uses two record layers:

- Internal records describe normalized source objects, object validation results, and expensive geometry labels. These may be stored in internal JSONL, Parquet, DuckDB, SQLite, or cache files during dataset construction.
- Public JSONL task rows are released for training and evaluation. They contain prompts, answers, scripts, labels, diagnostics, provenance, split metadata, and hashes.

Public rows must be self-contained for training and evaluation. Loading a released JSONL row must not require CadQuery execution, OpenCASCADE, STEP/STL/mesh files, render files, or access to local caches to read the official answer, labels, diagnostics, provenance, or split assignment.

## 2. Design Principles

- Store raw geometry values and derived labels separately. Raw values include volumes, intersection volume, normalized intersection, minimum distance, bounding boxes, and computation statuses. Derived labels include relation, binary answer, volume bucket, clearance bucket, containment label, and difficulty tags.
- Compute expensive geometry once per geometry record, then derive multiple task rows from that record. Prompt generation and export must not run CadQuery, Booleans, distance queries, meshing, or rendering.
- Use stable IDs and hashes. Dataset content must be reproducible from dataset version, source manifest, config, label policy, code commit, CadQuery/OCP versions, and deterministic seeds.
- Include provenance and split metadata in every public row. Public rows must carry enough group IDs to audit generator, object-pair, assembly-group, and counterfactual leakage.
- Do not store STEP, STL, OBJ, glTF, B-Rep dumps, serialized CadQuery/OpenCASCADE objects, meshes, or render images in default public rows.
- Optional artifacts should be referenced by stable artifact IDs, not local paths. Artifact bundles, debug reproducers, and render bundles are separate from the default public JSONL export.
- Keep keys stable across dataset versions. Add new optional fields under `metadata` or a versioned nested object before changing or removing public top-level fields.

## 3. Common Types

All coordinates and distances are in millimetres. Volumes are in cubic millimetres. Numeric fields must be finite unless explicitly nullable. JSON object keys use `snake_case`.

### `Transform`

Rigid transform applied to an object before geometry labeling.

| Field | Type | Required | Nullable | Meaning |
| --- | --- | --- | --- | --- |
| `translation` | array of 3 numbers | yes | no | `[x, y, z]` translation in millimetres. |
| `rotation_xyz_deg` | array of 3 numbers | yes | no | `[rx, ry, rz]` Euler rotations in degrees. |
| `rotation_order` | string enum | yes | no | Must be `"XYZ"` for all v0.1 records. |

```json
{
  "translation": [9.95, 0.0, 0.0],
  "rotation_xyz_deg": [0.0, 0.0, 15.0],
  "rotation_order": "XYZ"
}
```

### `BoundingBox`

Axis-aligned bounding box in world coordinates after transforms unless the field explicitly says it is object-local.

| Field | Type | Required | Nullable | Meaning |
| --- | --- | --- | --- | --- |
| `min` | array of 3 numbers | yes | no | Minimum `[x, y, z]` coordinates. |
| `max` | array of 3 numbers | yes | no | Maximum `[x, y, z]` coordinates. |

`max[i]` must be greater than or equal to `min[i]` for every axis.

```json
{
  "min": [-5.0, -5.0, -5.0],
  "max": [5.0, 5.0, 5.0]
}
```

### `LabelPolicy`

Threshold policy used to derive relation labels and bucket labels from raw geometry.

| Field | Type | Required | Nullable | Default for v0.1 |
| --- | --- | --- | --- | --- |
| `epsilon_volume_ratio` | number | yes | no | `1e-6` |
| `epsilon_distance_mm` | number | yes | no | `1e-4` |
| `near_miss_threshold_mm` | number | yes | no | `1.0` |

Derived value:

```text
epsilon_volume = epsilon_volume_ratio * min(volume_a, volume_b)
```

Changing any label policy value requires a new dataset version or an explicit relabeling audit.

```json
{
  "epsilon_volume_ratio": 0.000001,
  "epsilon_distance_mm": 0.0001,
  "near_miss_threshold_mm": 1.0
}
```

### `ArtifactIds`

Stable optional references to artifacts stored outside default public JSONL rows. Values are content IDs or manifest IDs, never local filesystem paths.

| Field | Type | Required | Nullable | Meaning |
| --- | --- | --- | --- | --- |
| `debug_step_id` | string | no | yes | Optional STEP artifact ID for manual audit. |
| `debug_mesh_id` | string | no | yes | Optional mesh artifact ID for manual audit. |
| `render_iso_id` | string | no | yes | Optional isometric render ID. |
| `render_orthographic_ids` | array of strings | no | yes | Optional render IDs for named orthographic views. |
| `overlap_render_id` | string | no | yes | Optional transparent overlap visualization ID. |
| `failure_reproducer_id` | string | no | yes | Optional reproducer artifact ID for failed jobs. |

```json
{
  "debug_step_id": null,
  "debug_mesh_id": null,
  "render_iso_id": "render_sha256_aaaaaaaaaaaaaaaa",
  "render_orthographic_ids": null,
  "overlap_render_id": null,
  "failure_reproducer_id": null
}
```

### `Hashes`

Hashes use lowercase SHA-256 strings with the prefix `sha256:` unless a future schema version explicitly permits another algorithm. The `Hashes` object is present in every major record type. A hash value may be `null` only when the hash does not apply to that record type.

| Field | Type | Required | Nullable | Meaning |
| --- | --- | --- | --- | --- |
| `source_code_hash` | string | yes | yes | Hash of normalized source code text. Non-null for source object records and public rows that include code. |
| `object_hash` | string | yes | yes | Hash of normalized object identity/content. For pair records, this may be a canonical combined hash of object A and object B. |
| `transform_hash` | string | yes | yes | Hash of canonical transform JSON and transform-family metadata. |
| `geometry_hash` | string | yes | yes | Hash of object hashes, transforms, label policy, and geometry-labeling config. |
| `config_hash` | string | yes | yes | Hash of the dataset generation configuration. |
| `prompt_hash` | string | yes | yes | Hash of prompt template version, task type, prompt text inputs, and geometry IDs. Non-null for public task rows. |

```json
{
  "source_code_hash": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
  "object_hash": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
  "transform_hash": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
  "geometry_hash": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
  "config_hash": "sha256:5555555555555555555555555555555555555555555555555555555555555555",
  "prompt_hash": "sha256:6666666666666666666666666666666666666666666666666666666666666666"
}
```

### Common Enums

`relation`:

```text
disjoint
touching
near_miss
intersecting
contained
invalid
```

`boolean_status`:

```text
ok
skipped_aabb_disjoint
failed
not_run
```

`distance_status`:

```text
ok
skipped_positive_overlap
failed
not_run
```

`label_status`:

```text
ok
invalid
```

`task_type`:

```text
binary_interference
relation_classification
volume_bucket
clearance_bucket
pairwise_interference
ranking_normalized_intersection
repair_direction
tolerance_fit
```

The v0.1 MVP public export should include `binary_interference`, `relation_classification`, and `volume_bucket`. Other task types are reserved for compatible future rows.

`split`:

```text
train
validation
test_random
test_generator_heldout
test_object_pair_heldout
test_near_boundary
test_topology_heldout
test_operation_heldout
```

The default public v0.1 export includes the first six split names.

`failure_reason`:

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

## 4. Source Object Record

Source object records are internal normalized records for one CAD object. They are generated before object validation and before assembly generation.

### Fields

| Field | Type | Required | Nullable | Meaning |
| --- | --- | --- | --- | --- |
| `object_id` | string | yes | no | Stable object ID, e.g. `obj_000001`. |
| `source` | string | yes | no | Source family, e.g. `synthetic`, `cadevolve`, `human_cadquery`, or `mechanical_motif`. |
| `source_id` | string | yes | no | Stable ID within the source. For CADEvolve this should identify the archive member or source row. |
| `generator_id` | string | yes | yes | Source generator/family ID used for generator holdout. Null only when the source has no generator concept. |
| `source_path` | string | yes | yes | Archive member path or repository-relative source path. Must not be a local absolute cache path. |
| `source_license` | string | yes | yes | Source license identifier such as `apache-2.0`; null only before license audit completion. |
| `object_name` | string | yes | no | Human-readable object name used for metadata and debugging. |
| `normalized_code` | string | yes | no | Normalized Python/CadQuery code defining the object function. |
| `object_function_name` | string | yes | no | Callable function name in `normalized_code`, e.g. `object_a_source`. |
| `cadquery_ops` | array of strings | yes | no | Stable unique list of CadQuery operations/methods detected in normalized code. Empty array when unavailable. |
| `topology_tags` | array of strings | yes | no | Tags such as `box`, `cylinder`, `ring`, `bracket`, `hollow`, `plate_with_holes`, or `unknown`. Empty array when unavailable. |
| `metadata` | object | yes | no | Source-specific JSON metadata. Must not contain local cache paths as required data. |
| `hashes` | `Hashes` | yes | no | `source_code_hash` and `object_hash` must be non-null. |

### Example

```json
{
  "object_id": "obj_000001",
  "source": "synthetic",
  "source_id": "synthetic_box_10mm",
  "generator_id": "gen_synthetic_primitives_v01",
  "source_path": null,
  "source_license": "cc-by-4.0",
  "object_name": "box_10mm",
  "normalized_code": "import cadquery as cq\n\ndef make_box_10mm():\n    return cq.Workplane(\"XY\").box(10.0, 10.0, 10.0)\n",
  "object_function_name": "make_box_10mm",
  "cadquery_ops": ["box"],
  "topology_tags": ["box", "primitive"],
  "metadata": {
    "units": "mm",
    "source_subset": "synthetic_primitives",
    "parameters": {
      "width": 10.0,
      "depth": 10.0,
      "height": 10.0
    }
  },
  "hashes": {
    "source_code_hash": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
    "object_hash": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    "transform_hash": null,
    "geometry_hash": null,
    "config_hash": "sha256:5555555555555555555555555555555555555555555555555555555555555555",
    "prompt_hash": null
  }
}
```

## 5. Object Validation Record

Object validation records are internal records produced by executing a normalized source object in an isolated worker and checking that it produces usable solid geometry.

### Fields

| Field | Type | Required | Nullable | Meaning |
| --- | --- | --- | --- | --- |
| `object_id` | string | yes | no | Source object being validated. |
| `valid` | boolean | yes | no | True only when the object executes, produces a supported solid, has positive finite volume, and has finite bounding box coordinates. |
| `volume` | number | yes | yes | Object volume in cubic millimetres. Null when invalid or unavailable. |
| `bbox` | `BoundingBox` | yes | yes | Object-local or canonical validation bounding box before assembly transforms. Null when invalid or unavailable. |
| `label_status` | string enum | yes | no | `ok` for valid objects, `invalid` for validation failures. |
| `failure_reason` | string enum | yes | yes | Null when `valid` is true; otherwise a `failure_reason` enum. |
| `cadquery_version` | string | yes | yes | CadQuery version used by the validation worker. |
| `ocp_version` | string | yes | yes | OCP/OpenCASCADE binding version used by the validation worker. |
| `validated_at_version` | string | yes | no | IntersectionQA code version or commit that produced this validation result. |
| `hashes` | `Hashes` | yes | no | `object_hash` and `config_hash` must be non-null. |

### Example

```json
{
  "object_id": "obj_000001",
  "valid": true,
  "volume": 1000.0,
  "bbox": {
    "min": [-5.0, -5.0, -5.0],
    "max": [5.0, 5.0, 5.0]
  },
  "label_status": "ok",
  "failure_reason": null,
  "cadquery_version": "2.5.2",
  "ocp_version": "7.8.1",
  "validated_at_version": "commit:0123456789abcdef0123456789abcdef01234567",
  "hashes": {
    "source_code_hash": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
    "object_hash": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    "transform_hash": null,
    "geometry_hash": null,
    "config_hash": "sha256:5555555555555555555555555555555555555555555555555555555555555555",
    "prompt_hash": null
  }
}
```

## 6. Geometry Label Record

Geometry label records are internal expensive records for one assembled object pair. One geometry record may produce multiple public task rows such as binary interference, relation classification, and volume bucket rows.

### Fields

| Field | Type | Required | Nullable | Meaning |
| --- | --- | --- | --- | --- |
| `geometry_id` | string | yes | no | Stable geometry ID, e.g. `geom_000001`. |
| `source` | string | yes | no | Source family for the object pair. Use `mixed` when objects come from different source families and store per-object source details in `metadata`. |
| `object_a_id` | string | yes | no | Stable ID of object A. |
| `object_b_id` | string | yes | no | Stable ID of object B. |
| `base_object_pair_id` | string | yes | no | Stable group ID for the base object pair, e.g. `pair_000001`. Used for split holdout. |
| `assembly_group_id` | string | yes | no | Stable group ID for the assembly/transform family, e.g. `asmgrp_000001`. Used for split holdout. |
| `counterfactual_group_id` | string | yes | yes | Shared ID for variants that differ by one controlled parameter. Null for non-counterfactual examples. |
| `variant_id` | string | yes | yes | Variant ID such as `cfg_000001_v01`. Null for non-counterfactual examples. |
| `changed_parameter` | string | yes | yes | Name of the single varied parameter, e.g. `translation_x`. Null for non-counterfactual examples. |
| `changed_value` | string, number, boolean, array, object, or null | yes | yes | Value of the changed parameter for this variant. Null for non-counterfactual examples. |
| `transform_a` | `Transform` | yes | no | Transform applied to object A. |
| `transform_b` | `Transform` | yes | no | Transform applied to object B. |
| `assembly_script` | string | yes | no | Complete deterministic script or script fragment sufficient to reconstruct the assembly from normalized object functions. |
| `labels` | object | yes | no | Raw geometry values and derived relation fields. See below. |
| `diagnostics` | object | yes | no | Computation statuses and consistency diagnostics. See below. |
| `difficulty_tags` | array of strings | yes | no | Tags such as `axis_aligned`, `rotated`, `compound`, `cavity_targeted`, `near_boundary`, `tiny_overlap`, `contact_vs_interference`, or `aabb_overlap_exact_disjoint`. |
| `label_policy` | `LabelPolicy` | yes | no | Threshold policy used for this record. |
| `hashes` | `Hashes` | yes | no | `object_hash`, `transform_hash`, `geometry_hash`, and `config_hash` must be non-null. |
| `metadata` | object | yes | no | Generation strategy, seeds, optional artifact IDs, source details, and debug metadata. |

### `labels`

| Field | Type | Required | Nullable | Meaning |
| --- | --- | --- | --- | --- |
| `volume_a` | number | yes | yes | Volume of transformed solid A. Transform does not change volume. Null if invalid. |
| `volume_b` | number | yes | yes | Volume of transformed solid B. Transform does not change volume. Null if invalid. |
| `intersection_volume` | number | yes | yes | Exact Boolean intersection volume. Use `0.0` when exact non-overlap is proven. Null if unavailable. |
| `normalized_intersection` | number | yes | yes | `intersection_volume / min(volume_a, volume_b)` for valid positive volumes. Null if unavailable. |
| `minimum_distance` | number | yes | yes | Exact minimum solid-to-solid clearance. For positive-overlap records this may be `0.0` or null if `distance_status` is `skipped_positive_overlap`. |
| `relation` | string enum | yes | no | One of the relation labels in Section 3. |
| `contained` | boolean | yes | yes | True if either object is contained in the other. False if containment was checked and neither is contained. Null if containment detection was disabled. |
| `contains_a_in_b` | boolean | yes | yes | True if all material of A is contained in B. False if checked and not contained. Null if disabled. |
| `contains_b_in_a` | boolean | yes | yes | True if all material of B is contained in A. False if checked and not contained. Null if disabled. |

### `diagnostics`

| Field | Type | Required | Nullable | Meaning |
| --- | --- | --- | --- | --- |
| `aabb_overlap` | boolean | yes | yes | Closed AABB overlap after transforms. Touching AABBs count as overlap. Null if unavailable. |
| `exact_overlap` | boolean | yes | yes | True iff `intersection_volume > epsilon_volume` under the record's `label_policy`. Null if unavailable. |
| `boolean_status` | string enum | yes | no | `ok`, `skipped_aabb_disjoint`, `failed`, or `not_run`. |
| `distance_status` | string enum | yes | no | `ok`, `skipped_positive_overlap`, `failed`, or `not_run`. |
| `label_status` | string enum | yes | no | `ok` when all fields needed by the relation and intended tasks are valid; otherwise `invalid`. |
| `failure_reason` | string enum | yes | yes | Null for valid labels; otherwise a `failure_reason` enum. |

### Example

```json
{
  "geometry_id": "geom_000001",
  "source": "synthetic",
  "object_a_id": "obj_000001",
  "object_b_id": "obj_000002",
  "base_object_pair_id": "pair_000001",
  "assembly_group_id": "asmgrp_000001",
  "counterfactual_group_id": "cfg_000001",
  "variant_id": "cfg_000001_v01",
  "changed_parameter": "transform_b.translation[0]",
  "changed_value": 9.95,
  "transform_a": {
    "translation": [0.0, 0.0, 0.0],
    "rotation_xyz_deg": [0.0, 0.0, 0.0],
    "rotation_order": "XYZ"
  },
  "transform_b": {
    "translation": [9.95, 0.0, 0.0],
    "rotation_xyz_deg": [0.0, 0.0, 0.0],
    "rotation_order": "XYZ"
  },
  "assembly_script": "import cadquery as cq\n\ndef place(solid, translation, rotation_xyz_deg):\n    return solid.rotate((0,0,0), (1,0,0), rotation_xyz_deg[0]).rotate((0,0,0), (0,1,0), rotation_xyz_deg[1]).rotate((0,0,0), (0,0,1), rotation_xyz_deg[2]).translate(tuple(translation))\n\ndef assembly():\n    a = place(object_a(), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))\n    b = place(object_b(), (9.95, 0.0, 0.0), (0.0, 0.0, 0.0))\n    return a, b\n",
  "labels": {
    "volume_a": 1000.0,
    "volume_b": 1000.0,
    "intersection_volume": 5.0,
    "normalized_intersection": 0.005,
    "minimum_distance": 0.0,
    "relation": "intersecting",
    "contained": false,
    "contains_a_in_b": false,
    "contains_b_in_a": false
  },
  "diagnostics": {
    "aabb_overlap": true,
    "exact_overlap": true,
    "boolean_status": "ok",
    "distance_status": "skipped_positive_overlap",
    "label_status": "ok",
    "failure_reason": null
  },
  "difficulty_tags": ["axis_aligned", "near_boundary", "tiny_overlap"],
  "label_policy": {
    "epsilon_volume_ratio": 0.000001,
    "epsilon_distance_mm": 0.0001,
    "near_miss_threshold_mm": 1.0
  },
  "hashes": {
    "source_code_hash": null,
    "object_hash": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    "transform_hash": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
    "geometry_hash": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
    "config_hash": "sha256:5555555555555555555555555555555555555555555555555555555555555555",
    "prompt_hash": null
  },
  "metadata": {
    "generation_strategy": "boundary_targeted",
    "random_seed": 1234,
    "boundary_type": "tiny_positive_overlap",
    "artifact_ids": {
      "debug_step_id": null,
      "debug_mesh_id": null,
      "render_iso_id": null,
      "render_orthographic_ids": null,
      "overlap_render_id": null,
      "failure_reproducer_id": null
    }
  }
}
```

## 7. Public Task Row

Public task rows are released as JSONL. Every line is one task example. The keys listed in this section are always present in public rows unless a future schema version explicitly states otherwise. Nullable fields remain present with `null` values.

### Fields

| Field | Type | Required | Nullable | Meaning |
| --- | --- | --- | --- | --- |
| `id` | string | yes | no | Globally unique public row ID, e.g. `intersectionqa_binary_000001`. |
| `dataset_version` | string | yes | no | Dataset version, e.g. `v0.1`. |
| `split` | string enum | yes | no | Public split name. |
| `task_type` | string enum | yes | no | Task type. |
| `prompt` | string | yes | no | Exact model-facing prompt. Must not reveal answer labels or diagnostics. |
| `answer` | string | yes | no | Canonical final answer for `task_type`. |
| `script` | string | yes | no | Complete code shown to the model or complete assembly script used by the prompt. |
| `geometry_ids` | array of strings | yes | no | One or more internal geometry IDs used to derive the task. MVP two-object rows usually contain one ID. |
| `source` | string | yes | no | Source family for the row. Use `mixed` for mixed-source rows. |
| `generator_id` | string | yes | yes | Generator/family split group. For two-generator pairs, use a deterministic composite and store individual IDs in `metadata.generator_ids`. Null only when no generator concept exists. |
| `base_object_pair_id` | string | yes | no | Base object-pair group ID. |
| `assembly_group_id` | string | yes | no | Assembly/transform-family group ID. |
| `counterfactual_group_id` | string | yes | yes | Counterfactual group ID. Null for non-counterfactual examples. |
| `variant_id` | string | yes | yes | Counterfactual variant ID. Null for non-counterfactual examples. |
| `changed_parameter` | string | yes | yes | Single varied parameter. Null for non-counterfactual examples. |
| `changed_value` | string, number, boolean, array, object, or null | yes | yes | Changed parameter value. Null for non-counterfactual examples. |
| `labels` | object | yes | no | Raw geometry labels needed for evaluation and slicing. Same core fields as geometry `labels`. |
| `diagnostics` | object | yes | no | Computation diagnostics needed for audits and slicing. Same core fields as geometry `diagnostics`. |
| `difficulty_tags` | array of strings | yes | no | Difficulty and diagnostic subset tags. Empty array is allowed. |
| `label_policy` | `LabelPolicy` | yes | no | Label policy used to derive `labels` and `answer`. |
| `hashes` | `Hashes` | yes | no | `source_code_hash`, `geometry_hash`, `config_hash`, and `prompt_hash` must be non-null for public rows. |
| `metadata` | object | yes | no | Split metadata, prompt template version, source details, optional artifact IDs, and task-specific extras. |

For non-counterfactual examples, `counterfactual_group_id`, `variant_id`, `changed_parameter`, and `changed_value` are required keys with value `null`.

Normal public task rows should include only `diagnostics.label_status == "ok"` examples. Invalid examples may appear only in an explicit invalid-task variant or audit bundle, and must include non-null `diagnostics.failure_reason`.

### Task Answer Contracts

- `binary_interference`: `answer` is `yes` iff `labels.relation` is `intersecting` or `contained`; `answer` is `no` iff `labels.relation` is `disjoint`, `touching`, or `near_miss`.
- `relation_classification`: `answer` equals `labels.relation`.
- `volume_bucket`: `answer` is one of `0`, `(0, 0.01]`, `(0.01, 0.05]`, `(0.05, 0.20]`, `(0.20, 0.50]`, `>0.50`, derived from `labels.intersection_volume`, `labels.normalized_intersection`, and `label_policy`.

### Example: `binary_interference`

```json
{
  "id": "intersectionqa_binary_000001",
  "dataset_version": "v0.1",
  "split": "train",
  "task_type": "binary_interference",
  "prompt": "You are given two CadQuery object-construction functions and assembly transforms.\n\nAssume:\n- Units are millimetres.\n- Euler rotations are XYZ order, in degrees.\n- \"interference\" means positive-volume overlap.\n- Touching at a face, edge, or point is not interference.\n- Do not execute code.\n\nObject code:\n```python\nimport cadquery as cq\n\ndef object_a():\n    return cq.Workplane(\"XY\").box(10.0, 10.0, 10.0)\n\ndef object_b():\n    return cq.Workplane(\"XY\").box(10.0, 10.0, 10.0)\n```\n\nTransforms:\nobject_a translation [0.0, 0.0, 0.0], rotation_xyz_deg [0.0, 0.0, 0.0]\nobject_b translation [9.95, 0.0, 0.0], rotation_xyz_deg [0.0, 0.0, 0.0]\n\nQuestion: After the transforms are applied, do object_a and object_b have positive-volume interference?\n\nAnswer with exactly one token: yes or no",
  "answer": "yes",
  "script": "import cadquery as cq\n\ndef object_a():\n    return cq.Workplane(\"XY\").box(10.0, 10.0, 10.0)\n\ndef object_b():\n    return cq.Workplane(\"XY\").box(10.0, 10.0, 10.0)\n\ndef assembly():\n    return object_a().translate((0.0, 0.0, 0.0)), object_b().translate((9.95, 0.0, 0.0))\n",
  "geometry_ids": ["geom_000001"],
  "source": "synthetic",
  "generator_id": "gen_synthetic_primitives_v01",
  "base_object_pair_id": "pair_000001",
  "assembly_group_id": "asmgrp_000001",
  "counterfactual_group_id": "cfg_000001",
  "variant_id": "cfg_000001_v01",
  "changed_parameter": "transform_b.translation[0]",
  "changed_value": 9.95,
  "labels": {
    "volume_a": 1000.0,
    "volume_b": 1000.0,
    "intersection_volume": 5.0,
    "normalized_intersection": 0.005,
    "minimum_distance": 0.0,
    "relation": "intersecting",
    "contained": false,
    "contains_a_in_b": false,
    "contains_b_in_a": false
  },
  "diagnostics": {
    "aabb_overlap": true,
    "exact_overlap": true,
    "boolean_status": "ok",
    "distance_status": "skipped_positive_overlap",
    "label_status": "ok",
    "failure_reason": null
  },
  "difficulty_tags": ["axis_aligned", "near_boundary", "tiny_overlap"],
  "label_policy": {
    "epsilon_volume_ratio": 0.000001,
    "epsilon_distance_mm": 0.0001,
    "near_miss_threshold_mm": 1.0
  },
  "hashes": {
    "source_code_hash": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
    "object_hash": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    "transform_hash": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
    "geometry_hash": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
    "config_hash": "sha256:5555555555555555555555555555555555555555555555555555555555555555",
    "prompt_hash": "sha256:6666666666666666666666666666666666666666666666666666666666666666"
  },
  "metadata": {
    "prompt_template_version": "binary_interference_v01",
    "split_group": "cfg_000001",
    "generator_ids": ["gen_synthetic_primitives_v01"],
    "artifact_ids": {
      "debug_step_id": null,
      "debug_mesh_id": null,
      "render_iso_id": null,
      "render_orthographic_ids": null,
      "overlap_render_id": null,
      "failure_reproducer_id": null
    }
  }
}
```

### Example: `relation_classification`

```json
{
  "id": "intersectionqa_relation_000001",
  "dataset_version": "v0.1",
  "split": "validation",
  "task_type": "relation_classification",
  "prompt": "You are given two CadQuery object-construction functions and assembly transforms. Classify the relation after transforms as exactly one of: disjoint, touching, near_miss, intersecting, contained, invalid. Do not execute code.\n\nObject code:\n```python\nimport cadquery as cq\n\ndef object_a():\n    return cq.Workplane(\"XY\").box(10.0, 10.0, 10.0)\n\ndef object_b():\n    return cq.Workplane(\"XY\").box(10.0, 10.0, 10.0)\n```\n\nTransforms:\nobject_a translation [0.0, 0.0, 0.0], rotation_xyz_deg [0.0, 0.0, 0.0]\nobject_b translation [10.0, 0.0, 0.0], rotation_xyz_deg [0.0, 0.0, 0.0]\n\nAnswer with exactly one label.",
  "answer": "touching",
  "script": "import cadquery as cq\n\ndef object_a():\n    return cq.Workplane(\"XY\").box(10.0, 10.0, 10.0)\n\ndef object_b():\n    return cq.Workplane(\"XY\").box(10.0, 10.0, 10.0)\n\ndef assembly():\n    return object_a().translate((0.0, 0.0, 0.0)), object_b().translate((10.0, 0.0, 0.0))\n",
  "geometry_ids": ["geom_000002"],
  "source": "synthetic",
  "generator_id": "gen_synthetic_primitives_v01",
  "base_object_pair_id": "pair_000001",
  "assembly_group_id": "asmgrp_000001",
  "counterfactual_group_id": "cfg_000001",
  "variant_id": "cfg_000001_v02",
  "changed_parameter": "transform_b.translation[0]",
  "changed_value": 10.0,
  "labels": {
    "volume_a": 1000.0,
    "volume_b": 1000.0,
    "intersection_volume": 0.0,
    "normalized_intersection": 0.0,
    "minimum_distance": 0.0,
    "relation": "touching",
    "contained": false,
    "contains_a_in_b": false,
    "contains_b_in_a": false
  },
  "diagnostics": {
    "aabb_overlap": true,
    "exact_overlap": false,
    "boolean_status": "ok",
    "distance_status": "ok",
    "label_status": "ok",
    "failure_reason": null
  },
  "difficulty_tags": ["axis_aligned", "contact_vs_interference", "near_boundary"],
  "label_policy": {
    "epsilon_volume_ratio": 0.000001,
    "epsilon_distance_mm": 0.0001,
    "near_miss_threshold_mm": 1.0
  },
  "hashes": {
    "source_code_hash": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
    "object_hash": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    "transform_hash": "sha256:7777777777777777777777777777777777777777777777777777777777777777",
    "geometry_hash": "sha256:8888888888888888888888888888888888888888888888888888888888888888",
    "config_hash": "sha256:5555555555555555555555555555555555555555555555555555555555555555",
    "prompt_hash": "sha256:9999999999999999999999999999999999999999999999999999999999999999"
  },
  "metadata": {
    "prompt_template_version": "relation_classification_v01",
    "split_group": "cfg_000001",
    "generator_ids": ["gen_synthetic_primitives_v01"],
    "artifact_ids": {
      "debug_step_id": null,
      "debug_mesh_id": null,
      "render_iso_id": null,
      "render_orthographic_ids": null,
      "overlap_render_id": null,
      "failure_reproducer_id": null
    }
  }
}
```

### Example: `volume_bucket`

```json
{
  "id": "intersectionqa_volume_bucket_000001",
  "dataset_version": "v0.1",
  "split": "test_near_boundary",
  "task_type": "volume_bucket",
  "prompt": "You are given two CadQuery object-construction functions and assembly transforms. Normalized intersection means intersection_volume / min(volume(object_a), volume(object_b)). Touching without positive-volume overlap has normalized intersection 0. Do not execute code.\n\nObject code:\n```python\nimport cadquery as cq\n\ndef object_a():\n    return cq.Workplane(\"XY\").box(10.0, 10.0, 10.0)\n\ndef object_b():\n    return cq.Workplane(\"XY\").box(10.0, 10.0, 10.0)\n```\n\nTransforms:\nobject_a translation [0.0, 0.0, 0.0], rotation_xyz_deg [0.0, 0.0, 0.0]\nobject_b translation [9.95, 0.0, 0.0], rotation_xyz_deg [0.0, 0.0, 0.0]\n\nWhich bucket contains the normalized intersection volume?\nAllowed answers:\n0\n(0, 0.01]\n(0.01, 0.05]\n(0.05, 0.20]\n(0.20, 0.50]\n>0.50\n\nAnswer with exactly one bucket string.",
  "answer": "(0, 0.01]",
  "script": "import cadquery as cq\n\ndef object_a():\n    return cq.Workplane(\"XY\").box(10.0, 10.0, 10.0)\n\ndef object_b():\n    return cq.Workplane(\"XY\").box(10.0, 10.0, 10.0)\n\ndef assembly():\n    return object_a().translate((0.0, 0.0, 0.0)), object_b().translate((9.95, 0.0, 0.0))\n",
  "geometry_ids": ["geom_000001"],
  "source": "synthetic",
  "generator_id": "gen_synthetic_primitives_v01",
  "base_object_pair_id": "pair_000001",
  "assembly_group_id": "asmgrp_000001",
  "counterfactual_group_id": "cfg_000001",
  "variant_id": "cfg_000001_v01",
  "changed_parameter": "transform_b.translation[0]",
  "changed_value": 9.95,
  "labels": {
    "volume_a": 1000.0,
    "volume_b": 1000.0,
    "intersection_volume": 5.0,
    "normalized_intersection": 0.005,
    "minimum_distance": 0.0,
    "relation": "intersecting",
    "contained": false,
    "contains_a_in_b": false,
    "contains_b_in_a": false
  },
  "diagnostics": {
    "aabb_overlap": true,
    "exact_overlap": true,
    "boolean_status": "ok",
    "distance_status": "skipped_positive_overlap",
    "label_status": "ok",
    "failure_reason": null
  },
  "difficulty_tags": ["axis_aligned", "near_boundary", "tiny_overlap"],
  "label_policy": {
    "epsilon_volume_ratio": 0.000001,
    "epsilon_distance_mm": 0.0001,
    "near_miss_threshold_mm": 1.0
  },
  "hashes": {
    "source_code_hash": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
    "object_hash": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    "transform_hash": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
    "geometry_hash": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
    "config_hash": "sha256:5555555555555555555555555555555555555555555555555555555555555555",
    "prompt_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "metadata": {
    "prompt_template_version": "volume_bucket_v01",
    "split_group": "cfg_000001",
    "generator_ids": ["gen_synthetic_primitives_v01"],
    "volume_bucket_boundaries": ["0", "(0, 0.01]", "(0.01, 0.05]", "(0.05, 0.20]", "(0.20, 0.50]", ">0.50"],
    "artifact_ids": {
      "debug_step_id": null,
      "debug_mesh_id": null,
      "render_iso_id": null,
      "render_orthographic_ids": null,
      "overlap_render_id": null,
      "failure_reproducer_id": null
    }
  }
}
```

## 8. Dataset Metadata File

The public export includes `metadata.json` at the dataset root. It summarizes the complete export and records the policy needed to reproduce labels and splits.

### Fields

| Field | Type | Required | Nullable | Meaning |
| --- | --- | --- | --- | --- |
| `dataset_name` | string | yes | no | Must be `"IntersectionQA"`. |
| `dataset_version` | string | yes | no | Dataset release version, e.g. `v0.1`. |
| `created_from_commit` | string | yes | no | Git commit used for generation/export. |
| `config_hash` | string | yes | no | Hash of the complete generation/export config. |
| `source_manifest_hash` | string | yes | no | Hash of the source manifest used for object ingestion. |
| `label_policy` | `LabelPolicy` | yes | no | Default label policy for the dataset version. |
| `splits` | object | yes | no | Map from split name to split summary. |
| `task_types` | array of strings | yes | no | Task types included in this export. |
| `counts` | object | yes | no | Dataset counts by row, task, split, relation, source, and failure category. |
| `cadquery_version` | string | yes | yes | CadQuery version used for labels. Null only if no CadQuery labels were generated. |
| `ocp_version` | string | yes | yes | OCP/OpenCASCADE version used for labels. |
| `license` | string | yes | no | Dataset release license. Source-specific license details should also appear in source metadata or dataset card. |
| `known_limitations` | array of strings | yes | no | Known limitations for this release. Empty array only after explicit review. |

### Example

```json
{
  "dataset_name": "IntersectionQA",
  "dataset_version": "v0.1",
  "created_from_commit": "0123456789abcdef0123456789abcdef01234567",
  "config_hash": "sha256:5555555555555555555555555555555555555555555555555555555555555555",
  "source_manifest_hash": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "label_policy": {
    "epsilon_volume_ratio": 0.000001,
    "epsilon_distance_mm": 0.0001,
    "near_miss_threshold_mm": 1.0
  },
  "splits": {
    "train": {
      "path": "train.jsonl",
      "row_count": 70000,
      "task_counts": {
        "binary_interference": 40000,
        "relation_classification": 15000,
        "volume_bucket": 15000
      },
      "holdout_rule": "training_split"
    },
    "validation": {
      "path": "validation.jsonl",
      "row_count": 10000,
      "task_counts": {
        "binary_interference": 6000,
        "relation_classification": 2000,
        "volume_bucket": 2000
      },
      "holdout_rule": "group_safe_validation"
    }
  },
  "task_types": ["binary_interference", "relation_classification", "volume_bucket"],
  "counts": {
    "total_rows": 100000,
    "source_objects": 5000,
    "validated_objects": 4800,
    "geometry_records": 50000,
    "failure_records": 200,
    "by_relation": {
      "disjoint": 30000,
      "touching": 15000,
      "near_miss": 15000,
      "intersecting": 35000,
      "contained": 5000,
      "invalid": 0
    }
  },
  "cadquery_version": "2.5.2",
  "ocp_version": "7.8.1",
  "license": "cc-by-4.0",
  "known_limitations": [
    "v0.1 is code-only and does not include default render artifacts.",
    "Relations depend on the recorded CadQuery/OCP version and label policy.",
    "Containment coverage depends on whether containment detection was enabled for a source subset."
  ]
}
```

## 9. Split Manifest

The split manifest records split construction and leakage-audit inputs. It may be embedded in `metadata.json` under `splits` for small releases or stored as a separate internal/public audit file such as `split_manifest.json`.

### Fields

| Field | Type | Required | Nullable | Meaning |
| --- | --- | --- | --- | --- |
| `dataset_version` | string | yes | no | Dataset version audited. |
| `split_names` | array of strings | yes | no | Split names included in the manifest. |
| `splits` | object | yes | no | Map from split name to split details. |
| `group_holdout_rules` | array of objects | yes | no | Rules applied to prevent leakage. |
| `leakage_audit` | object | yes | no | Audit status, checked groups, and violations. |

Each `splits.<split_name>` object must contain:

| Field | Type | Required | Nullable | Meaning |
| --- | --- | --- | --- | --- |
| `row_count` | integer | yes | no | Number of public rows in the split. |
| `task_counts` | object | yes | no | Row counts by `task_type`. |
| `label_distributions` | object | yes | no | Counts by relation, binary answer, and volume bucket where applicable. |
| `generator_ids` | array of strings | yes | no | Generator/family IDs present in the split. For mixed-generator rows, include individual generator IDs from `metadata.generator_ids`, not only the top-level composite `generator_id`. |
| `base_object_pair_ids` | array of strings | yes | no | Base object-pair IDs present in the split. |
| `assembly_group_ids` | array of strings | yes | no | Assembly group IDs present in the split. |
| `counterfactual_group_ids` | array of strings | yes | no | Counterfactual group IDs present in the split. Empty array if none. |
| `group_holdout_rule_ids` | array of strings | yes | no | Rules that apply to this split. |

Each `group_holdout_rules[]` object must contain:

| Field | Type | Required | Nullable | Meaning |
| --- | --- | --- | --- | --- |
| `rule_id` | string | yes | no | Stable rule ID. |
| `description` | string | yes | no | Human-readable rule. |
| `group_fields` | array of strings | yes | no | Group fields governed by the rule. |
| `forbidden_cross_split_pairs` | array of arrays | yes | no | Split pairs across which the group fields must not overlap. |
| `status` | string enum | yes | no | `pass`, `fail`, or `not_run`. |

`leakage_audit.status` must be `pass`, `fail`, or `not_run`. A passing audit requires no shared forbidden `generator_id`, `base_object_pair_id`, `assembly_group_id`, or `counterfactual_group_id` across the relevant split boundaries.

### Example

```json
{
  "dataset_version": "v0.1",
  "split_names": ["train", "validation", "test_random", "test_generator_heldout", "test_object_pair_heldout", "test_near_boundary"],
  "splits": {
    "train": {
      "row_count": 70000,
      "task_counts": {
        "binary_interference": 40000,
        "relation_classification": 15000,
        "volume_bucket": 15000
      },
      "label_distributions": {
        "relation": {
          "disjoint": 21000,
          "touching": 10500,
          "near_miss": 10500,
          "intersecting": 24500,
          "contained": 3500
        },
        "binary_answer": {
          "yes": 28000,
          "no": 42000
        },
        "volume_bucket": {
          "0": 42000,
          "(0, 0.01]": 7000,
          "(0.01, 0.05]": 7000,
          "(0.05, 0.20]": 7000,
          "(0.20, 0.50]": 5000,
          ">0.50": 2000
        }
      },
      "generator_ids": ["gen_synthetic_primitives_v01"],
      "base_object_pair_ids": ["pair_000001", "pair_000002"],
      "assembly_group_ids": ["asmgrp_000001", "asmgrp_000002"],
      "counterfactual_group_ids": ["cfg_000001", "cfg_000002"],
      "group_holdout_rule_ids": ["counterfactual_inseparable", "object_pair_holdout"]
    }
  },
  "group_holdout_rules": [
    {
      "rule_id": "counterfactual_inseparable",
      "description": "Rows sharing counterfactual_group_id must not cross train, validation, or test splits.",
      "group_fields": ["counterfactual_group_id"],
      "forbidden_cross_split_pairs": [["train", "validation"], ["train", "test_random"], ["train", "test_near_boundary"]],
      "status": "pass"
    },
    {
      "rule_id": "object_pair_holdout",
      "description": "Rows sharing base_object_pair_id or assembly_group_id must not cross held-out object-pair tests.",
      "group_fields": ["base_object_pair_id", "assembly_group_id"],
      "forbidden_cross_split_pairs": [["train", "test_object_pair_heldout"], ["validation", "test_object_pair_heldout"]],
      "status": "pass"
    }
  ],
  "leakage_audit": {
    "status": "pass",
    "audited_at": "2026-04-24T00:00:00Z",
    "checked_group_fields": ["generator_id", "base_object_pair_id", "assembly_group_id", "counterfactual_group_id"],
    "violation_count": 0,
    "violations": []
  }
}
```

## 10. Failure Manifest

Failure manifests record skipped or failed source objects and geometry jobs. Failures must not be silently dropped.

### Fields

| Field | Type | Required | Nullable | Meaning |
| --- | --- | --- | --- | --- |
| `failure_id` | string | yes | no | Stable failure ID, e.g. `fail_000001`. |
| `stage` | string enum | yes | no | Pipeline stage where the failure occurred. |
| `source` | string | yes | yes | Source family, if known. |
| `source_id` | string | yes | yes | Source ID, if known. |
| `object_id` | string | yes | yes | Object ID, if the failure is object-specific. |
| `geometry_id` | string | yes | yes | Geometry ID, if the failure is geometry-job-specific. |
| `failure_reason` | string enum | yes | no | Machine-readable reason from the shared `failure_reason` enum. |
| `error_summary` | string | yes | no | Short sanitized error summary. Do not include unbounded stack traces in public manifests. |
| `retry_count` | integer | yes | no | Number of retries attempted before the failure was accepted. |
| `hashes` | `Hashes` | yes | no | Applicable hashes for reproducing and deduplicating the failure. |

`stage` enum:

```text
source_loading
source_normalization
object_validation
assembly_generation
geometry_labeling
task_materialization
export_validation
```

### Example

```json
{
  "failure_id": "fail_000001",
  "stage": "object_validation",
  "source": "cadevolve",
  "source_id": "./CADEvolve-C/CADEvolve-C-core/example_000123.py",
  "object_id": "obj_001234",
  "geometry_id": null,
  "failure_reason": "missing_result_object",
  "error_summary": "Worker completed without producing result, shape, solid, or part.",
  "retry_count": 1,
  "hashes": {
    "source_code_hash": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "object_hash": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    "transform_hash": null,
    "geometry_hash": null,
    "config_hash": "sha256:5555555555555555555555555555555555555555555555555555555555555555",
    "prompt_hash": null
  }
}
```

## 11. ID Conventions

Stable IDs are opaque identifiers with deterministic assignment. Do not encode local filesystem paths or wall-clock timestamps in IDs.

Recommended patterns:

| Entity | Pattern | Example |
| --- | --- | --- |
| Source object | `obj_<6 digits>` | `obj_000001` |
| Base object pair | `pair_<6 digits>` | `pair_000001` |
| Assembly group | `asmgrp_<6 digits>` | `asmgrp_000001` |
| Counterfactual group | `cfg_<6 digits>` | `cfg_000001` |
| Counterfactual variant | `cfg_<6 digits>_v<2 digits>` | `cfg_000001_v01` |
| Geometry label record | `geom_<6 digits>` | `geom_000001` |
| Binary task row | `intersectionqa_binary_<6 digits>` | `intersectionqa_binary_000001` |
| Relation task row | `intersectionqa_relation_<6 digits>` | `intersectionqa_relation_000001` |
| Volume-bucket task row | `intersectionqa_volume_bucket_<6 digits>` | `intersectionqa_volume_bucket_000001` |
| Failure record | `fail_<6 digits>` | `fail_000001` |

Final IDs must be deterministic and must not depend on parallel job completion order. Implementations should assign IDs after sorting canonical content keys, or derive stable content IDs before assigning release counters.

Canonical ordering inputs should include, where applicable:

- `source`
- `source_id`
- normalized source code hash
- object IDs
- canonical base object-pair key
- assembly group key
- counterfactual group key and variant value
- canonical transform JSON
- label policy
- task type
- prompt template version

`base_object_pair_id` groups the same underlying object pair for split audits. It should be order-insensitive unless an experiment explicitly depends on object role order. `assembly_group_id` may be order-sensitive because object roles and transform families can matter.

## 12. Export Layout

Default public export layout:

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

`schema.json` is the machine-readable JSON Schema generated from the implementation models that follow this specification. It should be versioned with the dataset export.

Optional files may be added without changing the default layout:

```text
data/intersectionqa_v0_1/split_manifest.json
data/intersectionqa_v0_1/failure_manifest.jsonl
data/intersectionqa_v0_1/artifact_manifest.json
data/intersectionqa_v0_1/test_topology_heldout.jsonl
data/intersectionqa_v0_1/test_operation_heldout.jsonl
```

Optional artifact bundles, if produced, must be separate from the default JSONL rows and linked by `ArtifactIds`.

## 13. Validation Requirements

Validators must run before public export and should be usable on each JSONL shard independently.

Schema-level checks:

- Required fields are present in every record for its record type.
- Public rows have non-empty `id`, `dataset_version`, `split`, `task_type`, `prompt`, `answer`, `script`, `geometry_ids`, `source`, `base_object_pair_id`, `assembly_group_id`, `labels`, `diagnostics`, `label_policy`, `hashes`, and `metadata`.
- Public row `geometry_ids` is a non-empty array of strings.
- Nullable counterfactual fields are consistent: if `counterfactual_group_id` is null, then `variant_id`, `changed_parameter`, and `changed_value` must also be null; if it is non-null, then `variant_id` and `changed_parameter` must be non-null.
- Answers match labels for the task type.
- `binary_interference` answers are `yes` for `intersecting` or `contained`, and `no` for `disjoint`, `touching`, or `near_miss`.
- `relation_classification` answers equal `labels.relation`.
- `volume_bucket` answers match the official bucket boundaries from `label_rules.md`.
- Relations match `label_policy` using the official precedence: invalid, contained, intersecting, touching, near_miss, disjoint.
- `labels.normalized_intersection` equals `labels.intersection_volume / min(labels.volume_a, labels.volume_b)` within configured numeric tolerance for valid positive-volume records.
- `labels.normalized_intersection` is in `[0.0, 1.0 + epsilon_volume_ratio]` for valid records.
- `labels.intersection_volume` and `labels.minimum_distance` are finite and non-negative whenever non-null.
- `diagnostics.exact_overlap` equals `labels.intersection_volume > epsilon_volume` when both fields are available.
- Positive policy overlap does not coexist with positive clearance: when `intersection_volume > epsilon_volume`, `minimum_distance` must be `0.0`, null with `distance_status == "skipped_positive_overlap"`, or within `epsilon_distance_mm`.
- `touching` records have `intersection_volume <= epsilon_volume` and `minimum_distance <= epsilon_distance_mm`.
- `near_miss` records have `intersection_volume <= epsilon_volume` and `epsilon_distance_mm < minimum_distance <= near_miss_threshold_mm`.
- `disjoint` records have `intersection_volume <= epsilon_volume` and `minimum_distance > near_miss_threshold_mm`.
- `contained` records have at least one containment flag true, positive policy overlap, and intersection volume consistent with the smaller solid volume within tolerance.
- Invalid examples include non-null `failure_reason`.
- Normal public task rows either exclude invalid labels or use an explicit invalid-task policy.
- Split-sensitive groups do not cross forbidden splits. Audits must check `generator_id`, individual generator IDs from `metadata.generator_ids`, `base_object_pair_id`, `assembly_group_id`, and `counterfactual_group_id`.
- Every public row has dataset version, task type, prompt, answer, split, and provenance.
- Public rows do not contain local absolute paths or required external artifact paths.
- Public rows do not require CadQuery execution to read labels, diagnostics, answer, or split metadata.
- Hash fields that are required for a record type are non-null and match the canonical content used by the exporter.
