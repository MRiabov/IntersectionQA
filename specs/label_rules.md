# IntersectionQA Label Rules

Note: This specification describes IntersectionQA v0.1. Shared geometry, split,
provenance, and verifier concepts may be reused by IntersectionEdit, but task
semantics here are QA-specific unless stated otherwise.

## 1. Purpose

This file defines the official geometry label semantics and threshold policy for IntersectionQA. Public task answers must be derived from raw CadQuery/OpenCASCADE geometry fields using these rules, not from prompt text, heuristics, or task-specific recomputation.

Raw geometry values must always be stored alongside derived labels. This allows validation, threshold audits, future relabeling, and task materialization from one geometry record.

## 2. Raw Geometry Inputs

Each two-solid geometry label record must contain these fields. Numeric distances are in millimetres; volumes are in cubic millimetres.

| Field | Meaning |
| --- | --- |
| `volume_a` | Finite positive volume of transformed solid A. |
| `volume_b` | Finite positive volume of transformed solid B. |
| `intersection_volume` | Volume of the exact Boolean intersection. Use `0.0` when exact non-overlap is proven. |
| `normalized_intersection` | `intersection_volume / min(volume_a, volume_b)` for valid positive volumes. |
| `minimum_distance` | Exact minimum solid-to-solid clearance. For positive-overlap records this may be stored as `0.0`. |
| `contains_a_in_b` | `true` if all material of A is contained in B. `false` if checked and not contained. `null` only when containment detection is disabled. |
| `contains_b_in_a` | `true` if all material of B is contained in A. `false` if checked and not contained. `null` only when containment detection is disabled. |
| `aabb_overlap` | Closed-axis-aligned bounding-box overlap diagnostic after transforms. Touching AABBs count as overlap. |
| `exact_overlap` | Derived diagnostic for positive-volume overlap: `intersection_volume > epsilon_volume`. |
| `boolean_status` | Status of intersection computation: `ok`, `skipped_aabb_disjoint`, `failed`, or `not_run`. |
| `distance_status` | Status of distance computation: `ok`, `skipped_positive_overlap`, `failed`, or `not_run`. |
| `label_status` | `ok` when all fields needed by the final label are valid; otherwise `invalid`. |
| `failure_reason` | `null` for valid labels; otherwise one failure category from Section 10. |

`skipped_aabb_disjoint` is valid only when transformed AABBs are strictly separated, so `intersection_volume = 0.0` and `exact_overlap = false` are mathematically implied. Relation labels that distinguish touching, near-miss, and disjoint still require a valid `minimum_distance`.

## 3. Default Threshold Policy

Provisional default thresholds:

```text
epsilon_volume_ratio = 1e-6
epsilon_volume = epsilon_volume_ratio * min(volume_a, volume_b)
epsilon_distance_mm = 1e-4
near_miss_threshold_mm = 1.0
```

Thresholds are part of the label policy. Each dataset version must store them in metadata and include them in the config hash or equivalent reproducibility hash. Changing any threshold requires a new dataset version or an explicit relabeling audit.

## 4. Relation Labels

### `invalid`

Semantic meaning: the geometry, raw measurement, or label derivation failed or produced inconsistent fields.

Required conditions: `label_status = invalid`, or a required raw field/status is missing, failed, non-finite, or contradictory.

Binary interference: excluded from normal binary tasks.

Common edge cases: source scripts that do not execute, non-solid CadQuery results, zero-volume solids, Boolean failures, distance failures for non-overlap cases, worker timeouts, and validation contradictions.

### `contained`

Semantic meaning: all material of one solid is inside the other solid. This is a positive-volume overlap subtype, not a clearance or fit-in-cavity label.

Required conditions: `label_status = ok`; containment detection is implemented; `contains_a_in_b = true` or `contains_b_in_a = true`; `intersection_volume > epsilon_volume`; and the intersection volume is consistent with the smaller solid volume within threshold tolerance.

Binary interference: `yes`.

Common edge cases: if containment detection is disabled, this label is not emitted and positive-volume contained cases fall through to `intersecting`. Identical or coincident solids may set both containment flags; assign `contained` because containment has precedence. An object sitting inside an empty cavity with no material overlap is not `contained`; classify it by distance.

### `intersecting`

Semantic meaning: the solids have positive material overlap but neither containment flag has precedence.

Required conditions: `label_status = ok`; not `contained`; `intersection_volume > epsilon_volume`; `exact_overlap = true`.

Binary interference: `yes`.

Common edge cases: tiny sliver intersections at or below `epsilon_volume` are not `intersecting`; they are classified by distance. Face, edge, or point contact alone is not `intersecting`.

### `touching`

Semantic meaning: the solids have zero policy-positive volume overlap and are in contact within distance tolerance.

Required conditions: `label_status = ok`; `intersection_volume <= epsilon_volume`; `distance_status = ok`; `minimum_distance <= epsilon_distance_mm`.

Binary interference: `no`.

Common edge cases: face-sharing boxes, edge contact, vertex contact, or near-coincident surfaces with no positive-volume overlap. Kernel noise may produce a tiny intersection volume below `epsilon_volume`; distance then decides whether this is `touching`.

### `near_miss`

Semantic meaning: the solids do not overlap or touch, but the clearance is small enough to be a near-boundary case.

Required conditions: `label_status = ok`; `intersection_volume <= epsilon_volume`; `distance_status = ok`; `epsilon_distance_mm < minimum_distance <= near_miss_threshold_mm`.

Binary interference: `no`.

Common edge cases: AABBs may overlap for concave or hollow objects even when exact solids have positive clearance. Near-miss is based on exact solid distance, not AABB distance.

### `disjoint`

Semantic meaning: the solids have no positive-volume overlap and are separated by more than the near-miss threshold.

Required conditions: `label_status = ok`; `intersection_volume <= epsilon_volume`; `distance_status = ok`; `minimum_distance > near_miss_threshold_mm`.

Binary interference: `no`.

Common edge cases: AABBs may be separated or may overlap due to cavities/concavity. The label is still `disjoint` when exact solid clearance exceeds the threshold.

## 5. Relation Precedence

Derive `relation` in this exact order:

1. If geometry or label computation failed: `invalid`
2. Else if containment is implemented and one solid is fully inside the other: `contained`
3. Else if `intersection_volume > epsilon_volume`: `intersecting`
4. Else if `minimum_distance <= epsilon_distance_mm`: `touching`
5. Else if `minimum_distance <= near_miss_threshold_mm`: `near_miss`
6. Else: `disjoint`

`contained` also has positive-volume overlap: the intersection volume should equal the smaller solid's volume within threshold tolerance. Binary tasks must treat `contained` as interference (`yes`). Relation tasks keep it separate from ordinary `intersecting`.

## 6. Binary Interference Rule

Binary interference answers are derived only from `relation`:

| Relation | Binary answer |
| --- | --- |
| `contained` | `yes` |
| `intersecting` | `yes` |
| `disjoint` | `no` |
| `touching` | `no` |
| `near_miss` | `no` |
| `invalid` | excluded |

Invalid examples are excluded from normal binary tasks. They may appear only in an explicit invalid-task variant with its own answer schema.

## 7. Normalized Intersection

Definition:

```text
normalized_intersection = intersection_volume / min(volume_a, volume_b)
```

For valid positive-volume solids, the expected range is `0.0 <= normalized_intersection <= 1.0`, allowing only small numerical tolerance above `1.0` up to `1.0 + epsilon_volume_ratio`.

If either volume is zero, negative, missing, or non-finite, `normalized_intersection` is invalid and `failure_reason` must be `zero_or_negative_volume` or the more specific upstream failure. Do not clip stored raw values. Values outside the accepted range are validation failures; they must be rejected or marked `invalid`, not silently clipped.

## 8. Volume Buckets

Volume bucket tasks use the normalized intersection policy value. If `intersection_volume <= epsilon_volume`, assign bucket `0` even if the raw ratio is a tiny positive numerical artifact.

Official buckets and boundaries:

| Bucket | Condition |
| --- | --- |
| `0` | `intersection_volume <= epsilon_volume` |
| `(0, 0.01]` | `intersection_volume > epsilon_volume` and `0 < normalized_intersection <= 0.01` |
| `(0.01, 0.05]` | `0.01 < normalized_intersection <= 0.05` |
| `(0.05, 0.20]` | `0.05 < normalized_intersection <= 0.20` |
| `(0.20, 0.50]` | `0.20 < normalized_intersection <= 0.50` |
| `>0.50` | `normalized_intersection > 0.50` |

All nonzero buckets are open on the lower bound and closed on the upper bound. Invalid records have no volume bucket.

## 9. Clearance Buckets

Clearance bucket tasks use exact `minimum_distance` and apply only to non-intersecting examples unless a task explicitly defines an `intersecting` answer.

Provisional buckets:

| Bucket | Condition |
| --- | --- |
| `touching` | `intersection_volume <= epsilon_volume` and `minimum_distance <= epsilon_distance_mm` |
| `(0, 1]` | `epsilon_distance_mm < minimum_distance <= 1.0` |
| `(1, 5]` | `1.0 < minimum_distance <= 5.0` |
| `(5, 20]` | `5.0 < minimum_distance <= 20.0` |
| `>20` | `minimum_distance > 20.0` |

For `contained` and `intersecting`, do not assign a clearance bucket in the standard clearance task. A separate task may add an explicit `intersecting` answer if it documents that schema.

## 10. Invalid and Failure Handling

Failure reasons use this enum:

| Failure reason | Meaning | Default handling |
| --- | --- | --- |
| `source_parse_error` | Source code cannot be parsed. | Exclude; store in failure diagnostics. |
| `source_exec_error` | Source code raises during execution. | Exclude; store in failure diagnostics. |
| `missing_result_object` | No expected CadQuery result object is produced. | Exclude; store in failure diagnostics. |
| `invalid_cadquery_type` | Result is not a supported CadQuery/OpenCASCADE type. | Exclude; store in failure diagnostics. |
| `non_solid_result` | Result is not a solid or valid compound of solids for the task. | Exclude; store in failure diagnostics. |
| `zero_or_negative_volume` | A required solid volume is zero, negative, or non-finite. | Exclude; store in failure diagnostics. |
| `non_finite_bbox` | Bounding box has NaN, infinite, or missing coordinates. | Exclude; store in failure diagnostics. |
| `boolean_intersection_failed` | Exact intersection computation failed when required. | Exclude; store reproducer diagnostics if available. |
| `distance_query_failed` | Minimum-distance query failed when required. | Exclude; store reproducer diagnostics if available. |
| `timeout` | Worker exceeded the configured timeout. | Exclude; store diagnostics. |
| `worker_crash` | Isolated worker exited unexpectedly. | Exclude; store diagnostics. |
| `unknown_error` | Failure did not match a known category. | Exclude; store diagnostics and audit. |

Normal public task rows should include only `label_status = ok` examples. Invalid records may be kept in internal manifests, failure reports, audit bundles, or explicit invalid-task datasets.

## 11. Label Consistency Checks

Validation must enforce these checks before export:

- `volume_a` and `volume_b` are finite and positive for all `label_status = ok` records.
- `intersection_volume` and `minimum_distance` are finite and non-negative for all labels that use them.
- Positive overlap cannot coexist with positive clearance: `intersection_volume > epsilon_volume` implies `minimum_distance <= epsilon_distance_mm` or `distance_status = skipped_positive_overlap`.
- `normalized_intersection` is within the accepted range from Section 7.
- `exact_overlap` equals `intersection_volume > epsilon_volume`.
- `touching` has no policy-positive volume overlap.
- `near_miss` has `epsilon_distance_mm < minimum_distance <= near_miss_threshold_mm`.
- `disjoint` has `minimum_distance > near_miss_threshold_mm`.
- `contained` has a containment flag and positive-volume overlap consistent with the smaller solid volume.
- `invalid` includes a non-null `failure_reason`.
- Binary answer matches the relation label exactly as specified in Section 6.

## 12. Golden Test Cases

Use axis-aligned boxes in millimetres. Let box A span `[0, 10]` on X, Y, and Z unless otherwise stated.

| Case | Geometry | Expected label details |
| --- | --- | --- |
| Two separated boxes | B spans X `[20, 30]`, Y/Z `[0, 10]`. | `intersection_volume = 0`, `minimum_distance = 10`, `relation = disjoint`, binary `no`, volume bucket `0`, clearance bucket `(5, 20]`. |
| Two boxes sharing a face | B spans X `[10, 20]`, Y/Z `[0, 10]`. | `intersection_volume = 0`, `minimum_distance = 0`, `relation = touching`, binary `no`, volume bucket `0`, clearance bucket `touching`. |
| Two boxes overlapping by a small positive amount | B spans X `[9.99, 19.99]`, Y/Z `[0, 10]`. | `intersection_volume = 1.0`, `normalized_intersection = 0.001`, `relation = intersecting`, binary `yes`, volume bucket `(0, 0.01]`. |
| One box fully inside another | A spans `[0, 10]` on all axes; B spans `[2, 8]` on all axes. | `contains_b_in_a = true`, `intersection_volume = volume_b = 216`, `normalized_intersection = 1.0`, `relation = contained`, binary `yes`, volume bucket `>0.50`. |
| Near-miss boxes with 0.5 mm clearance | B spans X `[10.5, 20.5]`, Y/Z `[0, 10]`. | `intersection_volume = 0`, `minimum_distance = 0.5`, `relation = near_miss`, binary `no`, volume bucket `0`, clearance bucket `(0, 1]`. |
| Invalid zero-volume or failed object | Any object has zero/non-finite volume, returns a non-solid, or fails execution. | `relation = invalid`, `label_status = invalid`, non-null `failure_reason`, excluded from normal binary/relation/bucket tasks. |
