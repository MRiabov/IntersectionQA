# IntersectionEdit Task Specification Draft

This is a draft seed specification for the IntersectionEdit task family. The
first implemented vertical slice is `repair_direction`; it is intentionally
conservative and is not part of the frozen IntersectionQA v0.1 release
contract.

## Purpose

IntersectionEdit should test whether a model can propose geometry-preserving or
constraint-satisfying edits for CAD assemblies after diagnosing an intersection,
clearance, or fit problem. The task family moves from answering what is wrong to
choosing an action that repairs or improves the assembly.

## Relationship to IntersectionQA

IntersectionEdit should reuse the same source objects, CadQuery execution,
assembly transforms, raw geometry labels, provenance records, split grouping,
and verifier infrastructure used by IntersectionQA wherever possible.
IntersectionQA remains the canonical v0.1 task family; Edit semantics are
separate unless explicitly stated.

## Likely Task Families

- Repair direction: choose the axis or signed direction that reduces or removes
  an interference.
- Minimal translation repair: predict a translation vector or constrained move
  that clears positive-volume overlap.
- Clearance restoration: adjust placement to satisfy a target clearance margin.
- Pairwise and multi-object edit tasks: repair one violating pair without
  introducing new violations among related objects.

## Output and Action Schema Direction

Initial outputs should be simple, auditable, and verifier-friendly. Candidate
schemas include:

- discrete labels such as `+x`, `-x`, `+y`, `-y`, `+z`, `-z`, or `no_valid_move`;
- numeric translation vectors in millimetres;
- constrained action records with an object ID, action type, axis, magnitude,
  and optional target clearance.

The first schema should prefer deterministic parsing and bounded numeric ranges
over broad natural-language repair descriptions.

## Verifier and Reward Contract

An Edit answer is valid only if the verifier can apply the proposed action to
the same geometry record and measure the result. Rewards or scores should be
derived from exact geometry checks, for example overlap removal, clearance
restoration, action magnitude, and whether new collisions are introduced.
Heuristic or AABB-only checks may be diagnostic baselines, not official labels.

## Splits and Leakage

IntersectionEdit must reuse group-safe splitting. Source groups, object-pair
groups, assembly groups, and counterfactual groups must remain intact across
splits. Edit rows derived from a QA geometry record must not leak a held-out
geometry family through another task family.

## Implemented First Slice: `repair_direction`

The initial implemented slice materializes `repair_direction` rows only from
two-object geometry records whose stored relation is `intersecting` or
`contained`. `object_a` is fixed and `object_b` is the movable object. The model
must answer with exactly one signed world-axis direction: `+x`, `-x`, `+y`,
`-y`, `+z`, or `-z`.

The current label policy is
`conservative_aabb_separating_translation_v01`. It uses stored world-space
`bbox_a` and `bbox_b` metadata, computes the single-axis translation needed to
make the AABBs disjoint with `label_policy.epsilon_distance_mm` clearance for
each of the six signed directions, and selects the smallest absolute movement.
Ties are resolved deterministically in this order: `+x`, `-x`, `+y`, `-y`,
`+z`, `-z`.

Public row metadata stores the selected direction, movement magnitude,
translation vector, policy name, tie-break order, and all candidate moves. The
prompt shows object code and transforms but must not reveal labels, exact
volumes, diagnostics, selected answer, or candidate magnitudes.

This policy is a conservative AABB-separating first slice, not a claim of exact
minimal CAD repair. Verification can apply the selected translation to
`object_b` and remeasure exact geometry with the existing CadQuery path.

## Future Implementation Targets

Future slices may add exact minimal translation repair, clearance restoration,
multi-object edit constraints, or `no_valid_move` semantics where the verifier
has a concrete reason to emit that label. Avoid package or repository renaming
until these slices prove the shared-core boundary.
