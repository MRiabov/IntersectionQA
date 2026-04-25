# IntersectionEdit Task Specification Draft

This is a draft seed specification for the IntersectionEdit task family. The
first implemented slices are `repair_direction` and `repair_translation`; they
are intentionally conservative and are not part of the frozen IntersectionQA
v0.1 release contract.

## Purpose

IntersectionEdit should test whether a model can propose geometry-preserving or
constraint-satisfying edits for CAD assemblies after diagnosing an intersection,
clearance, or fit problem. The task family moves from answering what is wrong to
choosing an action that repairs or improves the assembly.

## Naming and Framing

The task family name is `IntersectionEdit`. It is a sibling task family built on
the IntersectionQA geometry, provenance, split, export, and verifier core. The
paper framing should describe IntersectionQA as diagnostic spatial reasoning
from CAD code, and IntersectionEdit as verified edit prediction: a model must
produce an auditable action or constrained choice that can be applied to the
same CAD assembly and checked by exact geometry measurement.

IntersectionEdit rows are not free-form design tasks. The implemented public
rows use constrained answer formats, target metadata, and verifier-oriented
structured answers so distance errors, movement overshoot, target satisfaction,
and candidate ranking quality can be scored consistently.

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

## Implemented Repair Slices

The initial implemented slices materialize repair rows only from two-object
geometry records whose stored relation is `intersecting` or `contained`.
`object_a` is fixed and `object_b` is the movable object.

`repair_direction` asks for exactly one signed world-axis direction: `+x`,
`-x`, `+y`, `-y`, `+z`, or `-z`.

`repair_translation` asks for the signed world-axis direction plus movement
magnitude under the same policy. The answer format is exactly
`<direction> <magnitude_mm>` with six digits after the decimal point, for
example `+x 0.500100`.

`axis_aligned_repair_vector` asks for the verified exact repair as a full
world-coordinate translation vector with one decimal per component, for example
`dx=1.2, dy=0.0, dz=0.0`.

`axis_aligned_repair_program` asks for the same verified exact repair as a
single CadQuery edit statement, for example
`object_b = object_b.translate((1.2, 0.0, 0.0))`.

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
Release-candidate builds must fail if stored edit rows do not pass the exact
repair verifier. Verifier execution is trusted-code only because CadQuery row
reconstruction executes the row script.

## Future Implementation Targets

Future slices may add exact minimal CAD-kernel translation repair, clearance
restoration, multi-object edit constraints, or `no_valid_move` semantics where
the verifier has a concrete reason to emit that label. Avoid package or
repository renaming until these slices prove the shared-core boundary.
