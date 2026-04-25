# IntersectionEdit Task Specification Draft

This is a draft seed specification for a future IntersectionEdit task family.
It is not implemented in IntersectionQA v0.1 and should not be treated as a
release contract.

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

## Initial Implementation Target

Start with one vertical slice, probably `minimal_translation_repair` or
`repair_direction`, using existing two-object geometry records and exact
verifier checks. Avoid package or repository renaming until this slice proves the
shared-core boundary.

