# Benchmark Family Direction

This repository is currently IntersectionQA-first. IntersectionQA v0.1 exists,
is frozen, and remains the canonical implemented dataset/task family.

The intended future shape is a shared CAD intersection benchmark codebase with
two sibling task families:

- **IntersectionQA**: diagnostic and question-answering tasks over CAD
  intersections, such as interference, relation, and volume-bucket prediction.
- **IntersectionEdit**: edit, repair, and action tasks over the same geometry,
  provenance, and verifier infrastructure.
- **Shared core**: geometry measurement, CadQuery execution, transforms,
  grouping, split assignment, leakage audits, export, evaluation, and
  verifier/reward logic.

QA and Edit tasks should share group IDs, counterfactual groups, leakage rules,
and geometry labels wherever the underlying geometry record is the same. Task
rows may differ, but they should not fork source provenance or split semantics.

## Migration Sequence

1. Keep current IntersectionQA docs and v0.1 specs stable.
2. Add an IntersectionEdit task specification without changing package or repo
   layout.
3. Implement one IntersectionEdit vertical slice against the existing shared
   geometry/verifier pipeline.
4. Only after that slice is working, consider package, repository, or directory
   renaming based on demonstrated shared-core boundaries.

