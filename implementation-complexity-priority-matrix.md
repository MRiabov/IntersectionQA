# Implementation Complexity vs Priority Matrix

This document ranks the IntersectionQA implementation work by research/demo priority, implementation complexity, debugging difficulty, dependencies, likely modules, checks, risks, and build recommendation.

It is based on `paper-spec.md`, `epics-and-stories.md`, and the current repo state.

## Assumptions

* The repo is currently a skeleton: `main.py`, `pyproject.toml`, `README.md`, and specification docs.
* The module paths below are proposed implementation targets, not existing files yet.
* The first deliverable should be a credible paper/demo MVP, not the full benchmark vision.
* CadQuery/OpenCASCADE installation, boolean stability, and distance computation are the main technical risks.
* v1 should use generator, object-pair, assembly-group, and counterfactual-group holdouts. Chamfer or point-cloud leakage filtering is future work.
* CADEvolve is the primary source corpus for real dataset examples. Synthetic primitives and local mechanical motifs are fixture-only fallbacks for golden tests, smoke/debug cases, and local development when CADEvolve is unavailable.
* RL/GRPO is valuable, but the first implementation must produce reliable labels and prompts before training complexity is added.

## Legend

| Field | Values |
| --- | --- |
| Research/demo priority | `P0` required for credible demo/paper, `P1` strengthens paper, `P2` optional or future-facing |
| Implementation complexity | `Low`, `Medium`, `High` |
| Debugging/validation difficulty | `Low`, `Medium`, `High` |
| Recommendation | `Build now`, `Build later`, `Defer` |

## Proposed Module Map

The specs imply this package structure:

| Area | Proposed paths |
| --- | --- |
| CLI/scripts | `scripts/generate_dataset.py`, `scripts/validate_dataset.py`, `scripts/evaluate_baseline.py`, `scripts/inspect_example.py` |
| Package root | `intersectionqa/__init__.py`, `intersectionqa/config.py`, `intersectionqa/schema.py`, `intersectionqa/logging.py` |
| Sources | `intersectionqa/sources/base.py`, `intersectionqa/sources/synthetic.py`, `intersectionqa/sources/cadevolve.py` |
| Geometry | `intersectionqa/geometry/transforms.py`, `intersectionqa/geometry/cadquery_exec.py`, `intersectionqa/geometry/labels.py`, `intersectionqa/geometry/bbox.py`, `intersectionqa/geometry/repair.py` |
| Generation | `intersectionqa/generation/objects.py`, `intersectionqa/generation/assembly.py`, `intersectionqa/generation/boundary.py`, `intersectionqa/generation/counterfactual.py`, `intersectionqa/generation/multi_object.py` |
| Prompts | `intersectionqa/prompts/binary.py`, `intersectionqa/prompts/relation.py`, `intersectionqa/prompts/buckets.py`, `intersectionqa/prompts/counterfactual.py`, `intersectionqa/prompts/repair.py` |
| Splits/export | `intersectionqa/splits/grouped.py`, `intersectionqa/export/jsonl.py`, `intersectionqa/export/dataset_card.py` |
| Evaluation | `intersectionqa/evaluation/aabb.py`, `intersectionqa/evaluation/model_runner.py`, `intersectionqa/evaluation/metrics.py`, `intersectionqa/evaluation/rewards.py` |
| Tests | `tests/test_transforms.py`, `tests/test_labels.py`, `tests/test_generation.py`, `tests/test_prompts.py`, `tests/test_splits.py`, `tests/test_export.py` |

## Epic-Level Matrix

| Epic | Scope | P | Impl | Debug | Dependencies | Likely modules/files | Required checks | Key risks/unknowns | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Research scope and paper framing | P0 | Low | Low | Existing specs | `tasks.md`, `label_rules.md`, `paper_outline.md` | Reviewer-objection checklist, frozen task definitions | Scope creep | Build now |
| 2 | Dataset source ingestion | P0 | Medium | Medium | Repo structure, schema | `sources/`, `schema.py`, `scripts/validate_sources.py` | Loader smoke tests, provenance checks, licensing notes | CADEvolve access/licensing, malformed scripts | Build now |
| 3 | Assembly generation | P0 | High | High | Source objects, transforms, labels | `generation/`, `geometry/transforms.py` | Golden primitive assemblies, deterministic seeds | Transform convention bugs, trivial class distribution | Build now |
| 4 | Geometry labeling and verification | P0 | High | High | CadQuery/OpenCASCADE | `geometry/cadquery_exec.py`, `geometry/labels.py`, `geometry/bbox.py` | Box-box golden tests, boolean failure logging, epsilon policy checks | Kernel tolerance, invalid solids, distance API reliability | Build now |
| 5 | Prompt and task generation | P0 | Medium | Medium | Schema, labels | `prompts/`, `schema.py` | Snapshot tests, parser tests, exact answer formats | Ambiguous prompts, label/prompt drift | Build now |
| 6 | Dataset splits and balancing | P0 | Medium | Medium | Group metadata, export | `splits/grouped.py`, `export/jsonl.py` | No shared groups across splits, distribution reports | Leakage through counterfactual/object-pair groups | Build now |
| 7 | Baselines and model evaluation | P0 | Medium | Medium | Dataset export, prompts | `evaluation/`, `scripts/evaluate_baseline.py` | AABB sanity checks, model-output parsing, invalid-rate report | API cost, baseline bugs hiding dataset flaws | Build now |
| 8 | Rendering and multimodal extensions | P2 | High | Medium | Geometry pipeline, rendering deps | `rendering/`, `prompts/multimodal.py` | Manual render audit, file linkage checks | Rendering stack friction, low MVP value | Defer |
| 9 | Dataset packaging and release | P0 | Medium | Low | Schema, splits, generation | `export/`, `README.md`, dataset card | JSONL schema validation, version/config hash checks | Schema churn if done too late | Build now |
| 10 | Paper experiments | P0 | Medium | Medium | Dataset, baselines, model runs | `evaluation/metrics.py`, `paper/`, result tables | Reproducible runs, result table scripts | Results may reveal weak benchmark difficulty | Build now |
| 11 | Paper writing | P0 | Low | Low | Scope, dataset stats, results | `paper/`, `paper-spec.md` | Internal review, claims match evidence | Claims outrun implementation | Build now |
| 12 | Repository and engineering infrastructure | P0 | Medium | Medium | None | package tree, `pyproject.toml`, `scripts/`, `tests/` | CI/smoke tests, import checks, lint if added | Overengineering before geometry works | Build now |
| 13 | QA and review | P0 | Medium | Medium | Dataset export, diagnostics | `scripts/audit_dataset.py`, `tests/`, `docs/` | Manual audit sample, leakage audit, reproducibility audit | Hard examples mislabeled, silent leakage | Build now |

## Story and Feature Matrix

### Epic 1: Research Scope and Paper Framing

| Story/feature | P | Impl | Debug | Depends on | Likely modules/files | Required checks | Risks/unknowns | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 Benchmark thesis | P0 | Low | Low | Current specs | `paper_outline.md`, `paper-spec.md` | One-paragraph thesis matches MVP | Overbroad claims | Build now |
| 1.2 Task families | P0 | Low | Low | Label definitions | `tasks.md`, `label_rules.md` | Each task has answer format and metric | Too many P0 tasks | Build now |
| 1.3 Evaluation regimes | P0 | Low | Low | Task families | `paper_outline.md` | Closed-book vs tool-assisted explicitly separated | Unfair baseline comparisons | Build now |
| 1.4 Paper success metrics | P0 | Low | Low | Scope freeze | `paper_outline.md`, milestones | Paper-ready checklist exists | Moving target | Build now |

### Epic 2: Dataset Source Ingestion

| Story/feature | P | Impl | Debug | Depends on | Likely modules/files | Required checks | Risks/unknowns | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.1 Select initial sources | P0 | Low | Medium | Licensing/source access | `data_sources.md`, `sources/README.md` | Source license and provenance recorded | CADEvolve may be inconvenient to access | Build now |
| 2.2 Loader interface | P0 | Medium | Medium | Schema | `sources/base.py`, `sources/cadevolve.py`, `sources/synthetic.py` | Loader returns stable records, malformed records logged | CADEvolve archive paths and script conventions vary | Build now |
| 2.3 Normalize object functions | P0 | Medium | High | Loader, execution sandbox | `sources/normalize.py`, `geometry/cadquery_exec.py` | Normalized functions execute and return solids | Arbitrary CadQuery scripts may have side effects | Build now |
| 2.4 Validate source CAD objects | P0 | High | High | CadQuery install, normalized scripts | `geometry/cadquery_exec.py`, `sources/validation.py` | Positive volume, finite bbox, failure reasons | OpenCASCADE failures, invalid solids | Build now |
| CADEvolve source loader | P0 | Medium | Medium | Source access, licensing | `sources/cadevolve.py` | Sample load, provenance, generator IDs, isolated execution manifest | Dataset format, malformed scripts, or license friction | Build now |
| Synthetic primitive/motif fixtures | P0 | Low | Low | Repo structure | `sources/synthetic.py`, `generation/fixtures.py` | Deterministic box/cylinder/ring fixtures for golden labels | Fixture leakage into released benchmark counts | Build now |

### Epic 3: Assembly Generation

| Story/feature | P | Impl | Debug | Depends on | Likely modules/files | Required checks | Risks/unknowns | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.1 Rigid transform representation | P0 | Medium | High | None | `geometry/transforms.py`, `schema.py` | Known transform examples, rotation order tests | Rotation order and CadQuery convention mismatch | Build now |
| 3.2 Assembly script generator | P0 | Medium | Medium | Transforms, object schema | `generation/assembly.py`, `prompts/common.py` | Generated script executes, object names stable | Script prompt differs from executable metadata | Build now |
| 3.3 Random broad-placement examples | P0 | Medium | Medium | Valid objects, assembly generator, labels | `generation/random.py` | Class distribution report, deterministic seed | Overproduces obvious disjoint examples | Build now |
| 3.4 Boundary-targeted examples | P0 | High | High | Bbox diagnostics, exact labels | `generation/boundary.py` | Near-touch/near-overlap golden tests | Epsilon/tolerance flakiness | Build now |
| 3.5 Cavity-targeted examples | P1 | High | High | CADEvolve object diagnostics, exact labels | `generation/cavity.py`, `sources/metadata.py` | AABB-overlap but exact-disjoint cases | Hard to generate robustly from arbitrary source objects | Build later |
| 3.6 Counterfactual groups | P0 | Medium | Medium | Boundary generation, labels, split groups | `generation/counterfactual.py`, `schema.py` | Same group has one changed parameter and label diversity | Accidental multi-parameter changes | Build now |
| 3.7 Counterfactual prompt formats | P1 | Medium | Medium | Counterfactual groups, prompt generators | `prompts/counterfactual.py` | Individual, pairwise, and ranking prompt snapshots | Derived prompts may leak labels or split groups | Build later |
| 3.8 Multi-object assemblies | P2 | High | High | Pairwise labels, assembly generator | `generation/multi_object.py`, `prompts/pairwise.py` | Pairwise relation matrix tests | Combinatorial growth and prompt complexity | Defer |

### Epic 4: Geometry Labeling and Verification

| Story/feature | P | Impl | Debug | Depends on | Likely modules/files | Required checks | Risks/unknowns | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.1 Exact intersection volume | P0 | High | High | CadQuery/OpenCASCADE, valid solids | `geometry/labels.py`, `geometry/cadquery_exec.py` | Box overlap/touch/disjoint/containment golden tests | Boolean tolerance and sliver solids | Build now |
| 4.2 Minimum clearance distance | P0 | High | High | Valid solids, distance API | `geometry/labels.py`, `geometry/distance.py` | Known distance cases, touching threshold tests | CadQuery distance APIs may be brittle | Build now |
| 4.3 Relation classifier | P0 | Medium | Medium | Volumes, distance, thresholds | `geometry/relations.py`, `label_rules.md` | Epsilon policy tests, relation snapshot tests | Ambiguous touching vs tiny overlap | Build now |
| 4.4 Bounding-box diagnostics | P0 | Medium | Low | Valid solids, transforms | `geometry/bbox.py`, `evaluation/aabb.py` | AABB overlap golden tests | OBB can remain a separate P1 baseline | Build now |
| 4.5 Containment detection | P1 | High | High | Exact boolean labels | `geometry/labels.py` | Fully-contained primitive tests | Boolean classification edge cases | Build later |
| 4.6 Label consistency validation | P0 | Medium | Medium | All raw label fields | `scripts/validate_dataset.py`, `geometry/validation.py` | Contradiction checks, schema checks | Silent mislabels if checks are weak | Build now |

### Epic 5: Prompt and Task Generation

| Story/feature | P | Impl | Debug | Depends on | Likely modules/files | Required checks | Risks/unknowns | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.1 Binary prompt generator | P0 | Low | Low | Assembly script, relation label | `prompts/binary.py` | Snapshot tests, exact final answer parse | Ambiguous definition of interference | Build now |
| 5.2 Relation prompt generator | P0 | Medium | Medium | Relation classifier | `prompts/relation.py` | Class definitions in prompt match label rules | Touching/near-miss ambiguity | Build now |
| 5.3 Volume-bucket prompt generator | P0 | Medium | Medium | Intersection volume, bucket rules | `prompts/buckets.py` | Bucket boundary tests | Numeric instability at bucket edges | Build now |
| 5.4 Clearance-bucket prompt generator | P1 | Medium | Medium | Minimum distance | `prompts/buckets.py` | Bucket boundary and intersecting-case tests | Distance failures reduce coverage | Build later |
| 5.5 Pairwise multi-object prompt | P2 | Medium | Medium | Multi-object assemblies | `prompts/pairwise.py` | Pair-set parser tests | Prompt length and pair normalization | Defer |
| 5.6 Ranking prompt generator | P1 | Medium | Medium | Counterfactual groups or multiple assemblies | `prompts/ranking.py`, `prompts/counterfactual.py` | Tie-handling tests, ordering tests | Ties and near-equal volumes | Build later |
| 5.7 Repair-direction prompt | P2 | High | High | Geometry labels, repair heuristic | `geometry/repair.py`, `prompts/repair.py` | Primitive repair golden tests | Multiple valid repairs, ambiguous minima | Defer |
| 5.8 Tolerance-aware fit prompt | P1 | Low | Medium | Clearance distance, thresholds | `prompts/fit.py` | Threshold edge tests | Same distance fragility as clearance | Build later |

### Epic 6: Dataset Splits and Balancing

| Story/feature | P | Impl | Debug | Depends on | Likely modules/files | Required checks | Risks/unknowns | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.1 Random split | P0 | Low | Low | Dataset records | `splits/grouped.py` | Seed reproducibility, class counts | Not credible as only split | Build now |
| 6.2 Generator-held-out split | P0 | Medium | Medium | `generator_id` metadata | `splits/grouped.py` | No shared generator IDs | CADEvolve family IDs may need derivation from path or metadata | Build now |
| 6.3 Object-pair and counterfactual holdout | P0 | Medium | Medium | Group IDs | `splits/grouped.py`, `schema.py` | No shared object-pair, assembly, or counterfactual IDs | Derived prompts can accidentally cross splits | Build now |
| 6.4 Topology-held-out split | P2 | Medium | Medium | Topology labels | `splits/grouped.py`, `sources/metadata.py` | Held-out labels absent from train | Requires reliable topology taxonomy | Defer |
| 6.5 Operation-held-out split | P2 | Medium | Medium | Operation extraction | `sources/ops.py`, `splits/grouped.py` | Held-out ops absent from train | AST parsing complexity | Defer |
| 6.6 Near-boundary hard split | P0 | Medium | Medium | Boundary examples, diagnostics | `splits/grouped.py` | Hard split contains targeted cases | Too small or too label-sensitive | Build now |
| 6.7 Class balancing | P0 | Medium | Medium | Labels, split assignment | `splits/balance.py`, `scripts/dataset_stats.py` | Per-split label distribution reports | Balancing can bias generation | Build now |

### Epic 7: Baselines and Model Evaluation

| Story/feature | P | Impl | Debug | Depends on | Likely modules/files | Required checks | Risks/unknowns | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7.1 AABB baseline | P0 | Low | Low | Bbox diagnostics | `evaluation/aabb.py` | AABB primitive tests, per-subset accuracy | If AABB performs too well, dataset is too easy | Build now |
| 7.2 OBB baseline | P1 | Medium | Medium | Transform math or mesh approx | `evaluation/obb.py` | Rotated CADEvolve and fixture cases | OBB implementation can be subtle | Build later |
| 7.3 Convex-hull baseline | P2 | High | Medium | Mesh/point sampling deps | `evaluation/convex_hull.py` | Cavity failure examples | Adds geometry dependencies | Defer |
| 7.4 Zero-shot LLM evaluation | P0 | Medium | Medium | Prompt export, model access | `evaluation/model_runner.py`, `evaluation/parsing.py` | Output parser tests, invalid-rate report | API cost, model availability | Build now |
| 7.5 Few-shot LLM evaluation | P1 | Medium | Medium | Zero-shot runner, split-safe example selection | `evaluation/model_runner.py` | Few-shot examples do not leak groups | Prompt length and leakage | Build later |
| 7.6 Fine-tune small code model | P1 | High | Medium | Dataset export, training infra | `training/`, external trainer configs | Train/eval reproducibility | GPU budget, trainer choice | Build later |
| 7.7 Structured rationale fine-tune | P2 | High | Medium | Rationale generation, SFT infra | `training/`, `prompts/rationales.py` | Final-answer accuracy only | Templated rationales may teach shortcuts | Defer |
| 7.8 Counterfactual training | P1 | High | Medium | Counterfactual prompts, SFT infra | `training/`, `prompts/counterfactual.py` | Held-out group checks | Hard to attribute gains | Build later |
| 7.9 Verifier-guided RL/GRPO | P1 | High | High | Reliable verifier, reward parser, training infra | `evaluation/rewards.py`, `training/rl/` | Reward unit tests, invalid format penalties | Significant training complexity | Build later |
| 7.10 Tool-assisted upper bound | P1 | Medium | Medium | Geometry verifier, script executor | `evaluation/tool_assisted.py` | Tool failure-rate report | Not fair as reasoning baseline | Build later |

### Epic 8: Rendering and Multimodal Extensions

| Story/feature | P | Impl | Debug | Depends on | Likely modules/files | Required checks | Risks/unknowns | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8.1 Object and assembly renders | P2 | High | Medium | Valid geometry, renderer | `rendering/render.py` | Manual render audit, missing-file checks | Rendering dependency friction | Defer |
| 8.2 Orthographic views | P2 | Medium | Medium | Renderer | `rendering/views.py` | View naming and camera tests | Inconsistent camera conventions | Defer |
| 8.3 Multimodal prompt variant | P2 | Medium | Medium | Renders, model eval runner | `prompts/multimodal.py` | File linkage and prompt snapshots | Extra evaluation cost | Defer |

### Epic 9: Dataset Packaging and Release

| Story/feature | P | Impl | Debug | Depends on | Likely modules/files | Required checks | Risks/unknowns | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9.1 Final dataset schema | P0 | Medium | Medium | Task, label, split definitions | `schema.py`, `docs/schema.md` | JSON schema/Pydantic validation | Schema churn if delayed | Build now |
| 9.2 Export JSONL files | P0 | Medium | Low | Dataset records, splits | `export/jsonl.py`, `scripts/generate_dataset.py` | File counts, schema validation, stable IDs | Partial exports after failures | Build now |
| 9.3 Dataset card | P1 | Low | Low | Data stats, sources, limitations | `dataset-card.md` | Matches released data and license | Missing limitation disclosure | Build later |
| 9.4 Reproducibility scripts | P0 | Medium | Medium | Generation, validation, evaluation | `scripts/` | Commands work from clean checkout | Script/config drift | Build now |
| 9.5 Versioning | P0 | Low | Low | Config system, git hash | `config.py`, `export/metadata.py` | Version/config hash in records | Hard to reproduce without this | Build now |

### Epic 10: Paper Experiments

| Story/feature | P | Impl | Debug | Depends on | Likely modules/files | Required checks | Risks/unknowns | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10.1 Dataset statistics | P0 | Medium | Low | Exported dataset | `scripts/dataset_stats.py`, `evaluation/metrics.py` | Counts by task/split/label/source/difficulty | Missing metadata limits analysis | Build now |
| 10.2 Baseline comparison table | P0 | Medium | Medium | AABB, zero-shot eval | `evaluation/reports.py` | Table regenerates from result files | Inconsistent result formats | Build now |
| 10.3 Task-family comparison | P0 | Medium | Medium | Multiple task prompts and eval | `evaluation/reports.py` | Per-task metrics | Too few tasks in MVP | Build now |
| 10.4 Diagnostic subset analysis | P1 | Medium | Medium | Diagnostic tags, baselines | `evaluation/reports.py` | AABB-failing and near-boundary slices | Subsets may be too small | Build later |
| 10.5 Training-data ablation | P1 | High | Medium | SFT/RL runs | `training/`, `evaluation/reports.py` | Same eval splits across runs | GPU budget and run variance | Build later |
| 10.6 Failure-case analysis | P0 | Low | Medium | Model results, inspection tool | `scripts/inspect_example.py`, `paper/failures.md` | At least 10 audited examples | Cherry-picking accusations | Build now |

### Epic 11: Paper Writing

| Story/feature | P | Impl | Debug | Depends on | Likely modules/files | Required checks | Risks/unknowns | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11.1-11.2 Abstract and introduction | P0 | Low | Low | Thesis and MVP scope | `paper/`, `paper_outline.md` | Claims match implemented tasks | Overclaiming | Build now |
| 11.3 Related work | P0 | Medium | Low | Citation gathering | `paper/related_work.md` | Citations verified | Missing relevant CAD benchmark | Build now |
| 11.4 Dataset construction section | P0 | Medium | Low | Generation pipeline and stats | `paper/dataset.md` | Matches code and schema | Implementation/spec mismatch | Build now |
| 11.5 Task section | P0 | Low | Low | Prompt specs | `paper/tasks.md`, `tasks.md` | Answer formats exact | Ambiguous labels | Build now |
| 11.6 Experiments section | P0 | Medium | Medium | Results tables | `paper/experiments.md` | Tables regenerate | Weak or incomplete results | Build now |
| 11.7-11.9 Discussion, limitations, conclusion | P0 | Low | Low | Failure analysis, QA checklist | `paper/discussion.md` | Limitations include kernel issues, CADEvolve source bias, and fixture limits | Hiding weaknesses | Build now |

### Epic 12: Repository and Engineering Infrastructure

| Story/feature | P | Impl | Debug | Depends on | Likely modules/files | Required checks | Risks/unknowns | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12.1 Repository structure | P0 | Low | Low | None | package tree, `scripts/`, `tests/` | Imports work, CLI entrypoints run | Premature folder sprawl | Build now |
| 12.2 Configuration system | P0 | Medium | Medium | Schema | `config.py`, `configs/*.yaml` | Config load tests, seed reproducibility | Config too flexible too early | Build now |
| 12.3 Logging/failure tracking | P0 | Medium | Medium | Generation and labeling | `logging.py`, `scripts/` | Failure reasons counted and exported | Silent skips bias dataset | Build now |
| 12.4 Unit tests | P0 | Medium | Medium | Core modules | `tests/` | `pytest`, golden geometry cases | CadQuery tests may be slow/flaky | Build now |
| 12.5 Smoke-test dataset generation | P0 | Medium | Medium | Minimal generator, labels, prompts, export | `configs/smoke.yaml`, `scripts/generate_dataset.py` | Generate 100 examples and validate | Smoke set may miss hard failures | Build now |

### Epic 13: Quality Assurance and Review

| Story/feature | P | Impl | Debug | Depends on | Likely modules/files | Required checks | Risks/unknowns | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13.1 Manual audit sample | P0 | Low | Medium | Exported examples, inspection tool | `scripts/inspect_example.py`, `docs/audit.md` | 100 random and 100 hard examples reviewed | Human audit is slow but necessary | Build now |
| 13.2 Reproducibility audit | P0 | Medium | Medium | Config, versioning, deterministic IDs | `scripts/reproduce_dataset.py` | Same seed produces same IDs and labels | Nondeterministic geometry operations | Build now |
| 13.3 Leakage audit | P0 | Medium | Medium | Split metadata | `scripts/audit_leakage.py`, `splits/grouped.py` | No shared generator/object-pair/assembly/counterfactual groups | Missing group IDs reduce credibility | Build now |
| 13.4 Reviewer-readiness checklist | P0 | Low | Low | Specs, results, audits | `docs/reviewer_checklist.md` | Answers common objections | Checklist not backed by data | Build now |

## Recommended MVP Implementation Path

Build the MVP in a narrow vertical slice first:

1. Create package structure, schema, config, and scripts.
2. Implement the CADEvolve tar loader, source manifest, provenance capture, and isolated object-validation worker.
3. Add minimal primitive fixtures for golden label and transform tests only.
4. Implement transforms, assembly script generation, and exact labels on fixtures plus a deterministic CADEvolve smoke subset.
5. Add binary, relation, and volume-bucket prompts.
6. Add group-safe random, generator-held-out, object-pair-held-out, and near-boundary hard splits.
7. Export JSONL and run schema/label/leakage validation.
8. Add AABB baseline and one zero-shot LLM evaluation path.
9. Generate dataset statistics and a first paper/demo result table.
10. Add counterfactual groups after the basic CADEvolve labeling path is stable.
11. Add SFT/RL experiments only after validation catches label, split, and prompt errors.

The first implementation target should be a `100` accepted-geometry smoke dataset that uses CADEvolve where possible and exercises:

* intersecting
* disjoint
* touching
* near-miss
* small positive overlap
* rotated examples
* counterfactual groups if implemented
* AABB-overlap but exact-disjoint if available from CADEvolve or a small ring/plate fixture

## P0 Features Required for Paper/Demo

These must exist before the project is credible:

| Feature | Why it is necessary |
| --- | --- |
| Stable dataset schema | All generation, prompts, splits, and evaluation depend on it |
| CADEvolve source loading and object validation | Establishes the real dataset source, provenance, and failure behavior |
| Minimal synthetic fixtures | Gives controllable golden cases without becoming the benchmark source |
| Rigid transforms and assembly script generation | Core task input |
| Exact intersection volume | Core ground truth |
| Minimum distance and relation classifier | Needed to distinguish disjoint, touching, near-miss, and intersecting |
| Binary, relation, and volume-bucket prompts | Enough task diversity for the initial paper |
| Boundary-targeted generation | Prevents the dataset from being mostly trivial |
| Counterfactual groups | Main differentiator and training value |
| Group-safe splits | Required to avoid leakage criticism |
| AABB baseline | Required to show shortcut solvability or difficulty |
| Zero-shot model evaluation | Minimum model benchmark |
| Dataset stats and validation scripts | Required for paper tables and debugging |
| Manual and leakage audits | Required to trust labels and splits |

## High-Priority High-Risk Features Needing Early Spikes

These should be spiked early because they can invalidate schedule assumptions:

| Feature | Spike goal | Success criterion |
| --- | --- | --- |
| CadQuery/OpenCASCADE execution | Prove install and script execution work locally | `pytest` can create valid solids and compute volume |
| Boolean intersection volume | Prove exact overlap labels are reliable enough | Golden box/cylinder cases pass |
| Minimum distance | Prove clearance labels can be computed | Disjoint and touching primitive cases pass |
| Transform convention | Prove stored transform equals executable script transform | Rotated/translated primitive bbox matches expectation |
| Boundary-targeted generation | Prove near-boundary examples can be generated deterministically | Smoke set includes touching, near-miss, and tiny-overlap cases |
| Group-safe split logic | Prove counterfactual and object-pair groups cannot leak | Audit script fails on intentionally leaked fixture |
| Model-output parser | Prove evaluations are robust to verbose answers | Parser accepts only documented final-answer formats |

## Low-Priority or High-Complexity Features to Defer

| Feature | Reason to defer |
| --- | --- |
| Chamfer or point-cloud leakage filtering | Adds meshing, sampling, nearest-neighbor search, and threshold calibration |
| Convex-hull baseline | Useful but adds geometry dependencies beyond MVP |
| Multi-object assemblies | Combinatorial complexity; two-object tasks are enough for first demo |
| Repair-direction task | Ambiguous labels for complex geometry; high verifier complexity |
| Rendered images and multimodal evaluation | Valuable extension but not necessary to prove code-spatial reasoning |
| Topology-held-out split | Needs reliable taxonomy and enough examples per class |
| Operation-held-out split | Requires robust CadQuery operation extraction |
| Structured-rationale fine-tuning | Can teach templates and consumes training time |
| Full RL/GRPO training | Worth doing after labels and prompts are stable; not before |

## Suggested Implementation Order With Rationale

| Order | Build | Rationale |
| --- | --- | --- |
| 1 | Repo structure, schema, config, tests | Prevents every later component from inventing its own record shape |
| 2 | CADEvolve source manifest and loader | Makes the real source corpus the first-class path |
| 3 | Isolated CadQuery execution and object validation | Exposes CADEvolve script, environment, and kernel problems early |
| 4 | Minimal synthetic fixtures | Supplies deterministic golden cases when CADEvolve does not hit exact edge cases |
| 5 | Transform and assembly generation | Core source of subtle bugs; must be tested before labels |
| 6 | Exact labels and relation rules | Ground truth must be trustworthy before generating prompts |
| 7 | Binary prompt and JSONL export | First end-to-end artifact |
| 8 | CADEvolve smoke generation and validation script | Makes debugging repeatable on the target corpus |
| 9 | Boundary-targeted generation | Adds benchmark difficulty before scaling volume |
| 10 | Counterfactual groups | Adds main research differentiator |
| 11 | Relation and volume-bucket prompts | Adds task diversity while reusing existing labels |
| 12 | Group-safe splits and leakage audit | Protects paper credibility |
| 13 | AABB baseline and dataset stats | Reveals whether the dataset is too easy |
| 14 | Zero-shot model evaluation | First model result for paper/demo |
| 15 | Cavity examples and OBB baseline | Strengthens diagnostic story |
| 16 | Fine-tuning and counterfactual training | Shows dataset usefulness after benchmark stability |
| 17 | RL/GRPO reward experiment | High-value but should use already-debugged prompts and verifier |
| 18 | Rendering, multimodal, convex hull, topology/operation holdouts | Useful extensions after the paper MVP is stable |

## Debugging Strategy While Building

Do not debug this only at full dataset scale. Add these checkpoints as implementation gates:

| Gate | Required artifact | Pass condition |
| --- | --- | --- |
| Geometry unit tests | `tests/test_labels.py` | Primitive golden cases pass |
| Transform unit tests | `tests/test_transforms.py` | Stored metadata and executable script produce same placement |
| Prompt snapshots | `tests/test_prompts.py` | Prompt text and final answer format are stable |
| Smoke dataset | `data/smoke/*.jsonl` or temp output | 100 examples generate, validate, and export |
| Label validation | `scripts/validate_dataset.py` | No contradictory raw labels |
| Split audit | `scripts/audit_leakage.py` | No shared generator, object-pair, assembly, or counterfactual groups |
| Baseline sanity | `scripts/evaluate_baseline.py --baseline aabb` | Reports overall and per-subset accuracy |
| Manual inspection | `scripts/inspect_example.py --id ...` | Can inspect code, transforms, labels, and prompt together |

## Practical Build/Defer Decision Rules

Use these rules when scope pressure appears:

| If this happens | Decision |
| --- | --- |
| CadQuery boolean labels are flaky | Reduce to simpler validated CADEvolve subsets and fixture-only golden cases until labels are stable |
| Minimum distance is unreliable | Keep binary and volume labels, delay clearance/tolerance tasks |
| Random examples are too easy | Prioritize boundary-targeted and counterfactual generation over adding more sources |
| AABB baseline performs too well | Prioritize CADEvolve objects with cuts, shells, holes, and cavity/concavity diagnostics before adding model experiments |
| Split audit catches leakage | Stop scaling data until group IDs and derived prompt splits are fixed |
| CADEvolve ingestion is slow | Shrink the CADEvolve smoke subset, improve worker/caching throughput, and use fixtures only for CI/golden coverage |
| Training setup becomes expensive | Ship zero-shot, AABB, and small SFT first; keep RL/GRPO as an optional experiment |
| Prompt parsing is messy | Tighten final answer tags before running more model evaluations |
