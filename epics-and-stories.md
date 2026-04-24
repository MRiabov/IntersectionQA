# Scrum Epics and Stories: IntersectionQA Paper + Dataset Delivery

## Product goal

Deliver **IntersectionQA**, a research paper and accompanying dataset for evaluating and training models on CAD-code spatial reasoning, especially interference, contact, clearance, and intersection-volume estimation from CadQuery assemblies. The v0.1 MVP focuses on two-object closed-book tasks; ranking, repair-direction, tolerance-fit, multimodal, and multi-object tasks remain reserved extensions unless a later release promotes them.

---

# Epic 1: Research Scope and Paper Framing

## Goal

Define the exact research contribution, task scope, benchmark claims, and paper narrative before implementation expands.

## Story 1.1 — Define benchmark thesis

**As a researcher**, I want a clear central thesis for IntersectionQA so that the paper has a focused contribution.

**Acceptance criteria**

* A one-paragraph thesis statement exists.
* The thesis distinguishes IntersectionQA from generic text-to-CAD datasets.
* The thesis explains why closed-book CAD spatial reasoning is useful.
* The thesis explains why tool-assisted evaluation is still relevant.

**Output**

```markdown
IntersectionQA tests whether language and code models understand the spatial consequences of CAD programs well enough to reason about assembly interference, contact, clearance, overlap magnitude, and repair direction.
```

---

## Story 1.2 — Define task families

**As a dataset designer**, I want the final task list defined so that implementation can target stable labels and prompts.

**Acceptance criteria**

The v0.1 MVP includes:

* Binary interference classification
* Relation classification
* Intersection-volume bucket prediction

The roadmap reserves compatible later task types for:

* Clearance bucket prediction
* Multi-object pairwise interference
* Ranking by normalized intersection volume
* Minimal repair-direction prediction
* Tolerance-aware fit checking

**Output**

`benchmark-task-spec.md` describing task types, answer formats, parser behavior, and label derivation, with thresholds and bucket boundaries delegated to `label_rules.md`.

---

## Story 1.3 — Define evaluation regimes

**As a reviewer-facing researcher**, I want to separate no-tool reasoning from tool-assisted reasoning so that claims are precise.

**Acceptance criteria**

The paper defines:

* Closed-book setting: no execution, model answers from code only.
* Tool-assisted setting: model may execute or call verifier tools.
* Exact CAD-kernel verifier as upper bound, not as a fair reasoning baseline.
* Optional multimodal setting with renders.

**Output**

Section draft: `Evaluation Settings`.

---

## Story 1.4 — Define success metrics for the paper

**As the project owner**, I want clear criteria for when the project is paper-ready.

**Acceptance criteria**

Paper-ready means:

* Dataset pipeline can generate valid examples reproducibly.
* At least 3 task types are implemented.
* At least 4 splits are available.
* At least 5 model/baseline evaluations are complete.
* Paper draft includes method, dataset, experiments, limitations, and release notes.

---

# Epic 2: Dataset Source Ingestion

## Goal

Ingest CADEvolve CadQuery programs as the primary dataset source and normalize them into a common representation. Synthetic fixtures are used only for golden tests, smoke/debug cases, and label edge cases.

---

## Story 2.1 — Select initial CAD program sources

**As a dataset builder**, I want to choose source datasets so that the examples are diverse and defensible.

**Acceptance criteria**

* At least one large CadQuery source is selected.
* Source licensing is checked.
* Dataset provenance is recorded.
* Each source has a short justification.

**Source policy**

* CADEvolve is the primary source corpus for real benchmark examples.
* Use executable CadQuery programs from `CADEvolve-P/` and `CADEvolve-C/` first.
* Human-written CadQuery examples may be used later for audit or supplemental analysis, not as the MVP source.
* Synthetic procedural primitives and custom motifs are limited to golden tests, transform convention tests, relation-label edge cases, and smoke/debug fixtures.
* Do not build a separate full synthetic CAD corpus before CADEvolve ingestion.

**Output**

`using-cadevolve-dataset-export.md` and the CADEvolve source-policy sections in `generation_policy.md`.

---

## Story 2.2 — Build source loader interface

**As an engineer**, I want a common loader API so that different CAD sources can be ingested uniformly.

**Acceptance criteria**

Loader returns records with:

```json
{
  "source": "cadevolve",
  "source_id": "...",
  "script": "...",
  "object_function_names": ["..."],
  "metadata": {}
}
```

* Loader handles malformed examples.
* Loader logs skipped records.
* Loader supports deterministic sampling by seed.

---

## Story 2.3 — Normalize CadQuery object functions

**As an engineer**, I want object scripts normalized so that downstream assembly generation is reliable.

**Acceptance criteria**

* Each object is wrapped as a callable function.
* Required imports are standardized.
* Object-return convention is standardized.
* Scripts producing non-solid outputs are filtered or tagged.
* Object names are canonicalized.

**Example**

```python
def object_a():
    return (
        cq.Workplane("XY")
        .box(10, 10, 10)
    )
```

---

## Story 2.4 — Validate source CAD objects

**As a dataset builder**, I want to verify that objects execute and produce valid solids.

**Acceptance criteria**

For each object:

* Script executes.
* Object is a valid CadQuery/CQ solid.
* Volume is positive.
* Bounding box is finite.
* Degenerate objects are rejected.
* Failure reason is stored.

**Output fields**

```json
{
  "valid": true,
  "volume": 1000.0,
  "bbox": [[-5, -5, -5], [5, 5, 5]],
  "failure_reason": null
}
```

---

# Epic 3: Assembly Generation

## Goal

Generate two-object and multi-object assemblies with controlled transforms, difficulty, and geometric relations.

---

## Story 3.1 — Implement rigid transform representation

**As an engineer**, I want a precise transform schema so that all examples are reproducible.

**Acceptance criteria**

Transform contains:

```json
{
  "translation": [x, y, z],
  "rotation_xyz_deg": [rx, ry, rz],
  "rotation_order": "XYZ"
}
```

* Transform order is documented.
* Generated script matches stored metadata.
* Transform application is deterministic.

---

## Story 3.2 — Implement assembly script generator

**As an engineer**, I want to generate complete CadQuery assembly scripts for prompts.

**Acceptance criteria**

Generated scripts include:

* Imports
* Object functions
* `place(...)` helper
* `assembly()` function
* Explicit object names
* Deterministic transform values

**Example output**

```python
def assembly():
    a = place(object_a(), (0, 0, 0), (0, 0, 0))
    b = place(object_b(), (8, 0, 0), (0, 0, 30))
    return a, b
```

---

## Story 3.3 — Generate random broad-placement examples

**As a dataset builder**, I want broad random examples so that the dataset has diversity.

**Acceptance criteria**

* Random object pairs are sampled.
* Random translations and rotations are applied.
* Relation labels are computed.
* Obvious invalid cases are filtered.
* Class distribution is measured.

---

## Story 3.4 — Generate boundary-targeted examples

**As a dataset builder**, I want examples near contact/interference boundaries so that the benchmark is not trivial.

**Acceptance criteria**

Generated examples include:

* Small positive clearance
* Exact or near-exact touching
* Small positive overlap
* Perturbations around boundary conditions

Metadata includes:

```json
{
  "boundary_type": "near_touching",
  "perturbation_mm": 0.05
}
```

---

## Story 3.5 — Generate cavity-targeted examples

**As a dataset builder**, I want examples where bounding boxes overlap but exact solids do not.

**Acceptance criteria**

* Examples include rings, holes, slots, brackets, shells, U-shapes, or cutouts.
* AABB overlap is often true.
* Exact overlap may be false.
* These examples are tagged as `cavity_targeted`.

---

## Story 3.6 — Generate counterfactual groups

**As a researcher**, I want groups of examples differing by one parameter so that models must reason precisely.

**Acceptance criteria**

Each counterfactual group has:

* Shared base object pair.
* Shared script structure.
* One varied parameter.
* At least two different labels across the group.
* Group ID stored.
* Variant ID stored for every member.
* Base object-pair ID stored for every member.
* Changed parameter and changed value stored for every member.
* Group members are marked as split-inseparable.

**Counterfactual dimensions**

* Translation
* Rotation
* Radius
* Width/height/depth
* Hole radius
* Cut depth
* Wall thickness
* Fillet radius

**Output**

```json
{
  "counterfactual_group_id": "cfg_000042",
  "variant_id": "cfg_000042_v1",
  "base_object_pair_id": "pair_000091",
  "changed_parameter": "translation_x",
  "changed_value": 9.9,
  "answer": "yes",
  "labels": {
    "relation": "intersecting",
    "intersection_volume": 1.0,
    "minimum_distance": 0.0
  }
}
```

---

## Story 3.7 — Materialize counterfactual prompt formats

**As a training-data designer**, I want counterfactual groups exportable as individual rows, pairwise prompts, and ranking prompts so that the same source examples can support SFT, evaluation, and RL.

**Acceptance criteria**

* Individual-row format exists for SFT and standard evaluation.
* Pairwise comparison format exists with answer choices `A`, `B`, `both`, and `neither`.
* Ranking format exists for 3-5 variants ordered by normalized intersection volume.
* Derived prompts reference source `counterfactual_group_id` and variant IDs.
* Pairwise and ranking labels are generated from exact geometry metadata.
* Group members are not split across train/validation/test unless an interpolation experiment explicitly opts in.

---

## Story 3.8 — Generate multi-object assemblies

**As a dataset builder**, I want assemblies with 3–10 parts so that pairwise reasoning can be evaluated.

**Acceptance criteria**

* Multi-object scripts are generated.
* All pairwise relations are computed.
* Prompt can ask for all interfering pairs.
* At least one subset contains no collisions.
* At least one subset contains multiple collisions.

---

# Epic 4: Geometry Labeling and Verification

## Goal

Compute reliable labels using exact CAD-kernel operations and store diagnostic metadata.

---

## Story 4.1 — Compute exact intersection volume

**As an engineer**, I want exact or kernel-derived intersection volume so that labels are grounded.

**Acceptance criteria**

For each object pair:

* Compute object volumes.
* Compute intersection solid.
* Compute intersection volume.
* Handle Boolean failures.
* Store raw and normalized intersection.

**Output**

```json
{
  "volume_a": 1000.0,
  "volume_b": 800.0,
  "intersection_volume": 47.2,
  "normalized_intersection": 0.059
}
```

---

## Story 4.2 — Compute minimum clearance distance

**As an engineer**, I want minimum distance for non-intersecting examples so that clearance tasks can be generated.

**Acceptance criteria**

* Minimum distance is computed where possible.
* Touching cases are identified.
* Distance failures are logged.
* Distance threshold is configurable.

---

## Story 4.3 — Implement relation classifier

**As a dataset builder**, I want raw geometry values mapped to stable relation labels.

**Acceptance criteria**

Relations include:

```text
disjoint
touching
near_miss
intersecting
contained
invalid
```

Rules are configurable:

```python
epsilon_volume = 1e-6 * min(volume_a, volume_b)
epsilon_distance = 1e-4
near_miss_threshold = 1.0
```

---

## Story 4.4 — Compute bounding-box diagnostics

**As a researcher**, I want AABB/OBB/convex-hull diagnostics so that shortcut baselines can be analyzed.

**Acceptance criteria**

For each v0.1 pair, store at least:

```json
{
  "aabb_overlap": true,
  "exact_overlap": true
}
```

* AABB baseline can be evaluated.
* OBB and convex-hull diagnostics can be added by later baseline jobs when implemented.
* Cases where heuristic and exact labels disagree are tagged.

---

## Story 4.5 — Detect containment

**As a dataset builder**, I want contained cases separated from ordinary intersection.

**Acceptance criteria**

* One object fully inside another can be detected.
* Containment is stored separately.
* Binary interference treats containment as positive overlap unless otherwise configured.

---

## Story 4.6 — Validate label consistency

**As a researcher**, I want automatic checks to catch inconsistent geometry labels.

**Acceptance criteria**

Validation catches:

* Positive intersection volume with positive minimum distance.
* Zero volume but relation marked intersecting.
* Invalid solids marked as valid.
* Normalized intersection outside expected range.
* Missing task labels.

---

# Epic 5: Prompt and Task Generation

## Goal

Generate final prompts and labels for each task type in a stable, reproducible format.

---

## Story 5.1 — Implement binary interference prompt generator

**As a dataset user**, I want simple yes/no examples.

**Acceptance criteria**

* Prompt includes definitions of interference.
* Prompt specifies no code execution.
* Final answer format is exactly `yes` or `no`.
* Label matches relation rules.

---

## Story 5.2 — Implement relation-classification prompt generator

**As a dataset user**, I want prompts that distinguish touching, near-miss, and interference.

**Acceptance criteria**

* Prompt includes class definitions.
* Answer labels are restricted.
* Labels are generated from geometry metadata.
* Invalid cases can be included or excluded.

---

## Story 5.3 — Implement volume-bucket prompt generator

**As a dataset user**, I want examples that test magnitude reasoning.

**Acceptance criteria**

* Prompt defines normalized intersection volume.
* Buckets are documented.
* Label is computed from normalized overlap.
* Zero-overlap examples are included.

---

## Story 5.4 — Implement clearance-bucket prompt generator

**As a dataset user**, I want examples that test fit/clearance reasoning.

**Acceptance criteria**

* Prompt applies only to non-intersecting examples or explicitly handles intersections.
* Clearance buckets are stable.
* Touching is distinct from small positive clearance.

---

## Story 5.5 — Implement pairwise multi-object prompt generator

**As a dataset user**, I want prompts for all interfering pairs in an assembly.

**Acceptance criteria**

* Object names are clear.
* Pair labels are deterministic.
* Answer format is comma-separated pairs or `none`.
* Pair order is normalized.

---

## Story 5.6 — Implement ranking prompt generator

**As a dataset user**, I want dense ranking examples.

**Acceptance criteria**

* Prompt contains 3–5 independent assemblies.
* Ranking target is normalized intersection volume.
* Output is a compact letter string, such as `DBCA`.
* Ties are handled by documented rule.

---

## Story 5.7 — Implement repair-direction prompt generator

**As a dataset user**, I want examples where the model predicts how to reduce interference.

**Acceptance criteria**

* Movable object is specified.
* Allowed directions are fixed.
* Label is computed from separating direction or minimum translation heuristic.
* Ambiguous cases can be tagged or excluded.

---

## Story 5.8 — Implement tolerance-aware fit prompt generator

**As a dataset user**, I want examples based on required clearance.

**Acceptance criteria**

* Required clearance is stated in mm.
* Label is `yes` only if minimum clearance meets threshold.
* Intersecting and touching cases are handled as `no`.

---

# Epic 6: Dataset Splits and Balancing

## Goal

Create robust train/test splits that prevent leakage and support meaningful evaluation.

---

## Story 6.1 — Implement random split

**As a researcher**, I want a basic random split for sanity-check evaluation.

**Acceptance criteria**

* Train/validation/test split exists.
* Class balance is reported.
* Random seed is recorded.

---

## Story 6.2 — Implement generator-held-out split

**As a researcher**, I want to test generalization to unseen CAD generator families.

**Acceptance criteria**

* Examples sharing a generator/source family are not split across train and test.
* Split group is stored.
* Leakage checks are run.

---

## Story 6.3 — Implement object-pair and counterfactual group holdout

**As a researcher**, I want near-identical variants kept in the same split so that evaluation does not leak by construction.

**Acceptance criteria**

* Examples sharing `base_object_pair_id` are not split across train and test.
* Examples sharing `assembly_group_id` are not split across train and test.
* Examples sharing `counterfactual_group_id` are not split across train and test.
* Pairwise and ranking prompts inherit the strictest split group from their source variants.
* Split reports include counts by generator group, object-pair group, assembly group, and counterfactual group.
* Any intentional interpolation split is named separately and cannot be confused with the main held-out tests.

---

## Story 6.4 — Implement topology-held-out split

**As a researcher**, I want to test generalization to unseen shape motifs.

**Acceptance criteria**

Held-out topology categories may include:

* ring
* bracket
* clamp
* hollow box
* shaft
* housing
* flange
* plate-with-holes

Train/test topology assignments are stored.

---

## Story 6.5 — Implement operation-held-out split

**As a researcher**, I want to test generalization to unseen CadQuery operations.

**Acceptance criteria**

Held-out operations may include:

* `fillet`
* `chamfer`
* `shell`
* `loft`
* `revolve`
* `sweep`
* `cutThruAll`

Examples are tagged by operation use.

---

## Story 6.6 — Implement near-boundary hard split

**As a researcher**, I want a hard split focused on tiny clearances and overlaps.

**Acceptance criteria**

Hard split includes:

* Small positive overlap
* Small positive clearance
* Touching
* AABB-overlap-but-no-intersection
* Convex-hull-overlap-but-no-intersection

---

## Story 6.7 — Balance dataset classes

**As a dataset builder**, I want label distributions controlled so that accuracy is meaningful.

**Acceptance criteria**

Binary target distribution approximates:

```text
40% intersecting
30% disjoint
15% touching
15% near_miss
```

Relation target distribution is reported.

Oversampling/undersampling logic is documented.

---

# Epic 7: Baselines and Model Evaluation

## Goal

Evaluate models and heuristics to demonstrate benchmark difficulty and usefulness.

---

## Story 7.1 — Implement AABB baseline

**As a researcher**, I want a simple bounding-box baseline to test shortcut solvability.

**Acceptance criteria**

* AABB overlap predicts binary interference.
* Accuracy is reported overall and by subset.
* Failure cases are logged.

---

## Story 7.2 — Implement OBB baseline

**As a researcher**, I want an oriented bounding-box baseline for rotated objects.

**Acceptance criteria**

* OBB overlap is computed or approximated.
* Results are compared to AABB and exact labels.
* Subsets where OBB fails are identified.

---

## Story 7.3 — Implement convex-hull baseline

**As a researcher**, I want a stronger geometric approximation baseline.

**Acceptance criteria**

* Convex-hull overlap prediction is available.
* Cavity/concavity failure cases are analyzed.
* Results are reported separately.

---

## Story 7.4 — Evaluate zero-shot LLMs

**As a researcher**, I want to evaluate closed-book model performance.

**Acceptance criteria**

* At least one frontier model is evaluated if available.
* At least one open code model is evaluated.
* Prompts are fixed and versioned.
* Temperature and decoding settings are recorded.
* Invalid output rate is reported.

---

## Story 7.5 — Evaluate few-shot LLMs

**As a researcher**, I want to test whether examples improve performance.

**Acceptance criteria**

* Few-shot prompt templates are created.
* Examples do not leak from test groups.
* Results are compared against zero-shot.

---

## Story 7.6 — Fine-tune small code model

**As a researcher**, I want to test whether supervised training improves CAD spatial reasoning.

**Acceptance criteria**

* A small open model is fine-tuned on answer-only examples.
* Evaluation is run on all splits.
* Results are compared to base model.

---

## Story 7.7 — Fine-tune with structured rationales

**As a researcher**, I want to test whether geometry traces improve performance.

**Acceptance criteria**

* Rationale training data exists for a subset.
* Model is fine-tuned with rationale + answer format.
* Evaluation measures final-answer accuracy.
* Results are compared to answer-only SFT.

---

## Story 7.8 — Evaluate counterfactual training

**As a researcher**, I want to test whether counterfactual examples improve near-boundary reasoning.

**Acceptance criteria**

* Fine-tuning run includes counterfactual groups.
* Individual-row counterfactual SFT is compared with pairwise or ranking counterfactual training.
* Held-out counterfactual groups are not leaked.
* Improvement is reported on near-boundary hard split.

---

## Story 7.9 — Run verifier-guided RL/GRPO experiment

**As a researcher**, I want to train on mechanically verifiable CAD tasks so that the paper tests RL in addition to SFT.

**Acceptance criteria**

* At least one RL/GRPO run uses deterministic verifier rewards.
* Candidate tasks include ranking, repair direction, multi-object pair detection, volume bucket, or clearance pass/fail.
* Binary-only RL is not the primary RL result unless dense tasks are unavailable.
* Rewards penalize invalid final-answer format.
* Ranking reward supports partial credit through pairwise ranking accuracy or Kendall tau.
* Repair reward executes the proposed transform and penalizes unnecessary movement.
* Evaluation compares the RL/GRPO model against answer-only SFT and counterfactual SFT.

---

## Story 7.10 — Evaluate tool-assisted upper bound

**As a researcher**, I want to compare closed-book models to tool-assisted CAD verification.

**Acceptance criteria**

* Tool-assisted pipeline executes code and computes exact answer.
* Tool-assisted result is clearly labelled as upper bound.
* Failure modes from execution are reported.

---

# Epic 8: Rendering and Multimodal Extensions

## Goal

Optionally support visual prompts and visual analysis for CAD assemblies.

---

## Story 8.1 — Generate object renders

**As a dataset builder**, I want rendered images of individual objects.

**Acceptance criteria**

For each example, optionally generate:

* Object A render
* Object B render
* Assembly render
* Transparent overlap render if possible

---

## Story 8.2 — Generate orthographic views

**As a dataset builder**, I want standard CAD-like views for multimodal evaluation.

**Acceptance criteria**

Views include:

* front
* top
* side
* isometric

Files are linked to dataset IDs.

---

## Story 8.3 — Create multimodal prompt variant

**As a researcher**, I want to compare code-only vs code-plus-image performance.

**Acceptance criteria**

* Prompt includes CadQuery script and one or more renders.
* Same labels are used as text-only task.
* Results are separately reported.

---

# Epic 9: Dataset Packaging and Release

## Goal

Package the dataset in a reproducible and usable form.

---

## Story 9.1 — Define final dataset schema

**As a dataset user**, I want a stable schema so that examples are easy to load.

**Acceptance criteria**

Schema includes:

* `id`
* `source`
* `split`
* `generator_id`
* `base_object_pair_id`
* `assembly_group_id`
* `counterfactual_group_id`
* `variant_id`
* `changed_parameter`
* `changed_value`
* `task_type`
* `prompt`
* `answer`
* `script`
* `labels`
* `diagnostics`
* `metadata`

---

## Story 9.2 — Export dataset files

**As a dataset user**, I want downloadable dataset files.

**Acceptance criteria**

Exports include:

```text
train.jsonl
validation.jsonl
test_random.jsonl
test_generator_heldout.jsonl
test_object_pair_heldout.jsonl
test_near_boundary.jsonl
```

Optional:

```text
test_topology_heldout.jsonl
test_operation_heldout.jsonl
renders/
metadata.parquet
```

---

## Story 9.3 — Create dataset card

**As a dataset user**, I want documentation for usage, limitations, and licensing.

**Acceptance criteria**

Dataset card includes:

* Motivation
* Task descriptions
* Label definitions
* Data sources
* Generation process
* Splits
* Known limitations
* Intended uses
* Out-of-scope uses
* License

---

## Story 9.4 — Create reproducibility scripts

**As a researcher**, I want others to regenerate or verify the dataset.

**Acceptance criteria**

Repository includes commands like:

```bash
python scripts/generate_dataset.py --config configs/intersectionqa_v1.yaml
python scripts/validate_dataset.py --input data/intersectionqa_v1
python scripts/evaluate_baseline.py --baseline aabb
```

---

## Story 9.5 — Add versioning

**As a dataset maintainer**, I want dataset versions to be traceable.

**Acceptance criteria**

* Dataset version is stored in each example.
* Code commit hash is stored in generation metadata.
* Config hash is stored.
* Release notes exist for v0.1 and v1.0.

---

# Epic 10: Paper Experiments

## Goal

Run the experiments needed to support the claims in the paper.

---

## Story 10.1 — Run dataset statistics

**As a paper author**, I want dataset statistics for the dataset section.

**Acceptance criteria**

Report includes:

* Number of examples per task
* Number of examples per split
* Label distribution
* Source distribution
* CadQuery operation distribution
* Difficulty distribution
* Object volume distribution
* Intersection-volume distribution
* Clearance distribution

---

## Story 10.2 — Run baseline comparison table

**As a paper author**, I want a main results table.

**Acceptance criteria**

Table includes:

* AABB
* OBB
* Convex hull
* Zero-shot LLM
* Few-shot LLM
* Fine-tuned code model
* Tool-assisted verifier

Reported across:

* random test
* generator-held-out
* object-pair / assembly-group held-out
* near-boundary hard
* topology-held-out if implemented

---

## Story 10.3 — Run task-family comparison

**As a paper author**, I want to show which tasks are hardest.

**Acceptance criteria**

Report model performance for:

* binary interference
* relation classification
* volume bucket
* clearance bucket
* ranking
* repair direction

---

## Story 10.4 — Run diagnostic subset analysis

**As a paper author**, I want to show that models fail beyond shortcut geometry.

**Acceptance criteria**

Analyze performance on:

* AABB-correct subset
* AABB-failing subset
* OBB-failing subset
* cavity-targeted subset
* near-boundary subset
* rotated-object subset
* boolean-heavy subset

---

## Story 10.5 — Run ablation on training data

**As a paper author**, I want to test which training data improves performance.

**Acceptance criteria**

Compare:

* no fine-tuning
* answer-only SFT
* rationale SFT
* counterfactual SFT
* pairwise/ranking counterfactual SFT
* mixed-task SFT
* verifier-guided RL/GRPO

---

## Story 10.6 — Analyze failure cases

**As a paper author**, I want qualitative examples for the discussion section.

**Acceptance criteria**

At least 10 representative failure cases are selected:

* touching mistaken as intersection
* cavity mistaken as collision
* rotation ignored
* wrong object dimensions inferred
* transform order mistake
* ranking magnitude error
* repair direction error

Each has prompt, ground truth, model answer, and short analysis.

---

# Epic 11: Paper Writing

## Goal

Write and polish the final research paper.

---

## Story 11.1 — Draft abstract

**As a paper author**, I want a concise abstract explaining the contribution.

**Acceptance criteria**

Abstract includes:

* Problem
* Dataset/task
* Labeling method
* Evaluation setup
* Key findings
* Release statement if applicable

---

## Story 11.2 — Draft introduction

**As a paper author**, I want to motivate why CAD-code spatial reasoning matters.

**Acceptance criteria**

Introduction explains:

* LLMs increasingly generate CAD/code.
* CAD correctness requires geometric reasoning.
* Interference and clearance are central assembly properties.
* Existing CAD benchmarks often evaluate final shape, not reasoning over assembly code.
* IntersectionQA fills this gap.

---

## Story 11.3 — Draft related work

**As a paper author**, I want to position the work against existing CAD and code-model datasets.

**Acceptance criteria**

Related work covers:

* Text-to-CAD
* CAD program generation
* CAD reconstruction datasets
* Code reasoning benchmarks
* Spatial reasoning benchmarks
* Tool-using CAD agents

---

## Story 11.4 — Draft dataset construction section

**As a paper author**, I want to describe how examples are generated and labelled.

**Acceptance criteria**

Section includes:

* Source ingestion
* Assembly generation
* Transform representation
* Exact labeling
* Difficulty categories
* Counterfactual generation
* Splits
* Filtering and validation

---

## Story 11.5 — Draft task section

**As a paper author**, I want to define all benchmark tasks clearly.

**Acceptance criteria**

Section includes:

* Prompt examples
* Answer formats
* Label definitions
* Task motivation
* Evaluation metrics

---

## Story 11.6 — Draft experiments section

**As a paper author**, I want to present model and baseline results.

**Acceptance criteria**

Section includes:

* Models evaluated
* Baselines
* Evaluation settings
* Main results table
* Split-specific results
* Task-specific results
* Ablations

---

## Story 11.7 — Draft discussion section

**As a paper author**, I want to explain what the results mean.

**Acceptance criteria**

Discussion covers:

* Where models succeed
* Where models fail
* Evidence of shortcut reasoning
* Closed-book versus tool-assisted reasoning
* Implications for CAD agents
* Dataset limitations

---

## Story 11.8 — Draft limitations section

**As a paper author**, I want to be explicit about limitations.

**Acceptance criteria**

Limitations include:

* CADEvolve source bias and generator-family leakage risk
* Limited synthetic fixtures may not cover every geometry edge case
* CadQuery/OpenCASCADE tolerance issues
* Closed-book setting may not reflect practical CAD workflows
* Possible leakage from source generators
* Limited coverage of real industrial assemblies
* Approximate repair labels for complex geometry

---

## Story 11.9 — Draft conclusion

**As a paper author**, I want to close with the contribution and future work.

**Acceptance criteria**

Conclusion summarizes:

* Dataset contribution
* Key experimental finding
* Future use for CAD-agent training
* Extensions to motion, constraints, and manufacturability

---

# Epic 12: Repository and Engineering Infrastructure

## Goal

Create a usable codebase for dataset generation, evaluation, and release.

---

## Story 12.1 — Create repository structure

**As an engineer**, I want a clean repo layout.

**Acceptance criteria**

Suggested structure:

```text
intersectionqa/
  configs/
  data/
  docs/
  intersectionqa/
    sources/
    geometry/
    generation/
    labeling/
    prompts/
    evaluation/
    rendering/
  scripts/
  tests/
  paper/
  README.md
```

---

## Story 12.2 — Add configuration system

**As an engineer**, I want generation runs controlled by config files.

**Acceptance criteria**

Config supports:

* source selection
* number of examples
* random seed
* task types
* difficulty mix
* split strategy
* thresholds
* export paths

---

## Story 12.3 — Add logging and failure tracking

**As an engineer**, I want failures to be visible and debuggable.

**Acceptance criteria**

Logs include:

* invalid scripts
* Boolean failures
* distance-computation failures
* skipped examples
* class balance
* generation throughput

---

## Story 12.4 — Add unit tests

**As an engineer**, I want core geometry and label logic tested.

**Acceptance criteria**

Tests cover:

* box-box intersection
* box-box touching
* box-box disjoint
* containment
* near-miss
* transform application
* bucket assignment
* prompt generation

---

## Story 12.5 — Add smoke-test dataset generation

**As an engineer**, I want a small reproducible dataset for CI.

**Acceptance criteria**

Command generates 100 examples.

```bash
python scripts/generate_dataset.py --config configs/smoke.yaml
```

CI validates schema and label consistency.

---

# Epic 13: Quality Assurance and Review

## Goal

Ensure the dataset and paper are credible, reproducible, and review-ready.

---

## Story 13.1 — Manual audit sample

**As a researcher**, I want to manually inspect examples to catch pipeline errors.

**Acceptance criteria**

* Random sample of at least 100 examples reviewed.
* Hard sample of at least 100 examples reviewed.
* Invalid or suspicious examples are categorized.
* Fixes are tracked.

---

## Story 13.2 — Reproducibility audit

**As a researcher**, I want to ensure generation is deterministic.

**Acceptance criteria**

* Same config and seed produce same IDs and labels.
* Different seeds produce different examples.
* Dataset version includes config hash.

---

## Story 13.3 — Leakage audit

**As a researcher**, I want to reduce train/test leakage.

**Acceptance criteria**

* No shared generator IDs between generator-held-out train/test.
* No shared base object-pair IDs between object-pair-held-out train/test.
* No shared assembly-group IDs between assembly-held-out train/test.
* Near-duplicate scripts are detected where possible.
* Counterfactual groups are not split across train/test unless intentionally used for a specific experiment.
* Chamfer-distance or point-cloud geometry filtering is documented as optional future leakage control, not a v1 release blocker.

---

## Story 13.4 — Reviewer-readiness checklist

**As a paper author**, I want to pre-empt obvious reviewer objections.

**Acceptance criteria**

Checklist answers:

* Why not just use CAD tools?
* Why closed-book reasoning?
* Why CadQuery?
* How are labels generated?
* Are bounding-box shortcuts enough?
* Does the CADEvolve-derived dataset generalize beyond its source generator families?
* Are splits leakage-resistant?
* What are the known limitations?

---

# Suggested Milestones

## Milestone 1 — Specification freeze

**Goal:** Freeze the dataset and paper scope.

Deliverables:

* `benchmark-task-spec.md`
* `label_rules.md`
* `paper-spec.md`
* Initial repo structure

---

## Milestone 2 — Geometry prototype

**Goal:** Generate and label simple two-object examples.

Deliverables:

* Primitive object generator
* Transform system
* Exact intersection labels
* Binary prompt generator
* 1k-example smoke dataset

---

## Milestone 3 — MVP task prototype

**Goal:** Support the v0.1 benchmark tasks.

Deliverables:

* Binary task
* Relation task
* Volume bucket task
* Dataset schema v0.1

Clearance, ranking, repair-direction, tolerance-fit, and multi-object prompt
families remain reserved in the schema and task spec, but are later milestones
rather than v0.1 MVP blockers.

---

## Milestone 4 — Hard examples and splits

**Goal:** Make the benchmark scientifically useful.

Deliverables:

* Boundary-targeted generation
* Cavity-targeted generation
* Counterfactual groups
* Counterfactual individual, pairwise, and ranking prompt formats
* Random split
* Generator-held-out split
* Object-pair / assembly-group held-out split
* Near-boundary hard split

---

## Milestone 5 — Baseline evaluation

**Goal:** Establish benchmark difficulty.

Deliverables:

* AABB baseline
* OBB baseline
* Convex-hull baseline
* Zero-shot model eval
* Few-shot model eval
* Initial results table

---

## Milestone 6 — Training experiments

**Goal:** Show dataset usefulness for improving models.

Deliverables:

* Answer-only SFT
* Rationale SFT
* Counterfactual SFT
* Small verifier-guided RL/GRPO experiment
* Evaluation across all splits
* Ablation table

---

## Milestone 7 — Paper draft

**Goal:** Produce a complete paper draft.

Deliverables:

* Abstract
* Introduction
* Related work
* Dataset construction
* Tasks
* Experiments
* Discussion
* Limitations
* Conclusion

---

## Milestone 8 — Release candidate

**Goal:** Prepare public release.

Deliverables:

* Dataset card
* Clean repository
* Reproducibility scripts
* Final dataset export
* Final paper PDF
* Final results tables

---

# Suggested Sprint Plan

## Sprint 1 — Scope and CADEvolve smoke prototype

**Duration:** 1 week

Stories:

* 1.1 Define benchmark thesis
* 1.2 Define task families
* 2.2 Build source loader interface
* 2.4 Validate source CAD objects
* 3.1 Implement rigid transform representation
* 3.2 Implement assembly script generator
* 4.1 Compute exact intersection volume
* 5.1 Implement binary interference prompt generator
* 12.1 Create repository structure

Sprint output:

* Working CADEvolve object-validation smoke path, using `cadevolve.tar` when available.
* Minimal primitive box/cylinder golden cases for label verification only.
* First binary QA JSONL smoke export with CADEvolve examples where available.

---

## Sprint 2 — Labeling and relation tasks

**Duration:** 1 week

Stories:

* 4.2 Compute minimum clearance distance
* 4.3 Implement relation classifier
* 4.5 Detect containment
* 5.2 Implement relation-classification prompt generator
* 5.3 Implement volume-bucket prompt generator
* 5.4 Implement clearance-bucket prompt generator
* 12.4 Add unit tests

Sprint output:

* Validated labels for binary, relation, and volume-bucket tasks.
* Minimum-distance labels are available where needed to distinguish `disjoint`,
  `touching`, and `near_miss`; clearance-bucket prompts remain P1.

---

## Sprint 3 — Source ingestion and generated diversity

**Duration:** 1–2 weeks

Stories:

* 2.1 Select initial CAD program sources
* 2.2 Build source loader interface
* 2.3 Normalize CadQuery object functions
* 2.4 Validate source CAD objects
* 3.3 Generate random broad-placement examples
* 10.1 Run dataset statistics

Sprint output:

* First diverse dataset from real or semi-real CadQuery programs.

---

## Sprint 4 — Hard examples

**Duration:** 1–2 weeks

Stories:

* 3.4 Generate boundary-targeted examples
* 3.5 Generate cavity-targeted examples
* 3.6 Generate counterfactual groups
* 3.7 Materialize counterfactual prompt formats
* 4.4 Compute bounding-box diagnostics
* 6.6 Implement near-boundary hard split

Sprint output:

* Hard diagnostic dataset that cannot be solved by AABB alone.

---

## Sprint 5 — Reserved task extensions

**Duration:** 1 week

Stories:

* 3.8 Generate multi-object assemblies
* 5.5 Implement pairwise multi-object prompt generator
* 5.6 Implement ranking prompt generator
* 5.7 Implement repair-direction prompt generator
* 5.8 Implement tolerance-aware fit prompt generator

Sprint output:

* Optional reserved task families implemented only after the v0.1 code-only
  two-object pipeline, splits, validation, and baselines are stable.

---

## Sprint 6 — Splits and packaging

**Duration:** 1 week

Stories:

* 6.1 Implement random split
* 6.2 Implement generator-held-out split
* 6.3 Implement object-pair and counterfactual group holdout
* 6.7 Balance dataset classes
* 9.1 Define final dataset schema
* 9.2 Export dataset files

Sprint output:

* Dataset v0.1 with leakage-resistant train/validation/test splits.

---

## Sprint 7 — Baselines

**Duration:** 1 week

Stories:

* 7.1 Implement AABB baseline
* 7.2 Implement OBB baseline
* 7.3 Implement convex-hull baseline
* 7.4 Evaluate zero-shot LLMs
* 7.5 Evaluate few-shot LLMs
* 10.2 Run baseline comparison table

Sprint output:

* Initial benchmark results.

---

## Sprint 8 — Fine-tuning experiments

**Duration:** 1–2 weeks

Stories:

* 7.6 Fine-tune small code model
* 7.7 Fine-tune with structured rationales
* 7.8 Evaluate counterfactual training
* 7.9 Run verifier-guided RL/GRPO experiment
* 10.5 Run ablation on training data
* 10.6 Analyze failure cases

Sprint output:

* Evidence that IntersectionQA is useful for training, not only evaluation.

---

## Sprint 9 — Paper writing

**Duration:** 1–2 weeks

Stories:

* 11.1 Draft abstract
* 11.2 Draft introduction
* 11.3 Draft related work
* 11.4 Draft dataset construction section
* 11.5 Draft task section
* 11.6 Draft experiments section
* 11.7 Draft discussion section
* 11.8 Draft limitations section
* 11.9 Draft conclusion

Sprint output:

* Full paper draft.

---

## Sprint 10 — Release and review

**Duration:** 1 week

Stories:

* 9.3 Create dataset card
* 9.4 Create reproducibility scripts
* 9.5 Add versioning
* 13.1 Manual audit sample
* 13.2 Reproducibility audit
* 13.3 Leakage audit
* 13.4 Reviewer-readiness checklist

Sprint output:

* Release candidate for dataset, code, and paper.

---

# MVP Scope

If you want the smallest credible version, implement only this first:

## MVP dataset

* Binary interference
* Relation classification
* Volume bucket
* Boundary-targeted examples
* Counterfactual groups
* Individual-row and pairwise counterfactual prompts
* Random split
* Generator-held-out split
* Object-pair / assembly-group holdout
* Near-boundary hard split

## MVP baselines

* AABB baseline
* OBB or approximate OBB baseline
* One zero-shot LLM
* One fine-tuned open code model
* One small verifier-guided RL/GRPO run if training budget allows
* Exact CAD-kernel upper bound

## MVP paper claim

> IntersectionQA shows that current language/code models struggle to infer geometric interference from CAD programs, especially on near-boundary, rotated, and concavity-heavy examples, and that counterfactual SFT plus verifier-scored training can improve this capability.

---

# Backlog Priority

## P0 — Required for paper viability

* Task specification
* CadQuery object validation
* Assembly generation
* Exact intersection labels
* Binary task
* Relation task
* Near-boundary examples
* Counterfactual groups
* Random, generator-held-out, object-pair-held-out, and hard splits
* AABB baseline
* Zero-shot model evaluation
* Dataset statistics
* Paper draft

## P1 — Strongly recommended

* Volume buckets
* Clearance buckets
* Pairwise/ranking counterfactual prompt formats
* Cavity-targeted examples
* OBB baseline
* Fine-tuning experiment
* Verifier-guided RL/GRPO experiment
* Failure-case analysis
* Dataset card

## P2 — Nice to have

* Multi-object assemblies
* Ranking task
* Repair-direction task
* Tool-assisted model setting
* Rendered images
* Multimodal evaluation
* Convex-hull baseline
* Topology-held-out split
* Operation-held-out split

## P3 — Future work

* Chamfer / point-cloud geometry leakage filtering
* Motion/interference over trajectories
* Assembly sequence feasibility
* Constraint satisfaction
* Manufacturability checks
* Tolerance stack-up reasoning
* CAD agent repair loops
* Build123D variant
* Fusion360/STEP-derived variant
