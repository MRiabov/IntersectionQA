Note: This specification describes IntersectionQA v0.1. Shared geometry, split,
provenance, and verifier concepts may be reused by IntersectionEdit, but task
semantics here are QA-specific unless stated otherwise.

Your core idea is strong because it isolates a capability that CAD-generation papers often hide inside “final model similarity”: spatial reasoning over exact executable CAD programs.

The main improvement is to frame IntersectionQA not as “another CAD benchmark”, but as a controlled geometry-grounding benchmark for code models: can a model mentally execute CadQuery enough to infer contact, penetration, clearance, and relative transforms?

CADEvolve is the primary source corpus for IntersectionQA because it provides executable CadQuery programs at scale: the paper reports 8k complex parametric generators expanded into about 1.3M scripts, with broad CadQuery operation coverage. IntersectionQA should derive its real benchmark examples from CADEvolve programs, while using synthetic primitives only as golden fixtures, smoke-test edge cases, and debugging examples. CADEvolve-derived examples still need duplicate control and leakage-safe grouping by generator family where available. ([arXiv][1])

This document captures the paper ambition and research narrative. For
implementation, the v0.1 MVP is narrower than the full paper roadmap:
`specs/benchmark-task-spec.md`, `specs/label_rules.md`, `specs/schema.md`, and
`specs/generation_policy.md` are canonical for task scope, labels, schema fields, and
generation rules. In particular, v0.1 implements `binary_interference`,
`relation_classification`, and `volume_bucket`; clearance, pairwise, ranking,
repair, and tolerance-fit tasks are reserved extensions unless a later release
explicitly promotes them.

## 1. The missing “usefulness” angle

The dataset becomes much more useful if it maps to real CAD workflows, not just abstract collision trivia.

The most useful framing is:

> Given CAD code and assembly transforms, determine whether two or more parts violate assembly constraints: interference, contact, required clearance, or fit.

This connects directly to engineering tasks:

| Task                           | Why useful                                                           |
| ------------------------------ | -------------------------------------------------------------------- |
| Interference detection         | Detects impossible assemblies and manufacturing mistakes             |
| Clearance estimation           | Needed for tolerances, motion envelopes, fit checks                  |
| Contact classification         | Distinguishes collision from intentional mating/contact              |
| Assembly ordering              | Helps reason about whether parts can be inserted or assembled        |
| Constraint repair              | Moves from diagnosis to design assistance                            |
| Ranking by intersection volume | Trains geometric magnitude reasoning, not just binary classification |

A better paper claim would be:

> Current LLMs can generate plausible CAD code, but they are weak at reasoning over the geometry induced by CAD programs. IntersectionQA tests whether models understand the spatial consequences of parametric CAD code and assembly transforms.

This is more compelling than “LLMs are bad at CAD modelling” generally.

## 2. Expand labels beyond yes/no

Binary “intersects: yes/no” is useful, but too coarse. It can be solved partly by shortcuts.

I would define several task tiers.

### Tier A: binary interference

Question:

> Do object A and object B have non-zero volume intersection after assembly?

Answer:

```text
yes
```

or

```text
no
```

Important: define “intersection” as positive volume overlap, not just touching faces/edges.

Otherwise your labels become ambiguous. Two cubes sharing a face have zero intersection volume but are in contact.

### Tier B: relation classification

Use 4–6 labels instead of only yes/no:

```text
disjoint
touching
intersecting
contained
near_miss
invalid
```

Where:

* `disjoint`: no contact, positive clearance
* `touching`: zero-volume contact
* `intersecting`: positive overlap
* `contained`: one object is fully inside another
* `near_miss`: small positive clearance below threshold
* `invalid`: script fails or produces non-solid geometry

This is far more useful for CAD engineering. It also prevents models from collapsing “touching” and “intersecting”.

### Tier C: scalar intersection volume

Ask for approximate normalized volume:

```text
intersection_volume / min(volume_a, volume_b)
```

This is better than raw volume because raw volume depends heavily on object scale.

Suggested answer formats:

```text
0.00
0.13
0.72
```

or bucketed:

```text
none
tiny
small
medium
large
contained
```

Bucketed labels may be better for LLMs and more robust to small kernel differences.

### Tier D: clearance distance

For non-intersecting pairs:

> What is the minimum distance between the two solids?

Use buckets:

```text
touching
<1mm
1-5mm
5-20mm
>20mm
```

This makes the dataset useful for actual mechanical fit checking.

### Tier E: ranking

Your “DBCA” ranking idea is good, but make it ranking over normalized overlap, not raw volume. Otherwise a large object pair will dominate even if the relative interference is trivial.

Better prompt:

> Rank the assemblies from highest to lowest normalized interference volume, where normalized interference is intersection volume divided by the smaller part volume.

Answer:

```text
DBCA
```

This tests comparative geometric reasoning and reduces scale artifacts.

### Tier F: repair action

This is likely the most useful training task.

Question:

> The assembly has interference. Which minimal translation along X, Y, or Z would remove the interference?

Answer options:

```text
+x
-x
+y
-y
+z
-z
none
```

Or scalar:

```text
translate object_b by +3.2 mm along X
```

This converts the dataset from passive QA into CAD-agent training data.

## 3. Add counterfactual groups

IntersectionQA should include counterfactual groups: sets of examples that share the same base CAD objects and assembly structure but differ in exactly one controlled parameter.

Example:

```python
box_a.translate((0, 0, 0))
box_b.translate((9.9, 0, 0))  # intersects
```

versus:

```python
box_b.translate((10.1, 0, 0))  # does not intersect
```

This is extremely valuable because it forces sensitivity to exact geometry and transforms. It also makes the training and evaluation story clearer: the same source group can be materialized as individual QA rows, pairwise comparison prompts, or ranking prompts.

### Default format: separate rows with shared metadata

For normal supervised fine-tuning and standard evaluation, the two variants should usually be separate dataset rows.

```json
{
  "id": "intersectionqa_000123",
  "counterfactual_group_id": "cfg_00042",
  "variant_id": "cfg_00042_v1",
  "base_object_pair_id": "pair_00091",
  "changed_parameter": "translation_x",
  "changed_value": 9.9,
  "task_type": "binary_interference",
  "answer": "yes",
  "labels": {
    "relation": "intersecting",
    "intersection_volume": 1.0,
    "minimum_distance": 0.0
  }
}
```

```json
{
  "id": "intersectionqa_000124",
  "counterfactual_group_id": "cfg_00042",
  "variant_id": "cfg_00042_v2",
  "base_object_pair_id": "pair_00091",
  "changed_parameter": "translation_x",
  "changed_value": 10.1,
  "task_type": "binary_interference",
  "answer": "no",
  "labels": {
    "relation": "disjoint",
    "intersection_volume": 0.0,
    "minimum_distance": 0.1
  }
}
```

Use this format for:

* SFT
* standard evaluation
* binary classification
* relation classification
* volume and clearance bucket prediction

### Derived format: pairwise comparison prompt

For contrastive training and hard sensitivity evaluation, two variants from the same group can be placed into one prompt.

```text
You are given two assemblies.

Assembly A:
box_b is translated to (9.9, 0, 0)

Assembly B:
box_b is translated to (10.1, 0, 0)

Which assembly has positive-volume interference?

Answer:
A
B
both
neither
```

This format is useful for counterfactual SFT and RL because the verifier can score the selected option directly.

### Derived format: ranking prompt

For 3-5 variants:

```text
Rank these assemblies by normalized intersection volume, highest to lowest.

A: box_b.translate((9.5, 0, 0))
B: box_b.translate((9.9, 0, 0))
C: box_b.translate((10.0, 0, 0))
D: box_b.translate((10.2, 0, 0))

Answer format: four letters.
```

Ranking prompts provide denser reward than binary prompts because partial order errors can receive partial credit.

All examples from the same `counterfactual_group_id` must stay in the same train/validation/test split unless the experiment is explicitly designed to test interpolation. Otherwise the dataset leaks near-identical variants across splits.

Recommended counterfactual dimensions:

| Parameter changed     | Example                            |
| --------------------- | ---------------------------------- |
| translation           | ± small epsilon near boundary      |
| rotation              | 44° vs 46°                         |
| object size           | radius 4.9 vs 5.1                  |
| wall thickness        | shell thickness changes            |
| boolean cut depth     | hole becomes through-hole vs blind |
| fillet/chamfer radius | local contact changes              |
| assembly order        | transformed object names swapped   |

This gives you “hard negatives” and “hard positives”, which are far more valuable than random pairs.

## 4. Control difficulty explicitly

You should not create only random intersections. Most random placements are obviously disjoint or obviously intersecting. The benchmark should have calibrated difficulty.

Suggested difficulty levels:

### Level 1: simple CADEvolve objects and fixture primitives

CADEvolve programs with simple operation signatures such as boxes, cylinders, simple extrusions, or similarly compact solids. Fixture primitives may be used for golden tests.

```python
box(10, 10, 10)
cylinder(radius=3, height=10)
```

Mostly tests bounding-box reasoning and validates that CADEvolve execution, normalization, and prompt construction work.

### Level 2: rotated simple objects

Same CADEvolve/simple-fixture objects, but with controlled Euler rotations.

Tests transform composition.

### Level 3: compound solids

Fillets, chamfers, holes, cuts, unions.

Tests actual CadQuery program understanding.

### Level 4: concavity and internal voids

Objects with holes, shells, cavities, brackets, clamps.

This is where bounding-box heuristics fail.

### Level 5: multi-part assemblies

3–10 parts, ask which pairs intersect or rank all pairwise intersections.

### Level 6: near-boundary cases

Touching, epsilon-clearance, epsilon-interference.

This is the most useful for evaluation, but the most label-sensitive.

## 5. Avoid leakage from generated variants

If you use CADEvolve, DeepCAD-derived code, or generated CadQuery corpora, avoid splitting randomly by script. CADEvolve specifically expands generators into many scripts, so script-level random splits could leak near-identical geometry/program patterns between train and test. ([Hugging Face][2])

Random splitting is useful only as a sanity check. The credible split is group-based.

Minimum credible holdout for v1:

1. Random split for smoke-test and sanity-check comparisons.
2. Generator-family split, where examples from the same source generator never cross train/test.
3. Object-pair or assembly-group split, where variants using the same base object pair and transform family stay together.
4. Counterfactual-group split, where all variants from the same counterfactual group stay together.
5. Near-boundary hard test set, built from touching, epsilon-clearance, epsilon-interference, and AABB-failing cases.

The split metadata should make these constraints explicit:

```json
{
  "source_id": "cadevolve_...",
  "generator_id": "gen_042",
  "topology_class": "bracket",
  "operation_signature": ["box", "extrude", "cut", "fillet"],
  "base_object_pair_id": "pair_00091",
  "assembly_group_id": "objectA_gen_042__objectB_gen_117__transform_family_003",
  "counterfactual_group_id": "cfg_00042",
  "split_group": "gen_042"
}
```

Stronger leakage controls can be added later:

* topology-held-out split
* operation-held-out split
* token or AST similarity filtering over CadQuery scripts
* geometry similarity clustering over sampled point clouds
* assembly-level geometry similarity filtering

Do not make Chamfer-distance filtering a v1 dependency. It is useful for leakage audits, but it adds meshing, point sampling, normalization, nearest-neighbor search, and threshold calibration. For the first paper, generator-family holdout plus intact object-pair, assembly, and counterfactual groups is a much better complexity-to-value tradeoff.

If geometry similarity filtering is added later, use it as a complement rather than a replacement for group-based splitting:

* object-shape similarity: center each object independently by center of mass or bounding-box center, scale by bounding-box diagonal, optionally PCA-align, then compare point clouds
* assembly-configuration similarity: center and scale the whole assembly jointly while preserving relative object transforms, then compare the combined point cloud

This distinguishes duplicate object geometry from duplicate assembled configurations.

## 6. Include “shortcut baselines”

To prove the benchmark is meaningful, compare models against non-LLM geometric heuristics.

Useful baselines:

| Baseline                                 | Purpose                                            |
| ---------------------------------------- | -------------------------------------------------- |
| AABB overlap from approximate dimensions | Shows whether tasks are solvable by bounding boxes |
| Oriented bounding box overlap            | Stronger geometric shortcut                        |
| Convex hull approximation                | Tests if concavity matters                         |
| Text/code embedding classifier           | Tests memorization / pattern leakage               |
| Small code model fine-tuned on labels    | Training baseline                                  |
| LLM zero-shot / few-shot                 | General reasoning baseline                         |
| LLM with rendered images                 | Multimodal comparison                              |
| LLM with tool execution                  | upper-bound agent baseline                         |

This is important because if AABB solves 90% of the benchmark, reviewers will say your task is too easy.

You want subsets where:

* AABB works
* OBB works but AABB fails
* convex approximation fails
* only exact B-Rep intersection works

That gives you a clean difficulty ladder.

## 7. Label generation pipeline

A robust pipeline could be:

1. Load or generate two CadQuery objects.
2. Validate each object:

   * script executes
   * object is a solid
   * volume > threshold
   * geometry is not degenerate
3. Apply assembly transforms.
4. Compute:

   * volume A
   * volume B
   * intersection volume
   * minimum distance
   * contact type
   * bounding-box overlap
   * normalized overlap
5. Store task variants:

   * binary
   * relation label
   * scalar/bucket volume
   * ranking
   * repair
6. Render optional views:

   * object A alone
   * object B alone
   * assembly
   * transparent overlap visualization
7. Save provenance:

   * source dataset
   * generator ID
   * transform seed
   * CadQuery operations used
   * difficulty bucket

The key is to store far more metadata than the prompt exposes. That lets you slice the benchmark later.

Suggested record schema:

```json
{
  "id": "intersectionqa_000123",
  "source": "cadevolve",
  "task_type": "binary_interference",
  "generator_id": "gen_042__gen_117",
  "base_object_pair_id": "pair_00091",
  "assembly_group_id": "pair_00091__transform_family_003",
  "counterfactual_group_id": "cfg_00042",
  "variant_id": "cfg_00042_v1",
  "changed_parameter": "translation_x",
  "changed_value": 8.7,
  "script": "...",
  "transform_a": {
    "translation": [0, 0, 0],
    "rotation_xyz_deg": [0, 0, 0],
    "rotation_order": "XYZ"
  },
  "transform_b": {
    "translation": [8.7, 0, 0],
    "rotation_xyz_deg": [0, 0, 15],
    "rotation_order": "XYZ"
  },
  "labels": {
    "volume_a": 1000.0,
    "volume_b": 800.0,
    "intersection_volume": 47.2,
    "normalized_intersection": 0.059,
    "minimum_distance": 0.0,
    "relation": "intersecting"
  },
  "diagnostics": {
    "aabb_overlap": true,
    "exact_overlap": true,
    "label_status": "ok",
    "failure_reason": null
  },
  "difficulty_tags": ["near_boundary", "rotated", "cadevolve_compound"],
  "cadquery_ops": ["box", "extrude", "cut", "fillet"],
  "answer": "yes",
  "metadata": {
    "split_group": "generator_042",
    "generator_ids": ["gen_042", "gen_117"]
  }
}
```

## 8. Prompt design improvements

Your prompt should be stricter and less ambiguous.

Current issue: `.move(...)` is not the usual clear CadQuery transform syntax, and your example repeats `object_a_name()` twice instead of placing object A and object B. Also, Euler angle convention must be specified.

Better prompt:

````text
You are given two CadQuery object-construction functions and an assembly function.
The assembly function applies rigid transforms to place the two solids in world coordinates.

Assume:
- Units are millimetres.
- Euler rotations are XYZ order, in degrees.
- "interference" means positive-volume overlap.
- Merely touching at a face, edge, or point is not interference.
- Do not execute code.
- Think through the geometry, then output exactly one final label: yes or no.

Script:
```python
...
````

Question:
After the transforms are applied, do object_a and object_b have positive-volume interference?

Final answer format: <answer>yes</answer> or <answer>no</answer>

````

For numeric tasks:

```text
Return the normalized intersection volume:
intersection_volume / min(volume(object_a), volume(object_b)).

Use one of:
0
(0, 0.01]
(0.01, 0.05]
(0.05, 0.20]
(0.20, 0.50]
>0.50
````

Buckets are easier to evaluate and less brittle.

## 9. Add natural-language explanations, but do not grade them as primary

Allow chain-of-thought internally, but for public datasets do not require long hidden reasoning as the label. Instead, you can include short rationales generated from geometry metadata.

Example rationale:

```text
The two boxes overlap along X by 2 mm, along Y by 10 mm, and along Z by 10 mm, so their intersection has positive volume.
```

This supports supervised fine-tuning without relying on opaque model-generated reasoning.

However, be careful: if rationales are templated too simply, models may learn templates rather than geometry.

Better:

* Train on answer-only and short-rationale variants.
* Evaluate answer-only.
* Include a rationale-quality subset manually inspected or generated from symbolic geometry for simple primitives.

## 10. Create diagnostic subsets

This is where the paper becomes more useful.

Suggested subsets:

| Subset                  | What it diagnoses                             |
| ----------------------- | --------------------------------------------- |
| Primitive-AABB          | basic dimension reasoning                     |
| Rotated-OBB             | rotation and transform reasoning              |
| Boolean-cut             | understanding holes/cavities                  |
| Thin-wall               | shell and wall-thickness reasoning            |
| Contact-vs-interference | zero-volume vs positive-volume distinction    |
| Near-miss               | numerical precision and tolerance reasoning   |
| Containment             | object inside cavity/object                   |
| Symmetry traps          | misleading code names or symmetric dimensions |
| Multi-object            | pairwise assembly reasoning                   |
| Repair                  | actionable CAD correction                     |

Then report per-subset accuracy. A single aggregate score will hide the interesting failures.

## 11. Training methods better than your initial proposal

You proposed QA-style supervised training. That is valid, but not necessarily the strongest.

### Method 1: geometry-verifier SFT

Train a model on:

```text
CadQuery code + transforms -> relation label / volume bucket
```

This is the simplest.

Use curriculum:

1. CADEvolve objects with simple operation signatures
2. CADEvolve objects with translations and single-axis rotations
3. CADEvolve compound/boolean-heavy objects
4. CADEvolve near-boundary and counterfactual cases
5. minimal synthetic primitive fixtures for golden validation only
6. multi-object assemblies

Curriculum is likely important because jumping directly to complex CadQuery may teach weak correlations.

### Method 2: contrastive counterfactual training

This is probably superior for learning exact spatial sensitivity.

Input pairs:

```text
Example A: translation x = 9.9 -> intersects
Example B: translation x = 10.1 -> does not intersect
```

Train the model to identify which one intersects, or rank them.

This directly targets the weakness you care about.

### Method 3: process supervision from symbolic traces

For generated simple/medium examples, produce exact intermediate facts:

```text
object_a x-range: [-5, 5]
object_b x-range: [3, 13]
overlap_x: 2
overlap_y: 10
overlap_z: 10
positive volume overlap: yes
```

Then train the model to reason through bounding boxes, transforms, and overlaps.

For complex objects, traces can include:

```text
AABB overlap: yes
convex hull overlap: yes
exact B-Rep intersection: no
reason: object_b passes through a cutout cavity
```

This is more useful than generic chain-of-thought.

### Method 4: verifier-guided RL / GRPO / rejection sampling

IntersectionQA is unusually suitable for RL because the answer can be scored mechanically. Have the model answer, parse the final response, and use exact CadQuery/OpenCASCADE labels or a deterministic verifier to score the completion.

Reward:

* correct yes/no
* correct relation class
* correct volume bucket
* correct clearance bucket
* correct pair set in multi-object assemblies
* correct ranking over counterfactual variants
* repair transform that actually removes interference
* penalize invalid final format
* optionally reward calibrated confidence

For GRPO-style training, sample multiple completions for the same prompt and score each completion with the verifier. Binary prompts are valid but sparse; ranking, repair, and multi-pair prompts provide better reward signal.

Example rewards:

```python
def binary_reward(pred, truth):
    return 1.0 if pred == truth else 0.0
```

```python
def volume_bucket_reward(pred_bucket, true_bucket):
    bucket_error = abs(pred_bucket.index - true_bucket.index)
    return max(0.0, 1.0 - 0.25 * bucket_error)
```

```python
def ranking_reward(pred_order, true_order):
    return pairwise_ranking_accuracy(pred_order, true_order)
```

```python
def repair_reward(predicted_transform, assembly):
    repaired = apply_transform(assembly, predicted_transform)

    if has_interference(repaired):
        return -1.0

    movement = transform_distance(predicted_transform)
    return 1.0 / (1.0 + movement)
```

SFT is still useful as a warmup because it teaches the prompt format, output schema, and basic geometry vocabulary. The intended training story is:

1. answer-only SFT on clean examples
2. counterfactual SFT on near-identical variants
3. verifier-guided RL/GRPO on ranking, repair, pairwise-interference, volume-bucket, and clearance tasks

This is attractive because labels are cheap once the geometry pipeline exists and because RL optimizes directly against the same CAD-kernel semantics used for evaluation.

### Method 5: tool-use agent training

For CAD agents, the best model may not be the one that mentally executes everything. A practical engineering assistant should know when to call a geometry kernel.

Train two modes:

1. no-tool mode: answer from code only
2. tool mode: write/check code, call intersection verifier, summarize result

Your benchmark can explicitly separate:

| Setting       | Meaning                                              |
| ------------- | ---------------------------------------------------- |
| closed-book   | no execution, tests internal spatial reasoning       |
| tool-assisted | can execute/intersect, tests CAD agent workflow      |
| hybrid        | model predicts first, then decides whether to verify |

The closed-book task is scientifically interesting; the tool-assisted task is practically useful.

### Method 6: multimodal augmentation

CADEvolve includes rendered geometry/code pairings, and recent CAD work often uses rendered or visual grounding. ([arXiv][1])

For IntersectionQA, generate:

* code-only input
* image-only input
* code + orthographic render
* code + transparent assembly render
* code + exploded-view render

Then compare whether images help. This matters because real CAD assistants often have both code/feature tree and viewport access.

## 12. Stronger benchmark tasks

Here are task templates I would include.

### Binary interference

```text
Given the CadQuery script and assembly transforms, do object_a and object_b have positive-volume interference?
Answer only yes or no.
```

### Contact vs interference

```text
Classify the relation between object_a and object_b:
A. disjoint
B. touching only
C. positive-volume interference
D. one object contained in the other
```

### Pair selection

```text
Which pair of objects has interference?
A. gear-housing
B. shaft-bearing
C. bracket-cover
D. none
```

### All-pairs matrix

```text
For objects A, B, C, D, output all interfering pairs.
Format: AB, AC, AD, BC, BD, CD, or none.
```

### Ranking

```text
Rank assemblies A-D by normalized intersection volume from largest to smallest.
Output only four letters.
```

### Minimal repair

```text
Object B intersects object A. Which single-axis translation direction most directly reduces the interference?
A. +X
B. -X
C. +Y
D. -Y
E. +Z
F. -Z
```

### Tolerance-aware fit

```text
The required clearance is 2 mm. Does the assembly satisfy the clearance requirement?
Answer yes or no.
```

This is highly engineering-relevant.

## 13. Be careful with exact geometry kernels

CadQuery uses OpenCASCADE under the hood, and exact Boolean/intersection operations can be sensitive to tolerance, invalid solids, tiny sliver faces, and near-coincident surfaces. You need a label policy.

Define:

```text
epsilon_volume = 1e-6 * min(volume_a, volume_b)
epsilon_distance = 1e-4 mm or 1e-3 mm
```

Then:

```text
if intersection_volume > epsilon_volume:
    intersecting
elif min_distance <= epsilon_distance:
    touching
else:
    disjoint
```

Also store raw values so future users can change thresholds.

## 14. Dataset generation strategy

A good generation strategy is not just “sample two objects and random transforms”.

Use a balanced mixture:

### Random broad sampling

Good for diversity.

```text
sample object_a
sample object_b
sample random SE(3) transforms
label exact relation
```

But this will overproduce obvious disjoint cases.

### Boundary-targeted sampling

Better.

1. Compute bounding boxes.
2. Place objects so bounding boxes almost overlap.
3. Randomly perturb.
4. Use exact Boolean result.
5. Keep examples near decision boundary.

### Penetration-targeted sampling

Generate known overlaps by intentionally placing object centers close together.

### Contact-targeted sampling

Place surfaces exactly adjacent, then perturb by ±ε.

### Cavity-targeted sampling

Use objects with holes/cutouts, place second object through/near the void.

This creates examples where bounding boxes say “collision” but exact solids say “no collision”.

## 15. Main paper contributions

I would formulate contributions like this:

1. IntersectionQA, a benchmark for spatial reasoning over executable CadQuery assemblies.
2. A scalable labeling pipeline using exact CAD-kernel operations to derive interference, contact, clearance, and normalized intersection volume.
3. A difficulty-controlled dataset with counterfactual near-boundary examples.
4. A diagnostic evaluation of LLMs, code models, VLMs, and geometry heuristics.
5. Training results showing that counterfactual and verifier-guided training improve CAD spatial reasoning.

That is a cleaner paper than “we made some yes/no questions”.

## 16. What reviewers may attack

Anticipate these.

### “Why not just run CAD collision detection?”

Answer:

> The goal is not to replace geometric kernels. The goal is to measure and improve whether CAD-code models understand the geometry they generate, especially in closed-loop design settings where models propose, inspect, and repair assemblies.

### “LLMs should use tools, not mentally execute geometry.”

Answer:

> We evaluate both closed-book and tool-assisted settings. Closed-book performance measures internal geometric grounding; tool-assisted performance measures practical CAD-agent capability.

### “Binary labels are too simple.”

Answer by including relation labels, clearance, volume buckets, ranking, and repair tasks.

### “Dataset can be solved by bounding boxes.”

Answer by including diagnostic subsets where AABB/OBB/convex hull baselines fail.

### “Synthetic data may not reflect real assemblies.”

Answer by making CADEvolve the primary released source corpus. Synthetic primitives and procedural motifs should be limited to golden tests, smoke/debug fixtures, and a small number of audited edge cases such as touching boxes or AABB/exact disagreement. Do not build a separate full synthetic CAD corpus before using CADEvolve.

## 17. Best minimal version for a first paper

For a tight first paper, do not overbuild. I would ship:

1. 100k binary examples
2. 20k relation-class examples
3. 20k near-boundary counterfactual examples
4. 5k ranking examples
5. 5k repair-direction examples
6. four test splits:

   * random
   * generator-held-out
   * object-pair / assembly-group held-out
   * near-boundary hard set

Models to evaluate:

* GPT-5.5 / Claude / Gemini class models if available
* Qwen coder model
* small open code model fine-tuned
* VLM with rendered image
* AABB/OBB baselines
* tool-assisted CAD-kernel agent as upper bound

Training experiments:

* SFT answer-only
* SFT with structured traces
* counterfactual pair training
* verifier-guided RL/GRPO on at least one dense task such as ranking, repair, or multi-pair detection

## 18. The highest-value improvement

The single best improvement is:

> Make IntersectionQA counterfactual and diagnostic, not just large.

A smaller dataset with hard, controlled, near-boundary, leakage-resistant examples will be more publishable than a million random yes/no intersections.

The second best improvement is:

> Include repair and clearance tasks.

That makes it directly useful for CAD agents, not only model evaluation.

A good final positioning:

> IntersectionQA tests whether code models understand the spatial semantics of CAD programs. Unlike text-to-CAD benchmarks that evaluate generated shape similarity, IntersectionQA isolates assembly-level geometric reasoning: interference, contact, clearance, overlap magnitude, and repair direction.

[1]: https://arxiv.org/abs/2602.16317?utm_source=chatgpt.com "CADEvolve: Creating Realistic CAD via Program Evolution"
[2]: https://huggingface.co/datasets/kulibinai/cadevolve?utm_source=chatgpt.com "kulibinai/cadevolve · Datasets at Hugging Face"
