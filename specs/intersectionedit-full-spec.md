# IntersectionEdit: dataset and training spec

Your second pipeline is a natural continuation of IntersectionQA.

IntersectionQA asks:

> Given CAD objects and transforms, what is the geometric relation?

IntersectionEdit asks:

> Given CAD objects, transforms, and a target spatial relation, what edit should be applied to achieve it?

This is more useful for CAD agents because it moves from diagnosis to correction.

A good framing:

> IntersectionEdit trains models to predict minimal geometric edits that transform an assembly into a target spatial state, such as resolving interference, achieving a desired clearance, or placing centroids at a specified distance while preserving non-intersection.

---

# 1. Core task definition

Each example contains:

1. Two or more CadQuery objects.
2. Current assembly transforms.
3. Current geometric state:

   * intersecting
   * touching
   * disjoint
   * clearance distance
   * centroid distance
   * intersection volume
4. Target geometric constraint.
5. Required edit format.
6. Verifier-computed ground-truth edit or reward.

The model outputs an edit, not just a label.

Example:

```text
Given object_a and object_b, object_b currently intersects object_a.

Move object_b by the smallest translation vector such that the two objects no longer have positive-volume interference and have at least 1.0 mm clearance.

Return:
dx, dy, dz
```

Expected answer:

```text
0.0, 0.0, 3.4
```

---

# 2. Rename options

`IntersectionEdit` is good. Alternatives:

| Name               | Meaning                                           |
| ------------------ | ------------------------------------------------- |
| IntersectionEdit   | Clear and directly related to IntersectionQA      |
| AssemblyEdit       | Broader; can include clearance, alignment, mating |
| ClearanceEdit      | More focused on distance/clearance                |
| CADResolve         | Emphasizes resolving assembly defects             |
| SpatialEditCAD     | More descriptive, less elegant                    |
| InterferenceRepair | Strong for collision-resolution tasks             |
| ContactEdit        | Good if you include target touching/contact tasks |

I would use:

```text
IntersectionEdit: Learning to Repair and Control CAD Assembly Interference
```

or:

```text
IntersectionEdit: Verifiable Spatial Editing for CAD Assemblies
```

---

# 3. Main task families

## Task 1: move closer to target clearance

Initial state: objects are disjoint.

Goal: move one object closer until face-to-face clearance equals a target distance.

Example:

```text
Object A and object B are non-intersecting.

Move object B so that the minimum distance between the two objects is exactly 2.5 mm.

Only translate object B.
Return dx, dy, dz in millimetres.
```

Answer:

```text
-4.2, 0.0, 0.0
```

This trains controlled spatial editing.

Important distinction:

* `minimum surface distance = 2.5 mm`
* not centroid distance
* not bounding-box distance

These must be separate task types.

---

## Task 2: move farther to target clearance

Initial state: objects are already disjoint but too close.

Goal: increase clearance.

```text
The current minimum distance between the objects is 0.6 mm.

Move object B so that the minimum distance is at least 5.0 mm.

Use the smallest possible translation.
Return dx, dy, dz.
```

This is useful for tolerance enforcement.

---

## Task 3: resolve intersection with minimal translation

Initial state: objects intersect.

Goal: remove positive-volume overlap with minimal movement.

```text
Object A and object B have positive-volume interference.

Move object B by the smallest translation vector such that the objects no longer intersect.

Return dx, dy, dz.
```

This is the core repair task.

However, exact “smallest possible movement” can be hard for arbitrary shapes. You may need to define the ground truth carefully.

Better for v1:

```text
Move object B along one of the six cardinal directions: +x, -x, +y, -y, +z, -z.
Choose the smallest movement that removes interference.
```

Answer:

```text
+x, 3.2
```

This is much more labelable and less ambiguous.

Then v2 can support arbitrary 3D vectors.

---

## Task 4: resolve intersection with target clearance

Initial state: objects intersect.

Goal: move them apart until they clear by some margin.

```text
Move object B by the smallest axis-aligned translation such that object A and object B have at least 2.0 mm clearance.
```

Answer:

```text
-z, 6.8
```

This is more practical than merely resolving collision.

---

## Task 5: centroid-distance editing

Initial state: any.

Goal: set centroid-to-centroid distance to target, while avoiding intersection.

```text
Move object B so that the centroid distance between object A and object B is exactly 50.0 mm, while ensuring the objects do not intersect.

Only translate object B along the current centroid-to-centroid direction.
Return the signed movement distance.
```

Answer:

```text
+12.4
```

This is easier than arbitrary translation because the movement direction is fixed.

Important: centroid-distance tasks are not the same as clearance tasks. A model may solve centroid distance while still causing intersection, so the verifier must check both.

---

## Task 6: target contact

Initial state: disjoint.

Goal: move until touching but not intersecting.

```text
Move object B along the given direction until it just touches object A without positive-volume overlap.

Direction: -x

Return the movement distance in millimetres.
```

Answer:

```text
7.3
```

This is very useful for mating/alignment behavior.

---

## Task 7: choose edit from candidates

Instead of asking for a continuous coordinate output, give candidates.

```text
Which edit achieves at least 1.0 mm clearance with the smallest movement?

A. translate object_b by (2.0, 0.0, 0.0)
B. translate object_b by (3.0, 0.0, 0.0)
C. translate object_b by (0.0, 2.0, 0.0)
D. translate object_b by (0.0, 0.0, 2.0)
```

This is excellent for RL and evaluation because the answer is discrete.

---

## Task 8: repair ranking

```text
Rank the candidate edits from best to worst.

Best means:
1. no interference
2. satisfies clearance target
3. smallest movement magnitude
```

Answer:

```text
BCDA
```

This gives a richer reward signal than a single coordinate prediction.

---

## Task 9: multi-object repair

```text
Object B intersects both object A and object C.

Move object B by the smallest translation that removes all interferences and gives at least 1.0 mm clearance from every other object.
```

This is closer to real assembly repair but should be v2, not v1.

---

# 4. Output formats

Avoid only free-form coordinates at first. Use several output modes with increasing difficulty.

## Format A: discrete direction

```text
+x
-x
+y
-y
+z
-z
none
```

Use when asking for the best separating direction.

---

## Format B: axis plus distance

```text
axis=x, direction=+, distance_mm=3.2
```

This is probably the best v1 format.

It is precise but still constrained.

---

## Format C: full translation vector

```text
dx=0.0, dy=-2.5, dz=1.0
```

Harder, more useful, more brittle.

---

## Format D: candidate selection

```text
B
```

Best for clean evaluation and RL.

---

## Format E: edit program

```python
object_b = object_b.translate((0.0, -2.5, 1.0))
```

Useful if you want the model to output code edits directly.

---

# 5. Recommended v1 task set

For a first strong paper/pipeline, I would implement these:

## P0 tasks

1. Best separating direction
2. Axis-aligned minimal movement to remove intersection
3. Axis-aligned movement to achieve target clearance
4. Move closer/farther along specified axis to target clearance
5. Candidate edit selection
6. Candidate edit ranking

Do not start with arbitrary 3D vector minimal translation for arbitrary shapes. It is hard to define uniquely and may create label ambiguity.

---

# 6. Important ambiguity: “smallest possible movement”

For two arbitrary CAD solids, the globally smallest translation vector that separates them can be non-unique and hard to compute exactly.

Example:

* Moving +x by 1.0 mm works.
* Moving +y by 1.0 mm also works.
* Moving diagonally by 0.71 mm may also work.
* Moving along the contact normal may be best, but normals may be undefined or multiple.

So define one of these instead.

## Safer v1 definition

```text
Smallest axis-aligned translation among the six cardinal directions.
```

Compute:

```text
+x minimum separating distance
-x minimum separating distance
+y minimum separating distance
-y minimum separating distance
+z minimum separating distance
-z minimum separating distance
```

Pick the smallest valid one.

Output:

```text
+y, 2.4
```

This is labelable and deterministic.

## More advanced v2 definition

```text
Smallest translation along a specified direction vector.
```

Example:

```text
Move object B along direction (1, 1, 0), normalized, until clearance is 2.0 mm.
```

Output:

```text
distance_mm=4.7
```

This is also deterministic.

## Hard v3 definition

```text
Globally minimal translation vector in R^3.
```

This is useful but harder to compute robustly.

I would avoid making this your first target.

---

# 7. Ground-truth generation

For each object pair, you can compute edit labels using verifier search.

## Axis-aligned separating distance

For a given direction `d`, find the smallest scalar `t >= 0` such that:

```text
relation( object_a, translate(object_b, t * d) ) != intersecting
```

For clearance target `c`:

```text
minimum_distance(object_a, translate(object_b, t * d)) >= c
```

Use binary search.

Pseudo-code:

```python
def min_translation_along_direction(a, b, direction, target_clearance=0.0):
    lo = 0.0
    hi = initial_hi

    while not satisfies(a, translate(b, hi * direction), target_clearance):
        hi *= 2.0

    for _ in range(num_steps):
        mid = (lo + hi) / 2.0
        if satisfies(a, translate(b, mid * direction), target_clearance):
            hi = mid
        else:
            lo = mid

    return hi
```

Then evaluate all six cardinal directions:

```python
directions = {
    "+x": (1, 0, 0),
    "-x": (-1, 0, 0),
    "+y": (0, 1, 0),
    "-y": (0, -1, 0),
    "+z": (0, 0, 1),
    "-z": (0, 0, -1),
}
```

Pick the best:

```python
best_direction = argmin(valid_direction_distance)
```

---

# 8. Decimal precision

You mentioned “exact range down to a decimal.”

Define tolerance explicitly.

For training/eval, do not require exact string match on `3.2` unless labels are rounded.

Use one of these:

## Rounded-label evaluation

Ground truth:

```text
distance_mm=3.2
```

Accept only exact rounded value.

Simple, but brittle.

## Numeric tolerance evaluation

Ground truth:

```text
3.23
```

Accept if:

```text
abs(pred - truth) <= 0.1 mm
```

Better.

## Relative tolerance

For larger distances:

```text
abs(pred - truth) <= max(0.1 mm, 0.02 * truth)
```

Good general rule.

Recommended:

```text
decimal_precision = 0.1 mm
acceptance_tolerance = 0.15 mm
```

This is fair if the model outputs one decimal place.

---

# 9. Prompt templates

## Template 1: axis-aligned collision repair

````text
You are given two CadQuery objects and their assembly transforms.

Object "{movable_object}" may be translated. Object "{fixed_object}" must remain fixed.

Goal:
Move "{movable_object}" by the smallest axis-aligned translation that removes positive-volume interference.

Allowed directions:
+x, -x, +y, -y, +z, -z

Definitions:
- Interference means positive-volume overlap.
- Touching is allowed.
- Use millimetres.
- Do not execute the code.
- You may reason before answering.

Script:
```python
{script}
````

Final answer format:
direction=<one of +x,-x,+y,-y,+z,-z>, distance_mm=<one decimal>

````

Example answer:

```text
direction=+x, distance_mm=3.2
````

---

## Template 2: repair with clearance

````text
You are given two CadQuery objects and their assembly transforms.

Object "{movable_object}" may be translated. Object "{fixed_object}" must remain fixed.

Goal:
Move "{movable_object}" by the smallest axis-aligned translation such that the two objects have at least {target_clearance_mm} mm clearance.

Allowed directions:
+x, -x, +y, -y, +z, -z

Definitions:
- Clearance is the minimum surface-to-surface distance between the two objects.
- If the objects intersect, clearance is negative/invalid until interference is resolved.
- Use millimetres.
- Do not execute the code.

Script:
```python
{script}
````

Final answer format:
direction=<direction>, distance_mm=<one decimal>

````

---

## Template 3: move closer to target clearance

```text
The two objects are currently non-intersecting.

Object "{movable_object}" may be translated along the specified direction:
{direction}

Goal:
Move "{movable_object}" so that the minimum surface-to-surface clearance between the objects is {target_clearance_mm} mm.

Return the signed movement distance in millimetres.

Script:
```python
{script}
````

Final answer format:
distance_mm=<signed number with one decimal>

````

Example:

```text
distance_mm=-4.7
````

---

## Template 4: centroid-distance target

````text
The centroid of "{fixed_object}" is fixed.

Move "{movable_object}" along the current centroid-to-centroid direction so that the final centroid distance is exactly {target_centroid_distance_mm} mm.

The final placement must not create positive-volume interference.

Script:
```python
{script}
````

Final answer format:
distance_mm=<signed number with one decimal>

````

---

## Template 5: candidate edit selection

```text
You are given two CadQuery objects and their assembly transforms.

Goal:
Choose the edit that satisfies the target condition with the smallest movement.

Target condition:
The objects must not intersect and must have at least {target_clearance_mm} mm clearance.

Candidate edits:
A. translate "{movable_object}" by ({dx_a}, {dy_a}, {dz_a})
B. translate "{movable_object}" by ({dx_b}, {dy_b}, {dz_b})
C. translate "{movable_object}" by ({dx_c}, {dy_c}, {dz_c})
D. translate "{movable_object}" by ({dx_d}, {dy_d}, {dz_d})

Script:
```python
{script}
````

Final answer format:
Return exactly one letter: A, B, C, or D.

````

---

## Template 6: candidate edit ranking

```text
Rank the candidate edits from best to worst.

A better edit:
1. satisfies the target clearance,
2. avoids positive-volume interference,
3. uses a smaller translation magnitude.

Target clearance: {target_clearance_mm} mm

Candidate edits:
A. translate object_b by ({dx_a}, {dy_a}, {dz_a})
B. translate object_b by ({dx_b}, {dy_b}, {dz_b})
C. translate object_b by ({dx_c}, {dy_c}, {dz_c})
D. translate object_b by ({dx_d}, {dy_d}, {dz_d})

Final answer format:
Return exactly four capital letters, such as DBCA.
````

---

# 10. Dataset schema

```json
{
  "id": "intersectionedit_000001",
  "source": "intersectionqa_v1",
  "task_type": "axis_aligned_repair",
  "script": "...",
  "fixed_object": "object_a",
  "movable_object": "object_b",

  "initial_state": {
    "relation": "intersecting",
    "intersection_volume": 42.7,
    "normalized_intersection": 0.08,
    "minimum_distance": 0.0,
    "centroid_distance": 18.4
  },

  "target": {
    "type": "clearance",
    "target_clearance_mm": 1.0,
    "allow_touching": false
  },

  "allowed_edit": {
    "edit_type": "translation",
    "directions": ["+x", "-x", "+y", "-y", "+z", "-z"],
    "rotation_allowed": false,
    "movable_only": "object_b"
  },

  "answer": {
    "direction": "+x",
    "distance_mm": 3.2,
    "translation": [3.2, 0.0, 0.0]
  },

  "verification": {
    "final_relation": "disjoint",
    "final_clearance_mm": 1.0,
    "final_intersection_volume": 0.0,
    "movement_magnitude": 3.2,
    "satisfies_target": true
  },

  "diagnostics": {
    "best_direction_margin": 0.8,
    "num_valid_directions": 3,
    "ambiguous": false,
    "difficulty": "axis_aligned_intersection_repair"
  },

  "split_group": "cfg_00042"
}
```

---

# 11. Difficulty categories

## Level 1: simple axis-aligned repair

* Boxes/cylinders
* No rotation
* One obvious separating axis

## Level 2: rotated primitive repair

* Rotated boxes/cylinders
* Axis-aligned edit still required
* More difficult because local object axes differ from world axes

## Level 3: clearance targeting

* Already disjoint or barely intersecting
* Need exact movement to achieve target clearance

## Level 4: cavities and concavity

* Moving one direction may appear wrong under AABB but correct under exact geometry
* Holes, brackets, rings, U-shapes

## Level 5: ambiguous repair

* Multiple directions work
* Need choose smallest movement
* Or mark as ambiguous and exclude from direct-answer training

## Level 6: multi-object repair

* Move one object to satisfy constraints against several fixed objects

---

# 12. Very important: ambiguity filtering

For SFT, avoid examples where multiple answers are nearly equally correct.

Example:

```text
+x requires 2.00 mm
+y requires 2.02 mm
```

The “correct” answer is technically +x, but a model choosing +y is not meaningfully wrong.

Add ambiguity metadata:

```python
best = distances[0]
second_best = distances[1]
margin = second_best - best

ambiguous = margin < 0.2  # mm
```

For strict SFT/evaluation:

```text
exclude ambiguous examples
```

For RL:

```text
keep ambiguous examples and reward based on verifier outcome/movement quality
```

This is important. RL can handle soft rewards better than SFT.

---

# 13. RL design

IntersectionEdit is even more RL-suitable than IntersectionQA.

Why:

* Outputs can be verified by executing the edit.
* Rewards can be continuous.
* Partial credit is meaningful.
* The model can improve beyond imitation labels.

## Reward for repair task

```python
def repair_reward(predicted_translation, assembly, target_clearance):
    final = apply_translation(assembly, predicted_translation)

    intersects = has_interference(final)
    clearance = min_distance(final)
    movement = norm(predicted_translation)

    if intersects:
        return -1.0

    if clearance < target_clearance:
        return 0.2 * (clearance / target_clearance)

    return 1.0 - 0.05 * movement
```

Better normalized version:

```python
reward = success_reward - movement_penalty - overshoot_penalty
```

Where:

```python
success_reward = 1.0 if satisfies_target else 0.0
movement_penalty = alpha * (movement / reference_distance)
overshoot_penalty = beta * abs(final_clearance - target_clearance)
```

---

## Reward for target clearance

```python
def clearance_reward(final_clearance, target_clearance, movement):
    clearance_error = abs(final_clearance - target_clearance)

    if final_clearance < 0:
        return -1.0

    return (
        1.0
        - min(clearance_error / tolerance_scale, 1.0)
        - 0.05 * movement
    )
```

---

## Reward for candidate selection

```python
reward = 1.0 if selected == best_candidate else 0.0
```

Or soft:

```python
reward = candidate_score[selected]
```

Where candidate scores are computed from verifier results.

---

## Reward for ranking

```python
reward = pairwise_ranking_accuracy(predicted_order, true_order)
```

This is useful for GRPO because completions can be compared cleanly.

---

# 14. SFT vs RL for IntersectionEdit

## SFT is useful for:

* teaching output format
* teaching basic movement logic
* learning common geometric patterns
* bootstrapping a 4B model

## RL is useful for:

* continuous numeric improvement
* reducing overshoot
* improving exact target clearance
* learning from multiple valid edits
* optimizing actual verifier success instead of imitating one label

Recommended pipeline:

```text
1. SFT on constrained edit tasks
2. SFT on counterfactual edit groups
3. GRPO on candidate-selection and ranking tasks
4. GRPO on coordinate-output tasks with verifier reward
5. Evaluate on held-out topology, held-out generator, and hard near-boundary repairs
```

---

# 15. Counterfactual edit groups

For IntersectionEdit, counterfactual groups are even more valuable.

Example group:

```text
same objects
same orientation
different initial x offset
same target clearance
```

Rows:

```text
initial clearance = 10.0 mm -> move closer by 7.5 mm
initial clearance = 5.0 mm -> move closer by 2.5 mm
initial clearance = 2.5 mm -> move by 0.0 mm
initial overlap = 1.0 mm -> move away by 3.5 mm
```

This teaches linear sensitivity.

Metadata:

```json
{
  "counterfactual_group_id": "edit_cfg_00031",
  "changed_parameter": "initial_translation_x",
  "target_clearance_mm": 2.5,
  "expected_distance_mm": 7.5
}
```

Useful counterfactual dimensions:

* initial translation
* target clearance
* target centroid distance
* object size
* rotation angle
* movable object choice
* allowed direction
* fixed/movable object swap

---

# 16. Candidate generation for RL

For each repair example, generate candidate edits:

1. exact best edit
2. slightly under-corrected edit
3. slightly over-corrected edit
4. wrong axis edit
5. correct axis but wrong sign
6. large but valid edit

Example:

```text
A. +x 2.0 mm  # still intersects
B. +x 3.2 mm  # exact/best
C. +x 6.0 mm  # valid but excessive
D. -x 3.2 mm  # wrong sign
E. +y 3.5 mm  # valid but not minimal
```

This is excellent for reward learning and diagnostic evaluation.

---

# 17. Evaluation metrics

## Direction task

```text
direction accuracy
axis accuracy
sign accuracy
```

## Axis + distance task

```text
direction accuracy
mean absolute distance error
within 0.1 mm accuracy
within 0.5 mm accuracy
target success rate after applying predicted edit
```

## Full vector task

```text
translation vector L2 error
target success rate
movement optimality ratio
overshoot error
final clearance error
```

## Repair task

```text
interference resolved rate
target clearance success rate
minimality ratio
```

Where:

```text
minimality_ratio = predicted_movement / optimal_movement
```

Only compute minimality ratio for successful predictions.

## Candidate selection

```text
exact accuracy
success-weighted accuracy
```

## Ranking

```text
exact order accuracy
pairwise ranking accuracy
Kendall tau
```

---

# 18. Best benchmark metrics

For the paper, I would focus on these:

```text
success_rate
mean_final_clearance_error
mean_movement_optimality_ratio
within_0.1mm_accuracy
near_boundary_success_rate
heldout_topology_success_rate
```

This tells a much stronger story than simple answer accuracy.

---

# 19. Splits

Reuse your IntersectionQA splitting logic, but add edit-specific groups.

Keep together:

```text
same CAD source generator
same object pair
same counterfactual edit group
same target-clearance family
same candidate set family
```

Suggested splits:

| Split                    | Purpose                    |
| ------------------------ | -------------------------- |
| random                   | sanity check               |
| object-pair-held-out     | unseen object combinations |
| generator-held-out       | unseen CAD families        |
| topology-held-out        | unseen shape classes       |
| target-distance-held-out | unseen clearance values    |
| near-boundary-hard       | tiny overlaps/clearances   |
| multi-object-hard        | multi-constraint repair    |

Target-distance-held-out is new and valuable.

Example:

```text
Train target clearances: 1, 2, 5, 10 mm
Test target clearances: 1.5, 3, 7.5 mm
```

This tests interpolation/extrapolation.

---

# 20. Recommended MVP

Given that your 4B fine-tune already reached 93%, I would make IntersectionEdit more ambitious but still controlled.

## MVP v0.1

Task types:

1. axis-aligned minimal repair
2. axis-aligned target clearance
3. move along specified direction to target clearance
4. candidate edit selection
5. candidate edit ranking

Objects:

* primitives
* rotated primitives
* simple compound solids
* cavity/concavity subset

Outputs:

* direction + one-decimal distance
* candidate letter
* ranking letters

Training:

* SFT warmup
* GRPO on candidate selection/ranking
* GRPO on direction+distance with verifier reward

Metrics:

* success rate
* within 0.1 mm
* movement optimality ratio
* final clearance error
* near-boundary hard performance

---

# 21. Stronger paper contribution

IntersectionQA showed that models can learn to classify CAD spatial relations.

IntersectionEdit can claim:

> We show that small code models can be trained not only to detect CAD assembly interference, but to predict verified geometric edits that resolve interference and satisfy clearance constraints.

That is much closer to a CAD agent.

Suggested abstract-level claim:

> IntersectionEdit converts CAD spatial reasoning into a verifiable editing problem: models must output translations that transform an assembly into a target relation. Because every edit can be checked by an exact geometry kernel, the benchmark supports both supervised learning and reinforcement learning with dense geometric rewards.

---

# 22. Potential failure modes to study

This could become a strong analysis section.

Models may fail by:

* choosing correct direction but wrong distance
* moving away too far
* resolving intersection but failing target clearance
* satisfying centroid distance but causing intersection
* confusing surface clearance with centroid distance
* using bounding-box logic on cavity objects
* ignoring rotation
* ignoring which object is movable
* choosing valid but non-minimal movement
* failing when multiple fixed objects constrain the movable object

These are exactly the failures CAD agents need to solve.

---

# 23. What I would add beyond your current idea

The highest-value additions are:

## 1. Candidate edit ranking

Do not only ask for coordinates. Ranking gives richer RL signal and easier evaluation.

## 2. Axis-constrained repair first

Avoid arbitrary 3D minimal vectors in v1. Use six cardinal directions or specified movement rays.

## 3. Target clearance, not just non-intersection

Engineering cares about clearance, not merely non-collision.

## 4. Movement optimality

Score not only whether the edit works, but whether it is minimal.

## 5. Verifier reward

Make RL central:

```text
answer is good if applying it satisfies the target constraint
```

This is better than imitation.

## 6. Ambiguity filtering

For supervised labels, remove cases where multiple edits are basically equally good.

For RL, keep them and reward all valid edits proportionally.

---

# 24. Concise final spec

```markdown
# IntersectionEdit

IntersectionEdit is a CAD spatial-editing dataset derived from executable CadQuery assemblies. Given object definitions, assembly transforms, and a target geometric condition, the model must output an edit that changes the assembly to satisfy the target.

Target conditions include:
- remove positive-volume interference
- achieve a required minimum clearance
- move objects closer to a specified clearance
- move objects farther to a specified clearance
- set centroid distance while avoiding interference
- choose or rank candidate edits by validity and minimality

The primary edit type is translation of one movable object while other objects remain fixed. Early tasks restrict translations to six world-axis directions or a specified direction vector, making labels deterministic and verifiable. Later tasks allow full 3D translation vectors.

Ground truth is computed by exact CAD-kernel verification and directional binary search. Each predicted edit is evaluated by applying it to the assembly and measuring final interference, clearance, centroid distance, and movement magnitude.

IntersectionEdit supports both SFT and RL. SFT teaches edit formats and common geometric patterns, while RL optimizes verified outcomes such as target success, clearance error, and movement minimality. Candidate-selection and ranking tasks provide especially clean rewards for GRPO-style training.
```

This is the right next step if your goal is “models that can predict and resolve intersection issues,” not just classify them.
