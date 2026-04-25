# Dataset Split Framing

IntersectionQA v0.1 should be read as a conventional supervised split plus
named diagnostic suites, not as six interchangeable train/validation/test
partitions.

## Recommended Interpretation

| Dataset split | Short role | Use for |
| --- | --- | --- |
| `train` | training | Model fitting and SFT/RL training. |
| `validation` | dev validation | Model selection, prompt/training debugging, and early stopping. |
| `test_random` | primary random test | Main in-distribution held-out score for routine reports. |
| `test_object_pair_heldout` | object-pair challenge test | Generalization to unseen base object-pair and assembly groups. |
| `test_near_boundary` | boundary/counterfactual challenge suite | Stress-testing touching, near-miss, tiny-overlap, pairwise, and ranking behavior. |
| `test_generator_heldout` | generator-family challenge test | Reserved for generator-family OOD evaluation; may be empty in current exports. |

For a simple train/validation/test setup, use:

- train: `train`
- validation: `validation`
- test: `test_random`

Report `test_object_pair_heldout` and `test_near_boundary` separately. They
answer different questions and should not be averaged into a single headline
score unless the report explicitly says it is a composite benchmark score.

## Leakage Semantics

The intended clean development path is:

- `validation` has no row ID, base object-pair, assembly, or counterfactual
  group overlap with `train`.
- `test_random` has no row ID, base object-pair, assembly, or counterfactual
  group overlap with `train`.
- `test_object_pair_heldout` is the strict object-pair/assembly holdout suite.
- `test_near_boundary` is a difficulty-targeted challenge suite. It forbids
  counterfactual group leakage, but it is not currently a strict base
  object-pair/assembly holdout. Interpret it as boundary/counterfactual stress
  testing, not as the clean primary test split.

The April 25, 2026 90K augmentation moved a leakage-safe subset of
counterfactual groups from `test_near_boundary` into `train` so pairwise and
ranking tasks are represented during training. The remaining near-boundary
counterfactual groups stay held out by `counterfactual_group_id`.

## Reporting Template

Use this shape for experiment summaries:

```text
Validation accuracy: ...
Primary test_random accuracy: ...
Object-pair holdout accuracy: ...
Near-boundary/counterfactual challenge accuracy: ...

Breakdown by task:
- binary_interference: ...
- relation_classification: ...
- volume_bucket: ...
- clearance_bucket: ...
- tolerance_fit: ...
- pairwise_interference: ...
- ranking_normalized_intersection: ...
```

If precision/recall/F1 are reported, prefer per-task metrics over a single
global number. Several tasks have strong label imbalance, so exact accuracy can
look better than macro precision or macro F1.
