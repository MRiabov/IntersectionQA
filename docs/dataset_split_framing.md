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
- `test_near_boundary` is a small difficulty-targeted challenge suite. It uses
  the same group-safe assignment policy as the other held-out splits and should
  stay roughly 5-10% of the release, not half of it.

The April 25, 2026 split redistribution changed near-boundary/counterfactual
assignment from "all hard examples go to `test_near_boundary`" to a
deterministic group-level split policy: most boundary groups go to `train` and
`validation`, with a small held-out `test_near_boundary` challenge slice.

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
global number. The release pipeline now balances the main relation target in
splits where all target classes exist and caps pairwise answers, but several
derived bucket tasks still have real geometry scarcity. Exact accuracy can
therefore look better than macro precision or macro F1.
