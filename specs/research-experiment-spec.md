# Research Experiment Specification

This document specifies the experiment suite for the IntersectionQA and
IntersectionEdit paper. It is paper-facing and researcher-facing: the goal is
not only to improve scores on this benchmark, but to identify which data,
training, prompting, reward, and evaluation choices improve LLM performance on
CAD spatial reasoning and constrained CAD repair more broadly.

The canonical task, label, schema, and generation semantics remain in
`benchmark-task-spec.md`, `intersectionedit-task-spec.md`, `label_rules.md`,
`schema.md`, and `generation_policy.md`. If this document conflicts with those
specifications, the canonical specifications win.

## Goals

The experiment suite should answer four questions:

1. What CAD spatial reasoning failures do current LLMs have?
2. Which dataset properties make those failures learnable?
3. Which training recipes transfer across diagnostic QA and edit/repair tasks?
4. What baselines, ablations, and artifacts should future researchers compare
   against?

The paper should avoid framing IntersectionQA as only a leaderboard. The
stronger claim is that exact CAD-derived spatial supervision can improve LLM
CAD reasoning, but the improvement depends on counterfactual data, task mixture,
split hygiene, and verifier-shaped edit supervision.

## Compute Budgets

The default target budget is approximately:

- 30 H100 hours, or
- 70-100 A100 80GB hours.

An expanded budget may be up to roughly 3x the default budget. The default
budget should be enough for one well-instrumented 4B training stack plus
diagnostic ablations. The expanded budget should be used for model-size scaling,
fuller transfer matrices, and RL variant comparisons.

Default step-count guidance for the Qwen3.5 4B stack:

- answer-only SFT canary: 20-50 optimizer steps on 128-512 train rows;
- answer-only full SFT: one epoch over public `train`; with the current 43,664
  row IntersectionQA train split and effective batch 32, this is approximately
  1,365 optimizer steps;
- reasoning-SFT canary: 20-50 optimizer steps;
- deterministic or rejection-sampled reasoning SFT: one epoch over the selected
  trace set, normally 500-1,500 optimizer steps depending trace count and
  packing;
- supervised data-scaling runs: one epoch each at 1k, 5k, 15k, 30k, and full
  train, roughly 3k total optimizer steps at effective batch 32 for the current
  train split;
- GRPO/GSPO canary: 50-100 optimizer steps;
- bounded main GRPO/GSPO from the best reasoning-SFT adapter: 1,500-2,000
  optimizer steps when the canary is healthy.

The historical H100 GRPO datapoint is about 9 seconds per optimizer step under
one candidate-feature configuration. Use that as an initial budget estimate,
but always record actual step time, eval overhead, and checkpoint overhead. At
9 seconds per step, 1,500-2,000 GRPO steps are roughly 3.75-5 raw GPU hours
before periodic quality evaluation and artifact upload.

All GPU experiments must record:

- GPU type and count;
- wall-clock time;
- approximate cost when rented;
- model, adapter, and tokenizer identifiers;
- exact command line;
- code commit or dirty-worktree note;
- dataset release directory and manifest hash when available;
- train/eval split names;
- random seeds;
- checkpoint selection rule;
- final and best-checkpoint metrics.

## Reporting Rules

Do not average all test splits into a single headline score unless the paper
also reports the constituent split scores. Report at least:

- `validation`;
- `test_random`;
- `test_object_pair_heldout`;
- `test_near_boundary`;
- `test_generator_heldout` when populated.

For each evaluated model or training run, report:

- exact answer accuracy;
- invalid output rate;
- task-type breakdown;
- answer-label breakdown where applicable;
- macro precision, recall, and F1 for classification tasks;
- bucket confusion matrices for bucket tasks;
- pairwise accuracy for comparison tasks;
- ranking accuracy or normalized rank metrics for ranking tasks;
- repair verifier success for edit tasks;
- direction accuracy, magnitude MAE, and within-tolerance rate for numeric edit
  tasks.

Training-time experiments must use only public `train` rows for optimizer
updates. Internal SFT/RL splits must be group-safe and must keep source groups,
object-pair groups, assembly groups, and counterfactual groups intact.

## Experiment Priority Levels

Use three priority levels:

- **P0 required**: needed for a credible paper and default budget.
- **P1 high-value**: run if default budget has room or if it replaces a weaker
  experiment.
- **P2 expanded-budget**: run only with the expanded budget or after P0/P1 are
  complete.

## P0: Dataset And Benchmark Characterization

Before model comparisons, characterize the benchmark itself.

Required reports:

- row counts by task, split, source, and task family;
- answer distribution by task and split;
- relation distribution: `disjoint`, `touching`, `near_miss`, `intersecting`,
  `contained`, and `invalid` if present;
- volume-bucket, clearance-bucket, tolerance-fit, pairwise, ranking, and edit
  answer distributions;
- difficulty-tag distribution, including near-boundary, tiny-overlap,
  transform-heavy, rotation-heavy, containment, AABB false-positive, and repair
  ambiguity tags when available;
- split leakage audit covering row IDs, source groups, generator families,
  object pairs, assembly groups, and counterfactual groups;
- exact-label availability and kernel status rates;
- stored edit verifier pass rates;
- public prompt length statistics by task.

Required outputs:

- machine-readable JSON reports;
- concise Markdown tables for the paper;
- a short description of which split should be treated as primary random test
  and which splits are challenge suites.

## P0: Heuristic And Tool Baselines

Run non-LLM baselines to calibrate the benchmark.

Required baselines:

- majority-class baseline per task;
- balanced random baseline per task;
- AABB-overlap baseline for binary/relation tasks;
- OBB or bbox-derived baseline if implemented;
- exact CAD-kernel oracle or tool-assisted upper bound where evaluation permits;
- edit-policy oracle for stored IntersectionEdit repair rows.

Tool-assisted results must be separated into three scientific categories:

- **Exact CAD-kernel oracle / scripted verifier**: the true upper bound. It may
  execute trusted dataset geometry or use stored exact labels where permitted,
  and it must report tool failure rate and failure reasons.
- **Standard instruct model plus tools**: the practical CAD-agent baseline. Use
  an off-the-shelf instruct/code model with a verifier or geometry tool and no
  IntersectionQA/IntersectionEdit fine-tuning. This answers how far current
  general tool-using models get.
- **Fine-tuned tool-using model**: a training experiment, not the baseline. This
  answers whether SFT/GRPO improves verifier-using CAD behavior.

Recommended diagnostic baselines:

- AABB false-positive subset score;
- AABB false-negative subset score if applicable;
- near-boundary-only heuristic score;
- relation-specific heuristic confusion matrix.

The paper should explicitly separate official closed-book model evaluation from
tool-assisted or oracle evaluation.

## P0: Closed-Book Zero-Shot Model Suite

Evaluate a fixed set of current LLMs with public code-only prompts and strict
parsing. Models must not execute CAD code or call geometry tools.

Minimum model suite:

- one small open model around 3-4B parameters;
- one medium open model around 7-14B parameters;
- one larger or stronger open model when practical;
- one frontier proprietary model;
- one CAD-specialized or code-specialized model if available;
- the main trained Qwen3.5 4B variant from this work.

Required prompt condition:

- canonical closed-book public prompt;
- deterministic or low-temperature decoding;
- guided or constrained final-answer decoding when the backend supports it;
- fixed max-output-token setting per task family;
- strict parser with invalid outputs counted as wrong.

The main closed-book table may use schema-constrained final-answer decoding, but
it must be labeled as such. Unconstrained decoding is not required for every
model and split. Instead, run it as a bounded diagnostic on a fixed validation
subset, for example 256-1k rows, to estimate how much schema guidance is
masking formatting weakness. Only run full unconstrained held-out evaluation
when the bounded diagnostic shows a large discrepancy or when compute is
available.

Required analysis:

- split-by-split score table;
- task-by-task score table;
- invalid output rate table;
- confusion matrices for relation and bucket tasks;
- near-boundary and counterfactual subset metrics.

## P0: Prompting Ablation

Run prompt ablations on a fixed validation subset before choosing the final
evaluation prompt.

Prompt conditions:

- direct answer only;
- answer tags such as `<answer>...</answer>`;
- concise reasoning plus final answer;
- definitions-only prompt;
- few-shot prompt;
- public closed-book prompt with explicit "do not execute code" instruction.

Optional diagnostic-only condition:

- trusted geometry-feature prompt, clearly labeled as non-public and not a
  closed-book benchmark setting.

Report accuracy, invalid output rate, average completion length, and task-wise
regressions. Choose one canonical public prompt family for the main benchmark
table and keep it fixed across models.

## P0: Answer-Only SFT Baseline

Train the simplest reproducible supervised baseline.

Default model:

- `unsloth/Qwen3.5-4B`, unless a later runbook supersedes it.

Default method:

- LoRA or QLoRA according to the current training runbook;
- assistant-answer-only loss or equivalent packed target masking;
- public `train` only;
- checkpoint selection on `validation`;
- exact-answer generation evaluation on all held-out splits.

Report:

- train rows used;
- task mixture;
- effective batch size;
- context length;
- trainable parameter count;
- best validation checkpoint;
- held-out split metrics;
- comparison against zero-shot base model.

This is the main cheap baseline that other researchers should be able to
reproduce.

## P0: Reasoning-SFT Baseline

Train a supervised model on generated reasoning traces derived from exact
labels, metadata, and verifier logic. Do not use unlabeled model-generated
reasoning as ground truth unless that condition is explicitly labeled.

Compare:

- answer-only SFT;
- concise reasoning SFT;
- reasoning SFT with strict final answer tag;
- reasoning SFT evaluated with reasoning enabled;
- reasoning-SFT adapter evaluated with short-answer decoding if compatible.

Trace requirements:

- traces must be deterministic or reproducibly generated;
- traces must not reveal test labels during training;
- traces must end in the canonical answer format;
- traces should expose real geometry reasoning, not filler.

If a main GRPO/GSPO run is planned for 1,500-2,000 optimizer steps, the
preferred supervised initializer is not a tiny reasoning-SFT canary. Build a
medium rejection-sampled reasoning trace set first:

- generate candidate completions from a base or answer-SFT model;
- accept only parse-valid completions with the correct canonical answer;
- reject traces with obvious repetition, filler, or reasoning-answer mismatch;
- keep reasoning length bounded, initially around 128-256 tokens for SFT when
  the trace contains real geometry reasoning;
- preserve the acceptance rate and rejection reasons as run artifacts;
- train one SFT epoch over the accepted traces before GRPO.

Report whether reasoning traces improve:

- object-pair holdout generalization;
- near-boundary/counterfactual sensitivity;
- edit/repair tasks;
- invalid output rate.

## P0: Data Scaling Experiment

Train the same supervised recipe on increasing fractions of public `train`.

Default row counts:

- 1k rows;
- 5k rows;
- 15k rows;
- 30k rows;
- full train set.

Keep model, optimizer, prompt format, target format, and evaluation fixed as
much as possible. Sampling should be task- and answer-stratified. If a subset
cannot preserve rare labels, record the missing labels explicitly.

Report scaling curves by:

- task type;
- random test;
- object-pair heldout;
- near-boundary/counterfactual;
- edit/repair task family;
- rare labels.

This experiment should answer whether more CAD geometry data continues to help
or whether the model saturates early.

## P0: Counterfactual Sensitivity

Evaluate examples that share base CAD objects and assembly structure but differ
by controlled parameters.

Metrics:

- pair consistency;
- flip accuracy for binary and relation tasks;
- monotonicity for clearance and overlap buckets;
- ranking consistency;
- near-boundary accuracy;
- exact same-prompt-family invalid rate.

Training ablation:

- train without counterfactual groups;
- train with counterfactual groups;
- train with pairwise/ranking rows derived from counterfactual groups when
  available.

The paper should highlight whether models respond to small geometric parameter
changes or collapse to object-shape priors.

## P0: Generalization By Split

Every main model must be evaluated separately on:

- random held-out rows;
- object-pair/assembly heldout rows;
- generator-family heldout rows when populated;
- near-boundary/counterfactual challenge rows.

If generator-heldout is empty in the current release, either:

- build a release candidate with a populated generator-heldout split, or
- explicitly state that generator-family OOD evaluation is deferred.

Do not train on validation or test rows. Do not move challenge rows into train
for a final reported model unless that model is explicitly labeled as trained
under a different release/split policy.

## P0: Error Taxonomy

Produce an error analysis for the main zero-shot model, answer-only SFT model,
reasoning-SFT model, and best edit/RL model if present.

Required categories:

- parse or format failure;
- touching versus intersecting confusion;
- near-miss versus disjoint confusion;
- AABB shortcut error;
- containment missed;
- rotation or transform error;
- wrong repair axis;
- right repair direction but wrong magnitude;
- bucket off-by-one;
- scale or units error;
- common-label collapse;
- reasoning-answer mismatch if reasoning traces are generated.

Report representative examples, but keep public paper examples short and avoid
including excessive source code.

## P1: Task Transfer Matrix

Train on task subsets and evaluate on all task families.

Minimum transfer rows:

- binary only;
- relation only;
- bucket tasks only;
- QA tasks only;
- edit/repair tasks only;
- QA plus edit mixed;
- full mixture.

Columns should include:

- `binary_interference`;
- `relation_classification`;
- `volume_bucket`;
- `clearance_bucket`;
- `tolerance_fit`;
- `pairwise_interference`;
- `ranking_normalized_intersection`;
- `repair_direction`;
- `repair_translation`;
- `target_clearance_move`;
- `target_contact_move`;
- `centroid_distance_move`.

The purpose is to identify which supervision teaches reusable CAD spatial
reasoning and which tasks need direct supervision.

## P1: IntersectionEdit Evaluation

Evaluate constrained, verifier-checkable edit tasks as a sibling task family,
not only as extra QA labels.

Required task families when available:

- `repair_direction`;
- `repair_translation`;
- `target_clearance_move`;
- `target_contact_move`;
- `centroid_distance_move`;
- `edit_candidate_selection`;
- `edit_candidate_ranking`.

Required metrics:

- exact parsed answer accuracy;
- verifier success;
- signed direction accuracy;
- magnitude MAE;
- median absolute error;
- within-tolerance rate;
- overshoot and undershoot rate;
- invalid output rate;
- task-specific confusion matrices.

For numeric movement tasks, report both strict canonical answer accuracy and
semantic numeric correctness when the parser can extract a valid number from a
non-canonical answer. Strict accuracy remains the official score.

## P1: Feature Exposure And Structured Diagnostics

Run training/evaluation ablations that expose trusted geometry features or edit
candidate features.

Conditions:

- public code-only prompt;
- training-only `edit_geometry` features;
- training-only `edit_geometry_with_candidates` features;
- feature-exposed evaluation;
- code-only evaluation after feature-exposed training.

Feature exposure must be labeled as a CAD-agent/curriculum condition, not as the
closed-book benchmark setting. The paper should distinguish:

- mental execution of CAD code by the LLM;
- use of structured CAD measurements supplied by external tools;
- hybrid CAD-agent pipelines that combine both.

Report whether feature exposure improves:

- repair direction;
- repair translation;
- target movement;
- counterfactual sensitivity;
- transfer back to code-only prompts.

## P1: Bounded RL / GRPO Experiment

Run RL only after a supervised baseline and prompt/reward smoke tests are
working. RL should not be the only training result in the paper.

Minimum conditions:

- GRPO or GSPO from base model;
- GRPO or GSPO from reasoning-SFT adapter;
- shaped edit rewards;
- feature-exposed edit curriculum if needed;
- guided or constrained final-answer generation as the default serious RL
  recipe when supported;
- one small unconstrained rollout canary to estimate parse-failure and
  policy-collapse risk.

Required logging:

- reward mean and standard deviation;
- exact quality metrics on a fixed held-out subset;
- invalid output rate;
- completion length;
- reasoning-section length when `<think>...</think>` is used;
- truncation rate;
- KL or equivalent policy-distance metric when available;
- sampled rollout completions for debug/eval batches;
- reward components per sampled completion;
- best-checkpoint selection by held-out quality.

Required comparison:

- base zero-shot;
- answer-only SFT;
- reasoning SFT;
- rejection-sampled reasoning SFT when used;
- RL final checkpoint;
- RL best checkpoint.

If RL improves reward but not strict held-out exact accuracy or verifier success,
the paper should say so directly.

For the default-budget RL result, the preferred serious condition is SFT+GRPO:
initialize GRPO/GSPO from the best deterministic or rejection-sampled
reasoning-SFT adapter, run a 50-100 step canary, and only then run the
1,500-2,000 step main job. A base-model GRPO run is still useful as a diagnostic
or ablation, but it should not replace the SFT+GRPO result when the paper claims
that training improves CAD reasoning.

## P1: Guided Generation And Reasoning Preservation

The failed naive RL experiments exposed two mitigation ideas that should be part
of the default serious recipe where available:

- guided or constrained generation to keep rollouts parseable and on-schema;
- reasoning-length shaping to prevent the policy from collapsing to terse
  shortcut answers while also avoiding filler.

Guided final-answer generation should be the default for serious training and
evaluation runs when the backend supports it. This is appropriate because CAD
agent pipelines usually need schema-valid actions, and spending most of the GPU
budget on parse failures is not informative. The reported setting must be
clearly labeled, for example `schema-constrained`.

Bounded diagnostic conditions:

- unconstrained decoding;
- answer-tag constrained decoding;
- task-specific grammar or structured-output decoding;
- vLLM guided decoding if supported by the training stack;
- constrained final answer only, while leaving reasoning text unconstrained.

Unconstrained decoding should be run as a bounded diagnostic, not a full
required comparison. Use a small fixed validation subset or a short rollout
canary to measure parse-failure rate, answer collapse, and whether schema
guidance hides task incompetence. Do not double the compute budget by repeating
every full experiment in unconstrained mode unless the diagnostic result makes
that comparison scientifically necessary.

Report strict accuracy, invalid output rate, reward, completion length,
throughput, and whether the constraint hides or exposes task competence.

Default completion-length policy:

- 256 completion tokens is the efficient default for main closed-book
  evaluations and bounded GRPO unless a task family clearly needs more room;
- run a 256 versus 512 token validation diagnostic for reasoning-SFT and GRPO
  checkpoints;
- use 1,024 token completions only on a small subset unless the 512-token
  diagnostic shows a material gain.

Reasoning-preservation conditions:

- no reasoning-length reward;
- minimum-length reward only;
- structured reasoning-length band;
- structured reasoning-length band plus repetition/filler penalty.

Do not rely on a blind `min_new_tokens` setting as the main method. A useful
reasoning-length reward should require:

- a nonempty `<think>` section;
- bounded useful reasoning length, initially around 64-192 tokens for RL;
- no obvious repetition or filler;
- a short schema-valid final answer;
- task-dependent exceptions for tasks where concise reasoning is sufficient.

Early reasoning-SFT traces may be longer, around 128-256 tokens, if they contain
real geometry reasoning. RL should use softer task-dependent length shaping so
the model is not rewarded for padding.

## P1: Reward Ablation For Edit Tasks

If RL is included, compare reward designs.

Reward conditions:

- exact-answer reward only;
- format plus exact-answer reward;
- shaped numeric reward;
- verifier-style reward;
- candidate-aware reward for conservative repair rows;
- structured reasoning-length reward, with and without filler/repetition
  penalties.

Report:

- reward-learning stability;
- exact answer accuracy;
- verifier success;
- numeric MAE;
- invalid output rate;
- task-wise regressions;
- whether the reward encourages terse answer collapse or preserves useful
  reasoning.

## P2: Model-Size Scaling

With expanded compute, repeat the best supervised recipe across model sizes.

Suggested models:

- 3-4B;
- 7-9B;
- 14B;
- 27B or larger if feasible;
- one MoE model if training throughput and memory are acceptable.

Report:

- parameter count;
- trainable adapter parameter count;
- tokens/sec or examples/sec;
- GPU hours;
- cost-normalized improvement;
- held-out split metrics;
- whether larger models improve hard splits more than random split.

## P2: Full RL Variant Comparison

With expanded compute, compare:

- GRPO;
- GSPO;
- Dr.GRPO if implemented cleanly;
- SFT plus GRPO;
- SFT plus GSPO;
- reward variants from the P1 reward ablation.

Keep dataset, prompt, sampling, and evaluation fixed. Use the same checkpoint
selection policy for every variant.

## P2: Synthetic-To-CADEvolve Transfer

Train and evaluate cross-source transfer.

Conditions:

- train on synthetic primitives only, evaluate on CADEvolve;
- train on CADEvolve only, evaluate on synthetic challenge fixtures;
- train on mixed data;
- train on curriculum: synthetic first, CADEvolve second.

This experiment should test whether simple geometric curricula teach reusable
spatial rules or overfit to primitive shapes.

## P2: Downstream CAD-Agent Evaluation

If a separate CAD-using LLM pipeline exists, evaluate whether IntersectionQA or
IntersectionEdit training improves it.

Possible downstream metrics:

- number of generated CAD programs with verified non-interference;
- repair success rate after model-suggested edits;
- number of verifier iterations required;
- final design validity;
- regression on general coding ability if measured.

This experiment should be clearly separated from the benchmark leaderboard. It
answers whether the benchmark improves a real CAD-agent workflow.

## Default-Budget Execution Plan

Under the default budget, run experiments in this order:

1. Dataset and benchmark characterization.
2. Heuristic baselines and the exact CAD-kernel/tool-assisted upper bound.
3. Closed-book zero-shot model suite.
4. Standard instruct-model-plus-tools baseline on the same evaluation protocol,
   or on a bounded validation/test subset if full tool-use evaluation is too
   expensive.
5. Prompting and guided-decoding ablation on a fixed validation subset.
6. Answer-only Qwen3.5 4B SFT.
7. Deterministic reasoning-SFT Qwen3.5 4B.
8. Rejection-sampled reasoning-SFT if a 1,500-2,000 step GRPO run is planned.
9. Data scaling for the best supervised recipe.
10. Counterfactual sensitivity.
11. Generalization by split.
12. Error taxonomy.
13. One bounded SFT+GRPO run from the best reasoning-SFT checkpoint if SFT
    baselines and reward smoke tests are healthy.

If time or GPU budget is tight, prefer completing the supervised scaling and
counterfactual experiments over launching a long RL run. If GRPO is launched,
prefer SFT+GRPO over base-model GRPO for the main claim.

## Expanded-Budget Execution Plan

With roughly 3x compute, add:

1. Model-size scaling.
2. Fuller task transfer matrix.
3. Full IntersectionEdit evaluation.
4. Feature-exposure and structured-diagnostic ablations.
5. Reward ablation.
6. RL variant comparison.
7. Synthetic-to-CADEvolve transfer.
8. Downstream CAD-agent evaluation if the external pipeline is ready.

## Artifact Requirements

Every experiment should follow the run artifact contract in
`docs/experiment_execution_runbook.md`. At minimum, every run should write or
preserve:

- run manifest;
- exact command;
- environment summary;
- git status and diff for GPU runs;
- metrics JSONL;
- artifact index;
- prediction JSONL for evaluated rows;
- selected checkpoint or adapter;
- best checkpoint or adapter when different from final;
- generated comparison tables;
- failure-case analysis report;
- dataset manifest or release-candidate report used for the run.

GRPO/RL runs should additionally preserve capped rollout samples and reward
component logs. Full logging of every rollout is not required and should not be
the default.

Training artifacts should be uploadable to durable storage. Prefer Hugging Face
Xet-backed buckets for run artifacts because they fit ML artifact workflows
better than standard S3-style storage. Paper tables should be reproducible from
checked-in scripts and saved prediction files, not from manual spreadsheet
edits.

## Main Paper Tables

Recommended table set:

- dataset statistics and split table;
- heuristic/tool baseline table, separating exact oracle, instruct+tools, and
  trained tool-using conditions;
- closed-book zero-shot model table;
- supervised training table: zero-shot versus answer-SFT versus reasoning-SFT
  versus rejection-sampled reasoning-SFT when used;
- data scaling table or curve;
- counterfactual sensitivity table;
- task transfer matrix;
- IntersectionEdit verifier table;
- SFT+GRPO and RL/reward ablation table if RL is included;
- failure taxonomy table.

## Main Paper Figures

Recommended figures:

- dataset task/relation distribution;
- data scaling curves by split;
- confusion matrix for relation classification;
- near-boundary/counterfactual flip accuracy;
- edit numeric error distribution;
- reward versus held-out exact accuracy for RL runs;
- cost-normalized improvement if model-size scaling is included.

## Non-Goals

The paper should not spend default-budget compute on:

- broad hyperparameter sweeps without a fixed evaluation protocol;
- many RL variants before supervised baselines are established;
- training on validation or test rows;
- public claims based only on training reward;
- leaderboard tables without split and task breakdowns;
- multi-object repair unless the verifier and dataset support are mature.
