# Apr 27 2026 OpenRouter Native Reasoning Distill

## Purpose

The local Qwen3.5-4B base-model rejection sampler failed because the model did
not reliably emit structured reasoning traces, even after batching and few-shot
prompt edits. The next probe uses OpenRouter native reasoning capture instead of
asking the model to place reasoning inside the visible answer format.

## Planned Method

- Dataset: `MRiabov/IntersectionQA-90K`, local mirror `data/IntersectionQA-90K`.
- Candidate rows: train split, MVP task types only:
  `binary_interference`, `relation_classification`, `volume_bucket`.
- Model: `deepseek/deepseek-v4-pro` through OpenRouter.
- Endpoint resolution: prefer `OPENROUTER_BASE_URL`, then
  `OPENROUTER_API_BASE`; if `OPENROUTER_API_KEY` is present, use direct
  `https://openrouter.ai/api/v1`; otherwise fall back to `OPENAI_API_URL` or
  the public OpenRouter URL. This avoids accidentally using a stale local proxy
  when a direct OpenRouter token is configured.
- Native reasoning switch: pass `reasoning: {"enabled": true}` in the Chat
  Completions request.
- Final answer constraint: use request-level JSON schema for `final_answer`
  when the provider accepts it; fall back to plain content parsing only if the
  provider rejects JSON schema.
- Routing policy: use OpenRouter model fallbacks and provider routing instead
  of waiting on one provider. The default route sorts compatible providers by
  throughput, allows fallbacks, and requires providers to support all requested
  parameters.
- Prompt policy: do not prompt for `<think>` tags or ask the model to describe
  an output format. The system prompt only states closed-book CAD reasoning,
  no execution, and no external tools.
- Acceptance policy: keep rows only when the final answer parses exactly,
  matches the canonical dataset answer, and native OpenRouter reasoning is
  present in `message.reasoning`, `message.reasoning_content`, or text/summary
  entries in `message.reasoning_details`.
- Artifact policy: persist raw responses, extracted traces, accepted SFT rows,
  request config, and summary report. Sync probe artifacts to the HF Xet-backed
  bucket if the local token has bucket-write access.
- Implementation note: the probe script is intentionally self-contained. It
  imports no `intersectionqa.*` modules and uses stdlib HTTP/JSONL handling, with
  an optional `huggingface_hub.sync_bucket` call only for bucket persistence.
- Persistence note: per-row artifacts are append-only by default. After each
  completed row, the script flushes `raw_responses.jsonl`, `traces.jsonl`, and
  `accepted_reasoning_sft.jsonl` when applicable. Reruns skip IDs already present
  in `traces.jsonl`; use `--fresh` only to intentionally discard previous
  per-row artifacts.

## Initial Probe Command

```bash
rtk uv run python -m scripts.training.distill_reasoning_openrouter \
  --dataset-dir data/IntersectionQA-90K \
  --output-dir runs/openrouter_native_reasoning_distill_probe \
  --model deepseek/deepseek-v4-pro \
  --fallback-models deepseek/deepseek-v4-flash qwen/qwen3.5-flash-02-23 \
  --splits train \
  --task-types binary_interference relation_classification volume_bucket \
  --max-rows 20 \
  --max-tokens 9000 \
  --temperature 0.2 \
  --continue-on-error \
  --hf-bucket hf://buckets/MRIabov/intersectionqa-qwen3p5-4b-grpo-artifacts/openrouter-native-reasoning-distill/apr-27-probe
```

## Stop Rules

- Stop after the 20-row probe before any larger spend.
- Do not proceed to a large distillation run if native reasoning is absent or
  final-answer parse/correctness is too low to produce a useful accepted set.
- If accepted long traces are useful, run a separate shortening pass to produce
  128-256 token SFT traces before reasoning SFT.

## Results

Initial execution reached OpenRouter but failed before producing any rows:

- `deepseek/deepseek-v4-flash` returned an upstream provider HTTP 429 from
  DeepInfra before the first row completed.
- Output JSONL artifacts remained at 0 rows.
- No concurrency was used: one local Python process, one request at a time.
- The retry loop was stopped manually rather than waiting through the long
  provider backoff.

Update: the script now supports OpenRouter model fallbacks with compatible
reasoning/structured-output models:

- primary: `deepseek/deepseek-v4-pro`;
- fallbacks: `deepseek/deepseek-v4-flash`, `qwen/qwen3.5-flash-02-23`.
- OpenRouter currently allows at most 3 total models in one fallback request.

Final 1000-row distillation run used `qwen/qwen3.5-flash-02-23` through
Pydantic AI's OpenRouter model support, not JSON-schema/tool output:

```bash
rtk uv run python -m scripts.training.distill_reasoning_openrouter \
  --dataset-dir data/IntersectionQA-90K \
  --output-dir runs/openrouter_native_reasoning_distill_1000 \
  --model qwen/qwen3.5-flash-02-23 \
  --splits train \
  --task-types binary_interference relation_classification volume_bucket \
  --max-rows 1000 \
  --batch-size 128 \
  --max-tokens 16 \
  --reasoning-max-tokens 8192 \
  --temperature 0.2 \
  --timeout-seconds 240 \
  --retries 0 \
  --continue-on-error \
  --no-provider-routing \
  --hf-bucket hf://buckets/MRIabov/intersectionqa-qwen3p5-4b-grpo-artifacts/openrouter-native-reasoning-distill/apr-27-1000
```

The run first used batch size 32, then resumed without `--fresh` at batch size
128. Existing row IDs were skipped and all JSONL artifacts were appended.

Aggregate results:

- rows attempted: 1000;
- raw responses persisted: 1000;
- extracted traces persisted: 1000;
- native reasoning present: 1000/1000;
- strict parse-valid final answers: 894/1000;
- accepted correct reasoning-SFT rows: 659/1000;
- request errors: 0;
- accepted task counts: binary interference 243, relation classification 219,
  volume bucket 197;
- total usage: 1,658,575 input tokens and 7,836,406 output tokens;
- observed reasoning length: mean 19,188 chars, min 4,082, max 26,220;
- accepted reasoning length: mean 18,815 chars, min 4,082, max 26,161.

Artifacts were uploaded to:

`hf://buckets/MRIabov/intersectionqa-qwen3p5-4b-grpo-artifacts/openrouter-native-reasoning-distill/apr-27-1000`

Uploaded files include `raw_responses.jsonl`, `traces.jsonl`,
`accepted_reasoning_sft.jsonl`, `request_config.json`, `report.json`, and
`aggregate_report.json`.
