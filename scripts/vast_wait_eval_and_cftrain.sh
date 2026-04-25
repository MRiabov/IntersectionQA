#!/usr/bin/env bash
set -euo pipefail

BASE_PID="${BASE_PID:-15919}"
BASE_DATASET="${BASE_DATASET:-/root/IntersectionQA-90K}"
CF_DATASET="${CF_DATASET:-/root/IntersectionQA-90K-cftrain}"
MODEL="${MODEL:-unsloth/Qwen3.5-4B}"
BASE_OUTPUT="${BASE_OUTPUT:-/root/outputs/sft_unsloth_qwen3p5_4b_intersectionqa_90k_2048_tpack_answer_b32}"
CF_OUTPUT="${CF_OUTPUT:-/root/outputs/sft_unsloth_qwen3p5_4b_intersectionqa_90k_cftrain_2048_tpack_answer_b32_qmetrics}"
BASE_LOG="${BASE_LOG:-/root/sft_unsloth_qwen3p5_4b_90k_2048_tpack_answer_b32.log}"
CF_LOG="${CF_LOG:-/root/sft_unsloth_qwen3p5_4b_90k_cftrain_2048_tpack_answer_b32_qmetrics.log}"
CONTROLLER_LOG="${CONTROLLER_LOG:-/root/wait_eval_and_cftrain.log}"

log() {
  printf '[controller] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%S%z)" "$*" | tee -a "$CONTROLLER_LOG"
}

wait_for_baseline() {
  log "waiting for baseline pid=${BASE_PID}"
  while kill -0 "$BASE_PID" 2>/dev/null; do
    sleep 60
  done
  log "baseline pid=${BASE_PID} exited"
}

evaluate_adapter() {
  local dataset_dir="$1"
  local output_dir="$2"
  local output_json="$3"
  if [[ ! -d "${output_dir}/adapter" ]]; then
    log "adapter missing at ${output_dir}/adapter; skipping eval"
    return 0
  fi
  log "evaluating adapter=${output_dir}/adapter dataset=${dataset_dir}"
  python /root/evaluate_text_model.py \
    --dataset-dir "$dataset_dir" \
    --model "$MODEL" \
    --adapter-dir "${output_dir}/adapter" \
    --splits validation test_random test_near_boundary \
    --max-rows-per-split 256 \
    --max-new-tokens 16 \
    --output-json "$output_json" \
    >> "$CONTROLLER_LOG" 2>&1 || log "eval failed for ${output_dir}; continuing"
}

run_cftrain() {
  if [[ -f "${CF_OUTPUT}/train_result.json" ]]; then
    log "cftrain result already exists at ${CF_OUTPUT}/train_result.json; skipping training"
    return 0
  fi
  log "starting cftrain dataset=${CF_DATASET} output=${CF_OUTPUT}"
  python /root/text_sft_train_unsloth.py \
    --dataset-dir "$CF_DATASET" \
    --model "$MODEL" \
    --output-dir "$CF_OUTPUT" \
    --eval-splits validation test_near_boundary \
    --max-eval-rows 1024 \
    --num-train-epochs 1 \
    --max-steps -1 \
    --max-seq-length 2048 \
    --per-device-train-batch-size 32 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --learning-rate 2e-4 \
    --warmup-ratio 0.03 \
    --logging-steps 10 \
    --eval-strategy no \
    --no-final-eval \
    --save-steps 100 \
    --save-total-limit 3 \
    --pack-tokenized \
    --assistant-only-loss \
    --metrics-log-file train_metrics.jsonl \
    --quality-eval-steps 100 \
    --quality-eval-max-rows 64 \
    --quality-metrics-log-file quality_metrics.jsonl \
    --quality-max-new-tokens 16 \
    > "$CF_LOG" 2>&1
  log "cftrain finished"
}

wait_for_baseline
evaluate_adapter "$BASE_DATASET" "$BASE_OUTPUT" "${BASE_OUTPUT}/eval_after_train.json"
run_cftrain
evaluate_adapter "$CF_DATASET" "$CF_OUTPUT" "${CF_OUTPUT}/eval_after_train.json"
log "done"
