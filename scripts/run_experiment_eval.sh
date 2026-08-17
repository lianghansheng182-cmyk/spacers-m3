#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_vl_2b_zero_shot_bf16}"
PRECISION="${PRECISION:-BF16}"
INPUT_PATH="${INPUT_PATH:-${REPO_ROOT}/datasets/mini_eval/mini_eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/vrsbench_mini/qwen3_vl_2b_zero_shot}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${REPO_ROOT}/models/Qwen3-VL-2B-Instruct}"
LORA_ADAPTER="${LORA_ADAPTER:-}"

cd "${REPO_ROOT}"

ARGS=(
  python src/inference/run_experiment_batch.py
  --experiment-name "${EXPERIMENT_NAME}"
  --base-model-path "${BASE_MODEL_PATH}"
  --precision "${PRECISION}"
  --input "${INPUT_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --register
)

if [[ -n "${LORA_ADAPTER}" ]]; then
  ARGS+=(--lora-adapter "${LORA_ADAPTER}")
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  ARGS+=(--dry-run)
fi

if [[ "${1:-}" == "--report-only" ]]; then
  ARGS+=(--report-only)
fi

"${ARGS[@]}"
