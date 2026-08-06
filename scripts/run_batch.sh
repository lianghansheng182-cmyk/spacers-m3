#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

python src/inference/run_batch.py \
  --model-path "${REPO_ROOT}/models/Qwen3-VL-2B-Instruct" \
  --input "${REPO_ROOT}/datasets/mini_eval/mini_eval.jsonl" \
  --output-dir "${REPO_ROOT}/results/vrsbench_mini/qwen3_vl_2b_zero_shot" \
  --batch-size 1 \
  --max-new-tokens 256
