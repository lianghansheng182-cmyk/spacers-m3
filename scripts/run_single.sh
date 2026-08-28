#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-${PROJECT_DIR}/.venv/bin/python}"
IMAGE="${IMAGE:-${PROJECT_DIR}/datasets/mini_eval/images/00002_0000.png}"
OUTPUT="${OUTPUT:-${PROJECT_DIR}/results/smoke_test.json}"
QUESTION="${QUESTION:-请详细描述图像中的主要场景、物体及其空间位置关系。}"
export SPACERS_M3_MODEL_PATH="${SPACERS_M3_MODEL_PATH:-${PROJECT_DIR}/models/Qwen3-VL-2B-Instruct}"

cd "${PROJECT_DIR}"
exec "${PYTHON}" "${PROJECT_DIR}/src/inference/run_single.py" \
  --image "${IMAGE}" \
  --question "${QUESTION}" \
  --output "${OUTPUT}"
