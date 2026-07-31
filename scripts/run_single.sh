#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/lhs/spacers_m3"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
IMAGE="${PROJECT_DIR}/data/test/demo.jpeg"
OUTPUT="${PROJECT_DIR}/results/smoke_test.json"
QUESTION="请详细描述图像中的主要场景、物体及其空间位置关系。"
export SPACERS_M3_MODEL_PATH="${SPACERS_M3_MODEL_PATH:-${PROJECT_DIR}/models/Qwen3-VL-2B-Instruct}"

cd "${PROJECT_DIR}"
exec "${PYTHON}" "${PROJECT_DIR}/src/inference/run_single.py" \
  --image "${IMAGE}" \
  --question "${QUESTION}" \
  --output "${OUTPUT}"
