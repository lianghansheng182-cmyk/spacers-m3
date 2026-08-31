#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PACKAGE_ADAPTER=""
for candidate in \
  "$ROOT/../03_model/qwen3_vl_2b_lora_v1_adapter"; do
  if [[ -d "$candidate" ]]; then
    PACKAGE_ADAPTER="$candidate"
    break
  fi
done
DEFAULT_ADAPTER="results/lora/qwen3_vl_2b_lora_v1/adapter_final"
if [[ ! -e "$DEFAULT_ADAPTER" && -n "$PACKAGE_ADAPTER" ]]; then
  mkdir -p "$(dirname "$DEFAULT_ADAPTER")"
  ln -s "$PACKAGE_ADAPTER" "$DEFAULT_ADAPTER"
fi

SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-results/smoke_submission}"

python scripts/check_environment.py
if [[ -n "${CHECK_ARTIFACT_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARTIFACT_ARGS=($CHECK_ARTIFACT_ARGS)
  python scripts/check_artifacts.py "${ARTIFACT_ARGS[@]}"
else
  python scripts/check_artifacts.py
fi
python scripts/benchmarks/run_benchmark.py --help >/dev/null

python scripts/benchmarks/run_benchmark.py vrsbench \
  --precision int4 \
  --limit 1 \
  --output-dir "$SMOKE_OUTPUT_DIR/vrsbench_int4"

python scripts/benchmarks/run_benchmark.py vrsbench \
  --precision bf16 \
  --limit 1 \
  --output-dir "$SMOKE_OUTPUT_DIR/vrsbench_bf16"

python scripts/benchmarks/run_benchmark.py mme-rs-lite \
  --precision int4 \
  --resource-mode auto \
  --limit 1 \
  --output-dir "$SMOKE_OUTPUT_DIR/mme_spacebudget_auto"

python scripts/benchmarks/run_benchmark.py xlrs-lite \
  --precision int4 \
  --tiling-mode grid_2x2 \
  --limit 1 \
  --output-dir "$SMOKE_OUTPUT_DIR/xlrs_highres_grid_2x2"

python scripts/reliability/run_fault_injection.py \
  --config configs/reliability/orbitguard_v1.yaml \
  --max-fault-runs 1 \
  --output-dir "$SMOKE_OUTPUT_DIR/orbitguard_integrity"

echo "[PASS] Submission smoke test completed: $SMOKE_OUTPUT_DIR"
