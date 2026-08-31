# SPACERS-M3 Qwen3-VL LoRA Benchmarks

All commands below assume the repository root as the working directory. Do not hard-code a user-specific absolute path.

Recommended checkout:

```bash
git clone -b release/submission-final https://github.com/lianghansheng182-cmyk/spacers-m3.git spacers_m3
cd spacers_m3
```

## Verified Runtime

Recommended validation environment:

- WSL2 / Linux
- Ubuntu 24.04
- Python 3.12
- CUDA 13.0 runtime
- NVIDIA GPU
- Verified GPU: RTX 5060 Laptop, compute capability sm_120

The locked wheel configuration in `requirements-verified.txt` and `requirements-linux-cu130.txt` is for the verified Linux/WSL CUDA 13 environment.

Windows users should prefer WSL2 Ubuntu or Linux for formal reproduction. Native Windows Python is limited to static checks or a small single-image fallback path when users install their own compatible Windows wheels; it is not the validated full inference or benchmark smoke environment.

Detailed reproduction guide:

[Detailed Reproduction Guide](docs/reproduction_guide.md)

Frozen model policy:

- Base model: Qwen3-VL-2B-Instruct
- Main adapter: LoRA v1
- Effect reference: LoRA v1 BF16
- Deployment candidate: LoRA v1 INT4 NF4
- LoRA v2 is kept only as an ablation result

No LoRA v3, prompt tuning, or benchmark-set modification is part of the current delivery branch.

## Environment

Verified Linux/WSL path:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-verified.txt
```

Equivalent one-click setup for the verified Ubuntu/Linux path:

```bash
bash scripts/setup_linux.sh
```

`requirements.txt` is a generic Python package list and intentionally does not pin the Linux CUDA wheel URL. Use it only after installing a compatible PyTorch build for your platform.

## Artifacts

Expected final local paths:

- Base model: `models/Qwen3-VL-2B-Instruct`
- LoRA v1 adapter: `results/lora/qwen3_vl_2b_lora_v1/adapter_final`
- Raw external benchmark data: `data/raw/`
- VRSBench internal eval data: `datasets/vrsbench_eval_v1/resolved_manifest.jsonl`

In the official submission package, the LoRA v1 adapter is included under `03_model/qwen3_vl_2b_lora_v1_adapter/`. `scripts/check_artifacts.py` detects this packaged adapter directly from `02_source_code/` without requiring `scripts/smoke_test.sh` to run first. The smoke script also links it into the code default path when needed.

`data/raw/`, base model weights, and training checkpoints are intentionally excluded from Git. External artifacts for the submission package are only the base model, public benchmark raw data, and any separately hosted large files.

Place external base model and raw benchmark data under the source-code root before running model inference:

```text
spacers_m3/
├── models/
│   └── Qwen3-VL-2B-Instruct/
└── data/
    └── raw/
        ├── mme_realworld_lite/
        ├── xlrs_bench_lite/
        └── levir_cc/
```

The tracked VRSBench mini and 90-sample internal validation images are already under `datasets/`. External benchmark raw data and the base model must be supplied separately.

Artifact policy:

- Included with Git/submission: source code, configs, docs, small result summaries, `datasets/mini_eval`, and `datasets/vrsbench_eval_v1`.
- Included with final submission package only: LoRA v1 `adapter_final` under `03_model/`.
- Supplied separately: `models/Qwen3-VL-2B-Instruct` and external benchmark raw data under `data/raw/`.
- Public benchmark raw data is excluded from Git because of size and redistribution constraints.
- Base model weights and training checkpoints are excluded from Git because of size.

LoRA v1 adapter reference:

- Expected directory: `results/lora/qwen3_vl_2b_lora_v1/adapter_final/`
- Submission package directory: `../03_model/qwen3_vl_2b_lora_v1_adapter/` when running from `02_source_code/`
- Expected size: about 34 MB
- Key files: `adapter_config.json`, `adapter_model.safetensors`, `README.md`
- SHA256 manifest: `docs/lora_v1_adapter_manifest.sha256`

Base model reference:

- Model: `Qwen3-VL-2B-Instruct`
- Expected directory: `models/Qwen3-VL-2B-Instruct/`
- The main code path uses the local directory and does not depend on an online Hugging Face API. Download the model before running offline inference.

Benchmark data reference:

- VRSBench smoke/internal validation: included under `datasets/mini_eval/` and `datasets/vrsbench_eval_v1/`.
- MME-RealWorld-RS Lite: place under `data/raw/mme_realworld_lite/`.
- XLRS-Bench Lite: place under `data/raw/xlrs_bench_lite/`. Current committed XLRS experiment is PARTIAL, 44 / 74 shards.
- LEVIR-CC: place under `data/raw/levir_cc/`.

## Quick Checks

```bash
nvidia-smi
python scripts/check_environment.py
python scripts/check_artifacts.py
python scripts/benchmarks/run_benchmark.py --help
```

Single-image smoke test:

```bash
bash scripts/run_single.sh
```

This uses `datasets/mini_eval/images/00002_0000.png` by default and writes `results/smoke_test.json`. Override paths when needed:

```bash
IMAGE=/path/to/image.png \
QUESTION="Describe this remote sensing image." \
OUTPUT=results/custom_smoke.json \
bash scripts/run_single.sh
```

If bash is unavailable, run the equivalent Python entry directly:

```bash
python src/inference/run_single.py \
  --model-path models/Qwen3-VL-2B-Instruct \
  --image datasets/mini_eval/images/00002_0000.png \
  --question "Describe this remote sensing image." \
  --output results/smoke_test.json \
  --max-pixels 1048576
```

This single-image smoke validates local Qwen3-VL model loading and image inference. The LoRA v1 adapter is validated by the INT4/BF16 benchmark smoke commands below.

## Unified Benchmark Entry

The frozen benchmark dispatcher is:

```bash
python scripts/benchmarks/run_benchmark.py <benchmark> --precision int4
```

Supported benchmark names:

- `vrsbench`
- `mme-rs-lite`
- `xlrs-lite`
- `levir-cc`

Use `--limit N` for smoke runs where supported. Use `--precision bf16`, `--precision int8`, or `--precision int4` to select the backend. For MME-RS Lite and XLRS-Lite, `--tiling-mode grid_2x2` enables the high-resolution multi-view ablation.

For MME-RS Lite and XLRS-Lite, SpaceBudget-RS can select the visual compute budget through `--resource-mode fixed|eco|balanced|precision|auto`. `fixed` preserves the explicit `--max-pixels` value. `auto` is a deterministic task-aware rule that uses only pre-inference task/question/image metadata.

## Smoke Sequence

One-click smoke after artifacts are placed:

```bash
bash scripts/smoke_test.sh
```

The script runs the environment check, artifact check, INT4 main path, BF16 fallback, SpaceBudget AUTO, HiRes grid_2x2, and OrbitGuard integrity smoke. The individual commands below are kept for debugging.

Expected final smoke output:

- environment profile is printed and saved to `logs/environment_info.txt`
- `scripts/check_artifacts.py` prints the detected LoRA adapter path and SHA256 values
- INT4 VRSBench smoke writes one successful prediction with `failure_count: 0`
- BF16 VRSBench fallback smoke writes one successful prediction with `failure_count: 0`
- SpaceBudget AUTO MME-RS smoke writes one successful prediction and official evaluator metadata
- HiRes grid_2x2 XLRS smoke writes one successful prediction and visual-token statistics
- OrbitGuard integrity smoke writes `orbitguard_summary.json` with healthy/faulted/recovered metrics

The latest WSL2 Ubuntu full-smoke logs are saved under `docs/smoke_logs/` and are copied into `05_reproduction/logs/` in the submission package.

Step 1: environment check.

```bash
python scripts/check_environment.py
```

Expected output: OS, Python, torch, CUDA, GPU, compute capability, and key package versions. Warnings are diagnostic and do not mask the root cause.

Step 2: artifact precheck.

```bash
python scripts/check_artifacts.py
```

Expected output: `[PASS]` for tracked VRSBench data, base model, and LoRA adapter. In the submission package, the checker reports the packaged `../03_model/qwen3_vl_2b_lora_v1_adapter/` path and SHA256 values even before `scripts/smoke_test.sh` creates the runtime symlink.

For source-only static checks before public raw data is placed:

```bash
python scripts/check_artifacts.py --skip-raw
```

Step 3: minimal INT4 main-path smoke.

```bash
python scripts/benchmarks/run_benchmark.py vrsbench \
  --precision int4 \
  --limit 1 \
  --output-dir results/smoke/vrsbench_int4
```

Expected output: one prediction, `failure_count: 0`, and summary files under `results/smoke/vrsbench_int4/`.

Step 4: BF16 fallback smoke.

```bash
python scripts/benchmarks/run_benchmark.py vrsbench \
  --precision bf16 \
  --limit 1 \
  --output-dir results/smoke/vrsbench_bf16
```

Expected output: one prediction, `failure_count: 0`, and summary files under `results/smoke/vrsbench_bf16/`.

Step 5: SpaceBudget AUTO smoke.

```bash
python scripts/benchmarks/run_benchmark.py mme-rs-lite \
  --precision int4 \
  --resource-mode auto \
  --limit 1 \
  --output-dir results/smoke/mme_spacebudget_auto
```

Expected output: one MME-RS Lite prediction, official evaluator return code 0, and summary files under `results/smoke/mme_spacebudget_auto/`.

Step 6: HiRes-View grid_2x2 smoke.

```bash
python scripts/benchmarks/run_benchmark.py xlrs-lite \
  --precision int4 \
  --tiling-mode grid_2x2 \
  --limit 1 \
  --output-dir results/smoke/xlrs_highres_grid_2x2
```

Expected output: one XLRS prediction, visual token statistics, and summary files under `results/smoke/xlrs_highres_grid_2x2/`.

Step 7: OrbitGuard integrity smoke.

```bash
python scripts/reliability/run_fault_injection.py \
  --config configs/reliability/orbitguard_v1.yaml \
  --max-fault-runs 1 \
  --output-dir results/smoke/orbitguard_integrity
```

Expected output: clean integrity PASS, corruption DETECTED, recovery PASS, and summary files under `results/smoke/orbitguard_integrity/`.

Step 8: run full benchmarks only after the smoke sequence passes.

```bash
python scripts/benchmarks/run_benchmark.py vrsbench --precision int4
python scripts/benchmarks/run_benchmark.py mme-rs-lite --precision int4
python scripts/benchmarks/run_benchmark.py xlrs-lite --precision int4
python scripts/benchmarks/run_benchmark.py levir-cc --precision int4
```

Expected output: benchmark-specific prediction and metric files under `results/`.

## Reproduction Commands

VRSBench 90-sample internal validation:

```bash
python scripts/benchmarks/run_benchmark.py vrsbench \
  --precision int4 \
  --output-dir results/lora/qwen3_vl_2b_lora_v1_int4_90
```

MME-RealWorld-Lite Remote Sensing subset:

```bash
python scripts/benchmarks/run_benchmark.py mme-rs-lite \
  --precision int4 \
  --output-dir results/mme_realworld_rs
```

XLRS-Bench-lite smoke or partial/full local run:

```bash
python scripts/benchmarks/run_benchmark.py xlrs-lite \
  --precision int4 \
  --limit 5 \
  --output-dir results/xlrs_bench/smoke_int4
```

```bash
python scripts/benchmarks/run_benchmark.py xlrs-lite \
  --precision int4 \
  --output-dir results/xlrs_bench/full_lite_int4
```

Current committed XLRS result is a 44 / 74 shard partial run at `results/xlrs_bench/partial_44shards_int4/`, because the remaining HuggingFace shards were not available locally.

High-resolution global+2x2 tiling ablation:

```bash
python scripts/benchmarks/run_benchmark.py xlrs-lite \
  --precision int4 \
  --tiling-mode grid_2x2 \
  --output-dir results/highres/xlrs_tiling_44shards_int4
```

```bash
python scripts/benchmarks/run_benchmark.py mme-rs-lite \
  --precision int4 \
  --tiling-mode grid_2x2 \
  --output-dir results/highres/mme_rs_tiling_full_int4
```

SpaceBudget-RS dynamic visual-budget ablation:

```bash
python scripts/benchmarks/run_benchmark.py mme-rs-lite \
  --precision int4 \
  --resource-mode auto \
  --output-dir results/spacebudget/mme_auto
```

```bash
python scripts/benchmarks/run_benchmark.py xlrs-lite \
  --precision int4 \
  --resource-mode auto \
  --output-dir results/spacebudget/xlrs_auto_full
```

```bash
python scripts/spacebudget/build_spacebudget_reports.py
```

LEVIR-CC official test split:

```bash
python scripts/benchmarks/run_benchmark.py levir-cc \
  --precision int4 \
  --output-dir results/levir_cc
```

LEVIR-CC METEOR needs Java for the official scorer. BLEU, ROUGE-L, and CIDEr run without Java in the current environment.

OrbitGuard-RS software-level reliability ablation:

```bash
python scripts/reliability/run_fault_injection.py \
  --config configs/reliability/orbitguard_v1.yaml
python scripts/reliability/write_orbitguard_summary.py
```

Integrity-only smoke:

```bash
python scripts/reliability/run_fault_injection.py \
  --config configs/reliability/orbitguard_v1.yaml \
  --max-fault-runs 1 \
  --output-dir results/smoke/orbitguard_integrity
```

Optional real corrupted-adapter mini_eval check:

```bash
python scripts/reliability/run_fault_injection.py \
  --config configs/reliability/orbitguard_v1.yaml \
  --run-inference \
  --max-fault-runs 3 \
  --output-dir results/reliability/real_fault_eval
```

OrbitGuard-RS is software-level SEU-equivalent validation only. It is not a real space radiation experiment or hardware qualification result.

Report figures for this module are generated at:

```bash
results/reliability/figures/
```

## Current Results

The consolidated result table is tracked at:

```bash
results/final_benchmark_summary.md
```

Current benchmark status:

| Benchmark | Result type | Status |
| --- | --- | --- |
| VRSBench | internal validation | complete |
| MME-RealWorld-RS Lite | official Lite RS subset | complete |
| XLRS-Bench Lite | internal partial, 44 / 74 shards | partial |
| LEVIR-CC | official test split | complete except METEOR |

Submission readiness notes are tracked at:

```bash
results/submission_readiness.md
```

## Important Reports

- `results/lora/lora_model_selection_report.md`
- `results/quantization/lora_v1_quantization_report.md`
- `results/data_audit/remaining_benchmarks_readiness.md`
- `results/mme_realworld_rs/report.md`
- `results/xlrs_bench/report.md`
- `results/levir_cc/report.md`
- `results/final_benchmark_summary.md`
- `results/submission_readiness.md`
- `results/highres/highres_input_audit.md`
- `results/highres/highres_tiling_ablation.md`
- `results/reliability/satellite_fault_tolerance_ablation.md`
- `results/reliability/orbitguard_summary.json`

## Data Policy

- Do not train on `datasets/mini_eval/` or `datasets/vrsbench_eval_v1/`.
- Do not modify the 30/90 VRSBench evaluation sets.
- Do not commit raw benchmark downloads, base model weights, large checkpoints, or full adapter weights.
- Keep Lite, partial, internal validation, and official split results explicitly labeled.
