# SPACERS-M3 Qwen3-VL LoRA Benchmarks

All commands below assume the repository root as the working directory. Do not hard-code a user-specific path such as `/home/lhs`.

Recommended checkout:

```bash
git clone -b release/submission-final https://github.com/lianghansheng182-cmyk/spacers-m3.git spacers_m3
cd spacers_m3
```

Frozen model policy:

- Base model: Qwen3-VL-2B-Instruct
- Main adapter: LoRA v1
- Effect reference: LoRA v1 BF16
- Deployment candidate: LoRA v1 INT4 NF4
- LoRA v2 is kept only as an ablation result

No LoRA v3, prompt tuning, or benchmark-set modification is part of the current delivery branch.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Expected local paths:

- Base model: `models/Qwen3-VL-2B-Instruct`
- LoRA v1 adapter: `results/lora/qwen3_vl_2b_lora_v1/adapter_final`
- Raw external benchmark data: `data/raw/`
- VRSBench internal eval data: `datasets/vrsbench_eval_v1/resolved_manifest.jsonl`

`data/raw/`, model weights, and adapter checkpoints are intentionally excluded from Git.

Place external artifacts under the repository root before running model inference:

```text
spacers_m3/
├── models/
│   └── Qwen3-VL-2B-Instruct/
├── results/
│   └── lora/
│       └── qwen3_vl_2b_lora_v1/
│           └── adapter_final/
└── data/
    └── raw/
        ├── mme_realworld_lite/
        ├── xlrs_bench_lite/
        └── levir_cc/
```

The tracked VRSBench mini and 90-sample internal validation images are already under `datasets/`. External benchmark raw data and model artifacts must be supplied separately.

## Quick Checks

```bash
nvidia-smi
python scripts/check_environment.py
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

Minimal INT4 main-path smoke:

```bash
python scripts/benchmarks/run_benchmark.py vrsbench \
  --precision int4 \
  --limit 1 \
  --output-dir results/smoke/vrsbench_int4
```

BF16 fallback smoke:

```bash
python scripts/benchmarks/run_benchmark.py vrsbench \
  --precision bf16 \
  --limit 1 \
  --output-dir results/smoke/vrsbench_bf16
```

SpaceBudget AUTO smoke:

```bash
python scripts/benchmarks/run_benchmark.py mme-rs-lite \
  --precision int4 \
  --resource-mode auto \
  --limit 1 \
  --output-dir results/smoke/mme_spacebudget_auto
```

HiRes-View grid_2x2 smoke:

```bash
python scripts/benchmarks/run_benchmark.py xlrs-lite \
  --precision int4 \
  --tiling-mode grid_2x2 \
  --limit 1 \
  --output-dir results/smoke/xlrs_highres_grid_2x2
```

Benchmark dispatcher help:

```bash
python scripts/benchmarks/run_benchmark.py --help
```

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
