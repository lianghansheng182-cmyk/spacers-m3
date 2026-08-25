# SPACERS-M3 Qwen3-VL LoRA Benchmarks

Project path: `/home/lhs/spacers_m3`

Frozen model policy:

- Base model: Qwen3-VL-2B-Instruct
- Main adapter: LoRA v1
- Effect reference: LoRA v1 BF16
- Deployment candidate: LoRA v1 INT4 NF4
- LoRA v2 is kept only as an ablation result

No LoRA v3, prompt tuning, or benchmark-set modification is part of the current delivery branch.

## Environment

```bash
cd /home/lhs/spacers_m3
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Expected local paths:

- Base model: `models/Qwen3-VL-2B-Instruct`
- LoRA v1 adapter: `results/lora/qwen3_vl_2b_lora_v1/adapter_final`
- Raw external benchmark data: `data/raw/`
- VRSBench internal eval data: `datasets/vrsbench_eval_v1/resolved_manifest.jsonl`

`data/raw/`, model weights, and adapter checkpoints are intentionally excluded from Git.

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

Use `--limit N` for smoke runs where supported. Use `--precision bf16`, `--precision int8`, or `--precision int4` to select the backend.

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

LEVIR-CC official test split:

```bash
python scripts/benchmarks/run_benchmark.py levir-cc \
  --precision int4 \
  --output-dir results/levir_cc
```

LEVIR-CC METEOR needs Java for the official scorer. BLEU, ROUGE-L, and CIDEr run without Java in the current environment.

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

## Data Policy

- Do not train on `datasets/mini_eval/` or `datasets/vrsbench_eval_v1/`.
- Do not modify the 30/90 VRSBench evaluation sets.
- Do not commit raw benchmark downloads, base model weights, large checkpoints, or full adapter weights.
- Keep Lite, partial, internal validation, and official split results explicitly labeled.
