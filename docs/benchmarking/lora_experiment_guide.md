# LoRA / Precision Experiment Guide

This guide extends the existing benchmark flow so future LoRA and precision
experiments can reuse the same 30/90-sample evaluation pipeline.

## Goals

Each experiment should record:

- experiment name and git commit
- base model / LoRA adapter
- precision
- batch size
- mean / P95 latency
- peak VRAM
- model / adapter size
- load time
- success / failure
- VQA accuracy
- Grounding mean IoU, Acc@0.5, Acc@0.7

## Precision Interface

`src/utils/precision_backend.py` defines a unified interface for:

- `BF16` — runnable now
- `INT8` — planned, do not run large-scale experiments yet
- `INT4` — planned, do not run large-scale experiments yet

Dry-run check:

```bash
python src/inference/run_experiment_batch.py \
  --experiment-name int8_planned \
  --base-model-path models/Qwen3-VL-2B-Instruct \
  --output-dir results/tmp/int8_planned \
  --precision INT8 \
  --dry-run
```

## Existing 30-sample Results

Rebuild the unified report and append registry from existing outputs:

```bash
python scripts/register_experiment.py \
  --experiment-name qwen3_vl_2b_zero_shot_bf16 \
  --results-dir results/vrsbench_mini/qwen3_vl_2b_zero_shot \
  --base-model Qwen/Qwen3-VL-2B-Instruct \
  --precision BF16
```

Or:

```bash
bash scripts/run_experiment_eval.sh --report-only
```

## Future LoRA Workflow

After training produces an adapter directory:

```bash
EXPERIMENT_NAME=qwen3_vl_2b_lora_v1_bf16 \
LORA_ADAPTER=checkpoints/lora_v1 \
OUTPUT_DIR=results/vrsbench_mini/qwen3_vl_2b_lora_v1 \
bash scripts/run_experiment_eval.sh
```

Current behavior:

- adapter path is validated and written to `experiment.json`
- actual adapter loading is reserved for a later integration step
- the same output layout (`predictions.jsonl`, `summary.json`, `metrics.json`, `benchmark_report.json`) is preserved

## 90-sample Evaluation

Use the same commands with a different input JSONL once the 90-sample split is available:

```bash
INPUT_PATH=datasets/mini_eval/mini_eval_90.jsonl \
OUTPUT_DIR=results/vrsbench_mini/qwen3_vl_2b_zero_shot_90 \
bash scripts/run_experiment_eval.sh
```

## Output Files

Each experiment directory should contain:

- `experiment.json`
- `run_config.json`
- `predictions.jsonl`
- `summary.json`
- `metrics.json`
- `benchmark_report.json`

The `benchmark_report.json` now includes an `experiment` section with unified
resource and evaluation fields.

## Do Not

- modify the model training mainline in this task
- run large-scale INT8 / INT4 experiments yet
- overwrite `results/experiment_registry.csv` headers or historical rows
