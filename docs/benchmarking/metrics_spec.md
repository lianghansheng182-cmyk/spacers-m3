# Benchmark Metrics Specification

This document defines the unified benchmark report produced by
`src/utils/benchmark_metrics.py`.

## Input sources

The report is generated from existing inference outputs under:

```text
results/vrsbench_mini/qwen3_vl_2b_zero_shot/
```

Primary inputs:

| File | Purpose |
|---|---|
| `predictions.jsonl` | Per-sample latency, VRAM and prediction text |
| `summary.json` | Run-level timing and recorded inference environment |
| `run_config.json` | Model and inference configuration recorded at run time |
| `metrics.json` | VQA and grounding evaluation metrics |

The report generator does **not** rerun inference and does not modify evaluation logic.

## Environment provenance

Environment fields in `benchmark_report.json` must describe the machine that
actually ran inference. The report generator therefore reads GPU / PyTorch /
CUDA and other available fields from `summary.json` and `run_config.json`.

It must **not** overwrite those fields using the computer that later runs
`generate_benchmark_report.py`.

For the current baseline, GPU / PyTorch / CUDA were recorded, while OS, CPU,
Python and driver version were not recorded during inference. Missing fields
remain `null` instead of being filled with values from another computer.

## Token and throughput fields

Tokenizer token counts are reported only when inference explicitly recorded
`output_tokens` or `generated_tokens`.

For the current 30-sample baseline, token counts were not recorded. Therefore:

- `total_output_tokens`: `null`
- `token_speed_tokens_per_s`: `null`
- `token_count_coverage`: `0.0`

The report also provides `estimated_output_words` and
`estimated_words_per_s`, calculated using whitespace-separated prediction
words. These are diagnostic estimates only and **must not be presented as
token counts or token/s**.

## Implemented metrics

- Model load time
- Per-sample inference latency
- Mean / P95 latency
- Peak VRAM
- Total runtime
- Success / failure / failure rate
- Caption / VQA / grounding task breakdown
- VQA accuracy
- Grounding mean IoU / Acc@0.5 / Acc@0.7
- Recorded GPU / PyTorch / CUDA metadata
- Precision / batch size / max_new_tokens / max_pixels
- Explicit token throughput when true token counts exist
- Diagnostic word-count throughput when true token counts are unavailable

## Not implemented

- TTFT: stored as `"暂未实现"`
- Input token count unless the inference code records it
- Driver / OS / CPU / Python for historical runs that did not record them

## Usage

```bash
python scripts/generate_benchmark_report.py
```

Custom paths:

```bash
python scripts/generate_benchmark_report.py \
  --results-dir results/vrsbench_mini/qwen3_vl_2b_zero_shot \
  --output results/vrsbench_mini/qwen3_vl_2b_zero_shot/benchmark_report.json
```

## Reproducibility notes

- Paths stored in metadata are project-relative when possible.
- Empty result directories produce a valid zero-sample report.
- The module resolves the project root from `__file__`, so it can be invoked
  from another working directory.

## Experiment section

When `experiment.json` exists in the results directory, or when using
`scripts/register_experiment.py`, the report also includes an `experiment`
section with:

- experiment name and git commit
- base model / LoRA adapter
- precision and batch size
- model / adapter size
- load time, latency, VRAM, success/failure
- VQA accuracy and grounding metrics

See `docs/benchmarking/lora_experiment_guide.md` for LoRA and precision workflow.
