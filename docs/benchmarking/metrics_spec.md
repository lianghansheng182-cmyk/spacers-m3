# Benchmark Metrics Specification

This document defines the unified benchmark report produced by
`src/utils/benchmark_metrics.py`.

## Input Sources

The tool reads existing batch inference outputs under:

```text
results/vrsbench_mini/qwen3_vl_2b_zero_shot/
```

Primary files:

| File | Purpose |
|------|---------|
| `predictions.jsonl` | Per-sample latency, VRAM, predictions |
| `summary.json` | Run-level aggregates and environment |
| `run_config.json` | Model and inference configuration |
| `metrics.json` | VQA / grounding evaluation metrics |

The tool does not rerun inference and does not modify evaluation logic.

## Output

```text
results/vrsbench_mini/qwen3_vl_2b_zero_shot/benchmark_report.json
```

Sections:

- `overall`
- `caption`
- `vqa`
- `grounding`
- `environment`
- `model_config`
- `metadata`

## Metrics

### Implemented

- Model load time
- Per-sample inference latency
- Mean / P95 latency
- Peak VRAM
- Estimated output token count when explicit counts are absent
- Token generation speed
- Total runtime
- Success / failure / failure rate
- Caption / VQA / grounding task breakdown
- VQA accuracy and grounding IoU from `metrics.json`
- GPU / CPU / OS / PyTorch / CUDA version
- Precision / batch size / max_new_tokens / image resolution

### Not Implemented

- `ttft_s` is always `"暂未实现"`
- Input token counts remain `null` unless present in source data

When `output_tokens` is missing from predictions, the tool estimates it
using whitespace-separated word count from the prediction text.

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

## Notes

- Metadata paths are stored relative to the project root.
- Empty result directories produce a valid zero-sample report.
- The module resolves project paths via `Path(__file__)`, so it works from any working directory.
