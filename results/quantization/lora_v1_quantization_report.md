# LoRA v1 Quantization Report

Generated: 2026-08-24T21:51:41

## Verdict

BF16 LoRA v1 is the only runnable precision in the current environment. INT8 and INT4 were not formally evaluated because the required bitsandbytes backend is missing from the active project virtual environment.

This report does not claim quantization support from API availability alone.

## Environment

```json
{
  "python": "Python 3.12.3",
  "packages": {
    "torch": "2.13.0+cu130",
    "transformers": "4.57.6",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "bitsandbytes": "MISSING",
    "qwen-vl-utils": "0.0.14",
    "pillow": "12.3.0"
  },
  "cuda_available": true,
  "torch_cuda": "13.0",
  "gpu_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
  "gpu_capability": [
    12,
    0
  ]
}
```

## Code Audit

```json
{
  "precision_backend_py": {
    "exists": false,
    "tracked": false,
    "note": "No tracked precision_backend.py was found; current precision handling is embedded in inference/evaluation scripts."
  },
  "lora_adapter_loading": {
    "script": "scripts/lora/evaluate_lora_adapter.py",
    "method": "Load Qwen3VLForConditionalGeneration first, then PeftModel.from_pretrained(base_model, adapter_path).",
    "adapter": "results/lora/qwen3_vl_2b_lora_v1/adapter_final"
  },
  "benchmark_framework": {
    "summary_files": "summary.json and metrics.json from evaluate_lora_adapter/evaluate_mini_eval",
    "resource_monitor": "src/utils/resource_monitor.py records latency_s and peak_vram_mb.",
    "ttft": "not implemented",
    "true_tokens_per_second": "not implemented; benchmark_metrics.py intentionally does not fabricate token/s from word counts."
  },
  "evaluation_protocol": {
    "prompt": "Prompt v1",
    "datasets": [
      "datasets/mini_eval/mini_eval.jsonl",
      "datasets/vrsbench_eval_v1/resolved_manifest.jsonl"
    ],
    "parser": "src/evaluation/evaluate_mini_eval.py parse_grounding_bbox"
  }
}
```

## BF16 Reference

LoRA v1 BF16 results are reused from the completed Prompt v1 evaluations with the same adapter and evaluation algorithm.

| Dataset | VQA acc | Grounding mean IoU | Acc@0.5 | Acc@0.7 | Parse failures | Peak VRAM MB | Mean latency s | P95 latency s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mini_eval 30 | 0.7 | 0.617452 | 0.8 | 0.2 | 0 | 4201.32 | 1.785099 | 4.818842 |
| vrsbench_eval_v1 90 | 0.7 | 0.443984 | 0.466667 | 0.266667 | 1 | 4203.17 | 1.768474 | 5.11779 |

## Quantization Smoke

| Precision | Status | Model loaded | Adapter loaded | Prediction checked | Reason/error |
|---|---|---:|---:|---:|---|
| INT8 | NOT_RUN | False | False | 0 | ModuleNotFoundError: No module named 'bitsandbytes' |
| INT4 | NOT_RUN | False | False | 0 | ModuleNotFoundError: No module named 'bitsandbytes' |

## Formal Quantized Eval

| Precision | 90-run status | VQA acc | Grounding mean IoU | Acc@0.5 | Acc@0.7 | Parse failures | Peak VRAM MB | Mean latency s | P95 latency s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | completed | 0.7 | 0.443984 | 0.466667 | 0.266667 | 1 | 4203.17 | 1.768474 | 5.11779 |
| INT8 | not run | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| INT4 | not run | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Storage

| Artifact | Size MB |
|---|---:|
| Base model directory | 4068.994 |
| LoRA v1 adapter | 33.311 |
| INT8 quantized model artifact | not generated |
| INT4 quantized model artifact | not generated |

## Answers

1. BF16 is the only completed precision. INT8/INT4 effects are unknown in this environment because quantized loading cannot be run without bitsandbytes.
2. VRAM savings for INT8/INT4 were not measured.
3. Storage savings for INT8/INT4 were not measured; no quantized model artifacts were generated.
4. Latency changes for INT8/INT4 were not measured.
5. The current deployment candidate remains LoRA v1 BF16.
6. No quantized precision-loss conclusion can be made until INT8/INT4 smoke tests actually load and generate predictions.

## Required Before Quantized Runs

- Install a CUDA-compatible bitsandbytes build in `.venv`.
- Add or enable an explicit precision backend for `BF16`, `INT8`, and `INT4` model loading.
- Re-run smoke on Caption/VQA/Grounding samples before any 90-sample quantized eval.
- Record real output token counts before reporting true token/s; TTFT is currently not implemented.
