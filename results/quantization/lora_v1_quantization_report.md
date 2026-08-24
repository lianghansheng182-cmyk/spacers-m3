# LoRA v1 Quantization Report

Generated: 2026-08-24T23:55:50

## Verdict

INT8 and INT4 were both truly validated with bitsandbytes on the current RTX 5060 Laptop GPU / CUDA 13 environment. Both smoke tests and both 90-sample Prompt v1 evaluations completed successfully.

INT4 is the strongest deployment candidate from this run: it has the lowest peak VRAM, faster latency than INT8, and the best 90-sample Grounding metrics, while keeping VQA equal to INT8 and above BF16. The caveat is one Grounding parse failure in INT4, caused by a malformed 6-number bbox output.

## Environment

```json
{
  "packages": {
    "torch": "2.13.0+cu130",
    "transformers": "4.57.6",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.1",
    "qwen-vl-utils": "0.0.14"
  },
  "cuda_available": true,
  "torch_cuda": "13.0",
  "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
  "compute_capability": [
    12,
    0
  ]
}
```

## Smoke Tests

| Precision | Status | Success | Load time s | Mean latency s | P95 latency s | Peak VRAM MB | Grounding parsed |
|---|---|---:|---:|---:|---:|---:|---:|
| INT8 | PASS | 3/3 | 8.045162 | 4.533836 | 9.401216 | 2540.41 | 1/1 |
| INT4 NF4 | PASS | 3/3 | 5.826323 | 3.033925 | 6.642285 | 1658.48 | 1/1 |

## VRSBench Eval v1 90

| Precision | Success | VQA acc | Grounding mean IoU | Acc@0.5 | Acc@0.7 | Parse failures | Peak VRAM MB | Load time s | Mean latency s | P95 latency s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 90/90 | 0.7 | 0.443984 | 0.466667 | 0.266667 | 1 | 4203.17 | 2.707903 | 1.768474 | 5.11779 |
| INT8 | 90/90 | 0.733333 | 0.477298 | 0.466667 | 0.4 | 0 | 2545.99 (39.43% saved) | 3.877763 | 4.402703 | 13.65974 |
| INT4 NF4 | 90/90 | 0.733333 | 0.50968 | 0.6 | 0.466667 | 1 | 1659.81 (60.51% saved) | 5.069761 | 2.214299 | 6.429084 |

## Storage

| Artifact | Size MB |
|---|---:|
| Base model directory | 4068.994 |
| LoRA v1 adapter | 33.311 |
| INT8 saved model artifact | unavailable; quantized base model was loaded but not saved |
| INT4 saved model artifact | unavailable; quantized base model was loaded but not saved |

## Precision Comparison

- BF16 reference: VQA 0.70, Grounding mean IoU 0.443984, Acc@0.5 0.466667, Acc@0.7 0.266667, parse failures 1, peak VRAM 4203.17 MB.
- INT8: VQA 0.733333, Grounding mean IoU 0.477298, Acc@0.5 0.466667, Acc@0.7 0.40, parse failures 0, peak VRAM 2545.99 MB. VRAM saved vs BF16: 39.43%.
- INT4 NF4: VQA 0.733333, Grounding mean IoU 0.50968, Acc@0.5 0.60, Acc@0.7 0.466667, parse failures 1, peak VRAM 1659.81 MB. VRAM saved vs BF16: 60.51%.

## Unsupported Metrics

- TTFT: unavailable; the current evaluator does not instrument first-token timing.
- true token/s: unavailable; the current evaluator does not record generated token counts. No word/s estimate is reported as token/s.
- quantized model storage: unavailable; no INT8/INT4 base-model artifacts were saved.

## Recommendation

Use LoRA v1 INT4 NF4 as the next deployment candidate, with one follow-up check on Grounding output formatting because INT4 still produced one malformed bbox. Keep BF16 as the accuracy reference and INT8 as a safer quantized fallback when zero parse failures are more important than VRAM and latency.
