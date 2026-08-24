# Final Benchmark Summary

Created: 2026-08-25

Model policy is frozen:

- Base: Qwen3-VL-2B-Instruct
- Main model: LoRA v1
- BF16 effect reference: LoRA v1 BF16
- Current deployment candidate: LoRA v1 INT4 NF4
- LoRA v2: ablation only

## VRSBench 90-Sample Internal Validation

These are internal validation results on `datasets/vrsbench_eval_v1/resolved_manifest.jsonl`, not official full VRSBench leaderboard results.

| Model | Precision | VQA Acc | Grounding mean IoU | Acc@0.5 | Acc@0.7 | Parse failures | Peak VRAM MB | Mean latency s | P95 latency s | Result type |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Base Qwen3-VL-2B-Instruct | BF16 | 0.600000 | 0.436447 | 0.466667 | 0.300000 | 0 | 4184.17 | 3.492092 | 10.716380 | internal validation |
| LoRA v1 | BF16 | 0.700000 | 0.443984 | 0.466667 | 0.266667 | 1 | 4203.17 | 1.768474 | 5.117790 | internal validation |
| LoRA v1 | INT8 bitsandbytes | 0.733333 | 0.477298 | 0.466667 | 0.400000 | 0 | 2545.99 | 4.402703 | 13.659740 | internal validation |
| LoRA v1 | INT4 NF4 bitsandbytes | 0.733333 | 0.509680 | 0.600000 | 0.466667 | 1 | 1659.81 | 2.214299 | 6.429084 | internal validation |

## Remaining Benchmarks

| Benchmark | Current status | Result type | Data status | Code support | Notes |
| --- | --- | --- | --- | --- | --- |
| MME-RealWorld-RS | Not run | missing-data blocker | Missing locally | Partial | Needs official data, RS-domain manifest, A-E parser, official evaluator integration. |
| XLRS-Bench | Not run | missing-data blocker | Missing locally | Partial | Needs official data, ultra-high-resolution handling, XLRS prompt/parser/evaluator. |
| LEVIR-CC | Not run | missing-data blocker | Missing locally | Missing | Needs dual-image input adapter and caption metrics. |

## Interpretation

- The only completed benchmark family in this repository is VRSBench internal validation.
- LoRA v1 INT4 NF4 remains the deployment candidate among validated variants because it gives the lowest measured peak VRAM with no observed VQA regression against LoRA v1 BF16 on the 90-sample internal set.
- No MME-RealWorld-RS, XLRS-Bench, or LEVIR-CC smoke/official scores are claimed.
