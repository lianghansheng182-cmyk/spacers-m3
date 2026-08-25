# Final Benchmark Summary

Created: 2026-08-25

Model policy is frozen:

- Base: Qwen3-VL-2B-Instruct
- Main model: LoRA v1
- BF16 effect reference: LoRA v1 BF16
- Current deployment candidate: LoRA v1 INT4 NF4
- LoRA v2: ablation only

## VRSBench

Result type: INTERNAL VALIDATION on `datasets/vrsbench_eval_v1/resolved_manifest.jsonl`. These are not official full VRSBench leaderboard results.

| Model | Precision | VQA Acc | Grounding mean IoU | Acc@0.5 | Acc@0.7 | Parse failures | Peak VRAM MB | Mean latency s | P95 latency s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base Qwen3-VL-2B-Instruct | BF16 | 0.600000 | 0.436447 | 0.466667 | 0.300000 | 0 | 4184.17 | 3.492092 | 10.716380 |
| LoRA v1 | BF16 | 0.700000 | 0.443984 | 0.466667 | 0.266667 | 1 | 4203.17 | 1.768474 | 5.117790 |
| LoRA v1 | INT8 bitsandbytes | 0.733333 | 0.477298 | 0.466667 | 0.400000 | 0 | 2545.99 | 4.402703 | 13.659740 |
| LoRA v1 | INT4 NF4 bitsandbytes | 0.733333 | 0.509680 | 0.600000 | 0.466667 | 1 | 1659.81 | 2.214299 | 6.429084 |

## MME-RealWorld-RS Lite

Result type: OFFICIAL LITE subset. This uses MME-RealWorld-Lite and only the official `Remote Sensing` subtask, not the complete MME-RealWorld benchmark.

| Model | Precision | Samples | Accuracy | Parse failures | Peak VRAM MB | Mean latency s | P95 latency s | Protocol |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LoRA v1 | INT4 NF4 bitsandbytes | 150 | 0.466667 | 0 | 1850.04 | 0.569578 | 0.693621 | official Lite RS subset |

Category accuracy:

| Category | Samples | Accuracy |
| --- | ---: | ---: |
| Position | 50 | 0.6200 |
| Color | 50 | 0.5800 |
| Count | 50 | 0.2000 |

## XLRS-Bench Lite

Result type: INTERNAL PARTIAL. The full official Lite dataset has 3080 samples in 74 Arrow shards. Current local data contains 1848 samples in 44 / 74 shards because HuggingFace download timed out / became unreachable before completion. This result must not be reported as an official complete Lite score.

| Model | Precision | Samples | Micro Acc | Macro category Acc | Parse failures | Peak VRAM MB | Mean latency s | P95 latency s | Protocol |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LoRA v1 | INT4 NF4 bitsandbytes | 1848 | 0.389069 | 0.478757 | 0 | 1901.08 | 0.555370 | 0.681128 | partial 44-shard local result |

Category accuracy:

| Category | Samples | Accuracy |
| --- | ---: | ---: |
| Complex reasoning / Anomaly Detection and Interpretation | 100 | 0.680000 |
| Complex reasoning / Environmental condition reasoning | 100 | 0.770000 |
| Counting / Counting with complex reasoning | 100 | 0.410000 |
| Counting / Overall counting | 60 | 0.216667 |
| Land use classification / Regional Land use classification | 200 | 0.640000 |
| Object properties / Object color | 728 | 0.274725 |
| Object properties / Object motion state | 60 | 0.516667 |
| Object spatial relationship / Object spatial relationship | 500 | 0.322000 |

## LEVIR-CC

Result type: OFFICIAL SPLIT on the LEVIR-CC `test` split. BLEU, ROUGE-L, and CIDEr use the official RSICC scorer files. METEOR is blocked because the host has no Java runtime and `sudo` requires a password.

| Model | Precision | Samples | Failures | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | CIDEr | METEOR | Peak VRAM MB | Mean latency s | P95 latency s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| LoRA v1 | INT4 NF4 bitsandbytes | 1929 | 0 | 0.324558 | 0.246340 | 0.187795 | 0.144016 | 0.383925 | 0.624115 | unavailable: missing Java | 1635.11 | 0.859499 | 2.390440 |

Change/no-change proxy from the official no-change phrase list:

| Group | Samples | Accuracy |
| --- | ---: | ---: |
| Change | 964 | 0.562241 |
| No change | 965 | 0.451813 |

## Current Interpretation

- VRSBench remains the strongest validated area for the LoRA v1 main model, with INT4 NF4 currently preferred for deployment because it reduces VRAM while preserving or improving the 90-sample internal metrics.
- MME-RealWorld-RS Lite now has a true official Lite RS-subset result. Count questions are the main weakness.
- XLRS-Bench has a real INT4 partial run over the locally available 44 shards. Full Lite completion should resume once HuggingFace access is stable.
- LEVIR-CC has a true official test split run for BLEU/ROUGE-L/CIDEr. METEOR needs Java before a complete official caption metric set can be claimed.
