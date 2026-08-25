# XLRS-Bench Report

Status: PARTIAL REAL RESULT

Result classification: INTERNAL PARTIAL result on locally available XLRS-Bench-lite shards. This is not a complete XLRS-Bench-lite score and not a full XLRS-Bench leaderboard result.

## Data

- Source: `initiacms/XLRS-Bench-lite`
- Local raw path: `data/raw/xlrs_bench_lite/`
- Official Lite split: `train`
- Official Lite size from dataset metadata: 3080 samples, 74 Arrow shards
- Locally available after interrupted HuggingFace download: 1848 samples, 44 / 74 Arrow shards
- Full Lite blocker: HuggingFace network connection timed out / became unreachable before the remaining shards could be downloaded
- Data fields: `path`, `index`, `question`, `multi-choice options`, `answer`, `category`, `l2-category`, `image`
- Observed resolution range in the partial run: min width/height 1361/1196, max width/height 11500/10000

Covered categories in the 44-shard partial run:

| Category | Samples |
| --- | ---: |
| Complex reasoning / Anomaly Detection and Interpretation | 100 |
| Complex reasoning / Environmental condition reasoning | 100 |
| Counting / Counting with complex reasoning | 100 |
| Counting / Overall counting | 60 |
| Land use classification / Regional Land use classification | 200 |
| Object properties / Object color | 728 |
| Object properties / Object motion state | 60 |
| Object spatial relationship / Object spatial relationship | 500 |

## Protocol

- Task type evaluated: XLRS-Bench-lite VQA / multiple choice.
- Prompt uses image, official question text, and official multiple-choice options.
- Standard questions require one letter from `A/B/C/D`.
- Overall Land Use Classification is handled as a possible multi-answer task, but the local 44-shard partial run contained 0 multi-answer samples.
- Metrics are computed with the XLRS-compatible local adapter over the official labels in the downloaded Arrow shards. `lmms-eval` was not installed into the stable environment.
- Processor max pixels: 1048576. Original image resolution and processor image grid metadata are recorded per prediction.

## Model

- Base: Qwen3-VL-2B-Instruct
- Adapter: LoRA v1
- Precision: INT4 NF4 bitsandbytes
- Max new tokens: 16
- BF16 fallback: not needed for the downloaded partial run

## Smoke

- Output: `results/xlrs_bench/smoke_int4/`
- Samples: 5
- Accuracy: 0.400000
- Parse failures: 0
- Inference failures: 0
- Model load time: 5.447755 s
- Mean latency: 0.998168 s
- P95 latency: 2.158694 s
- Peak VRAM: 1859.32 MB
- Observed original resolution range: 4096 x 4096 to 10000 x 10000

## Partial 44-Shard Result

- Output: `results/xlrs_bench/partial_44shards_int4/`
- Samples: 1848
- Correct: 719
- Micro accuracy: 0.389069
- Macro category accuracy: 0.478757
- Parse failures: 0
- Inference failures: 0
- Multi-answer samples: 0
- Model load time: 6.921170 s
- Mean latency: 0.555370 s
- P95 latency: 0.681128 s
- Peak VRAM: 1901.08 MB

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

## Notes

- This run verifies that the frozen Qwen3-VL-2B + LoRA v1 + INT4 NF4 entry can load XLRS embedded Arrow images, process very high resolution remote-sensing imagery through the configured processor resize path, generate parseable multiple-choice predictions, and record latency/VRAM.
- Full XLRS-Bench-lite remains blocked only by incomplete data availability in this environment. The current partial result must not be reported as an official Lite score.
- The weakest categories in the partial run are color, spatial relationship, and counting; no tuning was performed.
