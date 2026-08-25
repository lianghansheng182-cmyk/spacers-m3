# MME-RealWorld-RS Report

Status: COMPLETE

Result classification: OFFICIAL LITE subset result. This uses MME-RealWorld-Lite and only the official `Remote Sensing` subtask. It is not a full MME-RealWorld leaderboard result.

## Data

- Source: `yifanzhang114/MME-RealWorld-Lite`
- Local raw path: `data/raw/mme_realworld_lite/`
- Archive: `data/raw/mme_realworld_lite/data.zip`
- Question JSON: `data/raw/mme_realworld_lite/extracted/data/MME-RealWorld-Lite.json`
- Total Lite samples: 1919
- Remote Sensing samples: 150
- Remote Sensing task: Perception only
- Remote Sensing categories: position 50, color 50, count 50
- Missing images: 0
- Image resolution range in RS subset: min width/height 921/1062, max width/height 11500/7500

## Protocol

- Prompt follows the official multiple-choice structure:
  - question
  - `The choices are listed below:`
  - `(A)` through `(E)` choices
  - select the best answer and respond only with the letter
- Prediction file preserves official fields and adds `Output`.
- Official evaluator used: `data/raw/mme_realworld_official_eval/eval_your_results.py`
- Official eval stdout: `results/mme_realworld_rs/official_eval_stdout.txt`

## Model

- Base: Qwen3-VL-2B-Instruct
- Adapter: LoRA v1
- Precision: INT4 NF4 bitsandbytes
- Max new tokens: 8
- Max pixels: 1048576
- BF16 fallback: not needed

## Smoke

- Output: `results/mme_realworld_rs/smoke_int4/`
- Samples: 5
- Accuracy: 1.000000
- Failures: 0
- Official evaluator returncode: 0
- Peak VRAM: 1848.66 MB

## Full RS Lite Result

- Output: `results/mme_realworld_rs/`
- Samples: 150
- Correct: 70
- Accuracy: 0.466667
- Parse failures: 0
- Inference failures: 0
- Official evaluator returncode: 0
- Model load time: 4.678999 s
- Mean latency: 0.569578 s
- P95 latency: 0.693621 s
- Peak VRAM: 1850.04 MB

Official category accuracy:

| Category | Samples | Accuracy |
| --- | ---: | ---: |
| Position | 50 | 0.6200 |
| Color | 50 | 0.5800 |
| Count | 50 | 0.2000 |
| Overall Remote Sensing | 150 | 0.4667 |

## Notes

- Count questions are the main weak point in this Lite RS run.
- This result should be reported as MME-RealWorld-Lite Remote Sensing only, not as the complete MME-RealWorld benchmark.
