# Prompt v1 vs v2 Ablation

## Setup

- Dataset: `datasets/mini_eval/mini_eval.jsonl`, 30 samples.
- Baseline: `results/vrsbench_mini/qwen3_vl_2b_zero_shot/`
- Prompt v2: `results/vrsbench_mini/qwen3_vl_2b_prompt_v2/`
- Model/config unchanged: Qwen3-VL-2B-Instruct, BF16, batch size 1, `max_pixels=1048576`, `max_new_tokens=256`.
- Evaluation logic unchanged: `src.evaluation.evaluate_mini_eval.evaluate_files`.

## Prompt Differences

### Caption

- v1: uses the original question directly.
- v2: asks for 2-4 factual remote-sensing sentences, prioritizing visible main targets, approximate counts when clear, spatial relationships, and scene type.
- v2 adds a constraint not to invent source, resolution, unseen details, or unsupported fine categories.

### VQA

- v1: appends `Answer with a concise short answer.`
- v2: answer only from the image, shortest dataset-style answer, no explanation.
- v2 adds task-specific output rules: yes/no must be `Yes` or `No`; counting must be only a number.
- v2 asks object/category answers to use clear remote-sensing target or scene labels rather than color/material-only descriptions.
- v2 defines position/direction conventions: object center in image thirds; north up, east right, west left, north top, south bottom.

### Grounding

- v1: asks for one bbox in normalized 0-100 xyxy format `{<x1><y1><x2><y2>}`.
- v2: uses the explicit `target` description, asks the model to select exactly one object using attributes, position words, and relative relationships.
- v2 emphasizes tight boxes around only the target, excluding adjacent objects and large background.
- v2 outputs only one JSON array `[x1, y1, x2, y2]`; the coordinate protocol and evaluator remain unchanged.

## Metrics

| Metric | Baseline v1 | Prompt v2 | Absolute Change |
| --- | ---: | ---: | ---: |
| Samples | 30 | 30 | 0 |
| Success | 30 | 30 | 0 |
| Failure | 0 | 0 | 0 |
| VQA accuracy | 0.500000 | 0.800000 | +0.300000 |
| VQA correct | 5/10 | 8/10 | +3 |
| Grounding mean IoU | 0.565188 | 0.668165 | +0.102977 |
| Grounding Acc@0.5 | 0.700000 | 0.800000 | +0.100000 |
| Grounding Acc@0.7 | 0.400000 | 0.700000 | +0.300000 |
| Mean latency (s) | 4.203245 | 2.357201 | -1.846044 |
| P95 latency (s) | 12.125152 | 7.019293 | -5.105859 |
| Peak VRAM (MB) | 4184.17 | 4184.17 | 0.00 |

Caption has no automatic metric in this evaluation.

## Original VQA Errors

| ID | GT | v1 Prediction | v2 Prediction | Fixed? | Note |
| --- | --- | --- | --- | --- | --- |
| `vqa_001` | `expressway-toll-station` | `blue structure` | `expressway-toll-station` | Yes | Category prompt fixed color/appearance-only answer. |
| `vqa_005` | `urban` | `residential` | `urban` | Yes | Broad scene-label instruction fixed granularity. |
| `vqa_008` | `middle` | `bottom` | `bottom` | No | Ambiguous vertical-position judgment remains unresolved. |
| `vqa_009` | `Yes` | `no` | `No` | No | Direction/eastward reasoning still failed. |
| `vqa_010` | `Chimney` | `cooling tower` | `Chimney` | Yes | Dataset-style category instruction fixed label choice. |

## Original Grounding Low-IoU Cases

| ID | v1 IoU | v2 IoU | Improved? | Acc@0.5 Change | Note |
| --- | ---: | ---: | --- | --- | --- |
| `grounding_002` | 0.007276 | 0.876923 | Yes | false -> true | Bottom-left vehicle disambiguation improved strongly. |
| `grounding_009` | 0.401145 | 0.250444 | No | false -> false | Left-edge large vehicle box shifted/shortened; prompt did not solve vertical extent. |
| `grounding_010` | 0.319398 | 0.308670 | No | false -> false | Right-side large vehicle remains too wide and includes adjacent region. |

## New Errors

- VQA: no new exact-match errors among samples that were correct in v1.
- Grounding: no new Acc@0.5 failures among samples that passed in v1.
- New error count under the same criteria: 0.

Prompt v2 error cases are `vqa_008`, `vqa_009`, `grounding_009`, and `grounding_010`.

## Remaining Prompt Limits

- Fine spatial bins such as top/middle/bottom remain ambiguous when the object spans multiple thirds or sits near a boundary.
- Directional reasoning still failed for the eastward tracks case, despite explicitly binding east to image-right.
- Tight single-object vehicle boxes remain difficult when long vehicles are adjacent to similar objects or when the expected box includes more vertical extent than the model selects.
- Caption quality is not automatically measured here, so any caption improvement/regression needs manual review or a caption metric.

## Recommendation

Use Prompt v2 as the default prompt for the next 90-sample validation run, with caution. On this fair 30-sample mini_eval, it improves VQA accuracy by +0.30, Grounding mean IoU by +0.102977, Acc@0.5 by +0.10, and Acc@0.7 by +0.30, with no new VQA exact-match or Grounding Acc@0.5 regressions. The remaining weak spots should be tracked explicitly in the 90-sample run, especially direction questions and tight boxes for adjacent vehicles.
