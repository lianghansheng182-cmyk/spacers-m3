# Prompt v1 vs v2 A/B on 90-Sample Internal Validation

## Scope

This is a project-internal 90-sample validation set, not an official VRSBench test result.

- Dataset: `datasets/vrsbench_eval_v1/resolved_manifest.jsonl`
- Prompt v1 output: `results/vrsbench_eval_v1/qwen3_vl_2b_prompt_v1/`
- Prompt v2 output: `results/vrsbench_eval_v1/qwen3_vl_2b_prompt_v2/`
- Same model/config for both runs: Qwen3-VL-2B-Instruct, BF16, batch size 1, `max_pixels=1048576`, `max_new_tokens=256`
- Same evaluation code: `src.evaluation.evaluate_mini_eval.evaluate_files`
- Caption predictions and GT are preserved; no unreliable automatic caption score is reported.

## Overall Metrics

| Metric | Prompt v1 | Prompt v2 | Absolute Change |
| --- | ---: | ---: | ---: |
| Samples | 90 | 90 | 0 |
| Success | 90 | 90 | 0 |
| Failure | 0 | 0 | 0 |
| Mean latency (s) | 3.492092 | 1.899090 | -1.593002 |
| P95 latency (s) | 10.716380 | 5.551861 | -5.164519 |
| Peak VRAM (MB) | 4184.17 | 4184.17 | 0.00 |
| VQA accuracy | 0.600000 | 0.533333 | -0.066667 |
| VQA correct | 18/30 | 16/30 | -2 |
| Grounding mean IoU | 0.436447 | 0.374667 | -0.061780 |
| Grounding Acc@0.5 | 0.466667 | 0.400000 | -0.066667 |
| Grounding Acc@0.7 | 0.300000 | 0.266667 | -0.033333 |

## Does the 30-Sample Gain Hold?

No. On the 30-sample mini_eval, Prompt v2 improved VQA accuracy, Grounding mean IoU, Acc@0.5, and Acc@0.7. On this larger internal 90-sample validation set, the direction reverses:

- VQA drops from 0.600000 to 0.533333.
- Grounding mean IoU drops from 0.436447 to 0.374667.
- Grounding Acc@0.5 drops from 0.466667 to 0.400000.
- Grounding Acc@0.7 drops from 0.300000 to 0.266667.

Prompt v2 does reduce latency substantially, mainly because its Caption outputs are shorter, but the task metrics do not generalize.

## Correctness Transitions

Correctness is defined as normalized exact match for VQA and Acc@0.5 for Grounding. Caption is excluded from this automatic transition analysis.

### Prompt v1 Correct, v2 Wrong

6 samples:

| ID | Task | Note |
| --- | --- | --- |
| `vqa_ext_012` | VQA | GT `Airport`; v1 `airport`; v2 `airplane` |
| `vqa_ext_026` | VQA | GT `1`; v1 `1`; v2 `Yes` |
| `grounding_ext_006` | Grounding | IoU 0.602042 -> 0.445556 |
| `grounding_ext_015` | Grounding | IoU 0.657425 -> 0.107327 |
| `grounding_ext_020` | Grounding | IoU 0.767596 -> 0.000000 |
| `grounding_ext_029` | Grounding | IoU 0.766462 -> 0.000000 |

### Prompt v1 Wrong, v2 Correct

2 samples:

| ID | Task | Note |
| --- | --- | --- |
| `grounding_ext_008` | Grounding | IoU 0.137174 -> 0.856525 |
| `grounding_ext_016` | Grounding | Acc@0.5 false -> true |

### Both Wrong

26 samples:

`vqa_ext_001`, `vqa_ext_004`, `vqa_ext_005`, `vqa_ext_006`, `vqa_ext_011`, `vqa_ext_015`, `vqa_ext_017`, `vqa_ext_022`, `vqa_ext_023`, `vqa_ext_024`, `vqa_ext_025`, `vqa_ext_028`, `grounding_ext_004`, `grounding_ext_005`, `grounding_ext_007`, `grounding_ext_010`, `grounding_ext_013`, `grounding_ext_017`, `grounding_ext_019`, `grounding_ext_021`, `grounding_ext_023`, `grounding_ext_024`, `grounding_ext_025`, `grounding_ext_026`, `grounding_ext_027`, `grounding_ext_030`.

## Grounding IoU Largest Improvements

| ID | Target | v1 IoU | v2 IoU | Delta |
| --- | --- | ---: | ---: | ---: |
| `grounding_ext_008` | baseball-diamond center | 0.137174 | 0.856525 | +0.719351 |
| `grounding_ext_024` | bridge middle-left | 0.000000 | 0.309598 | +0.309598 |
| `grounding_ext_026` | large vehicle top-left | 0.000000 | 0.140479 | +0.140479 |
| `grounding_ext_010` | expressway-toll-station bottom | 0.000000 | 0.090767 | +0.090767 |
| `grounding_ext_002` | baseball diamond middle-right | 0.664719 | 0.746539 | +0.081820 |
| `grounding_ext_023` | small vehicle top-middle pavement | 0.392835 | 0.437790 | +0.044955 |
| `grounding_ext_009` | airplane top-right | 0.940332 | 0.968760 | +0.028428 |
| `grounding_ext_012` | right-most top-middle vehicle | 0.639175 | 0.663056 | +0.023881 |
| `grounding_ext_011` | storage tank bottom | 0.826504 | 0.843467 | +0.016963 |
| `grounding_ext_003` | small plane top-left tarmac | 0.772595 | 0.782580 | +0.009985 |

## Grounding IoU Largest Declines

| ID | Target | v1 IoU | v2 IoU | Delta |
| --- | --- | ---: | ---: | ---: |
| `grounding_ext_020` | small ship top edge of harbor | 0.767596 | 0.000000 | -0.767596 |
| `grounding_ext_029` | furthest-right harbor bottom edge | 0.766462 | 0.000000 | -0.766462 |
| `grounding_ext_015` | harbor pier middle-to-top | 0.657425 | 0.107327 | -0.550098 |
| `grounding_ext_030` | storage tank above center | 0.402633 | 0.000000 | -0.402633 |
| `grounding_ext_021` | windmill middle-left | 0.291819 | 0.000000 | -0.291819 |
| `grounding_ext_027` | vehicle middle-left | 0.265060 | 0.000000 | -0.265060 |
| `grounding_ext_006` | small train station right side | 0.602042 | 0.445556 | -0.156486 |
| `grounding_ext_028` | airplane top-left | 0.726486 | 0.686601 | -0.039885 |
| `grounding_ext_007` | left-most bottom-middle container crane | 0.306393 | 0.276817 | -0.029576 |
| `grounding_ext_014` | airport lower-right quadrant | 0.991287 | 0.969102 | -0.022185 |

## VQA Error Types

Prompt v1 wrong VQA count: 12.

| Error type | Count |
| --- | ---: |
| Other visual understanding | 5 |
| Category or label granularity | 4 |
| Counting | 2 |
| Yes/no or direction | 1 |

Prompt v2 wrong VQA count: 14.

| Error type | Count |
| --- | ---: |
| Other visual understanding | 6 |
| Category or label granularity | 4 |
| Counting | 3 |
| Yes/no or direction | 1 |

Notable v2 regressions:

- `vqa_ext_012`: setting question regressed from `airport` to object answer `airplane`.
- `vqa_ext_026`: count question regressed from `1` to `Yes`, despite the count-only instruction.

Common persistent VQA failures include label granularity mismatch (`ship` vs `boat`, `Parking area` vs `parking lot`, `suburban` vs `urban`), object attribute confusion (`large` vs `small` airplane), and hard spatial/location answers.

## Caption Length And Qualitative Notes

| Field | GT | Prompt v1 | Prompt v2 |
| --- | ---: | ---: | ---: |
| Mean words | 52.17 | 167.33 | 99.40 |
| Min words | 29 | 68 | 68 |
| Max words | 76 | 224 | 147 |

Prompt v2 captions are shorter than v1 and still retain visible objects and spatial context. This likely explains much of the latency reduction. However, both v1 and v2 are longer than GT on average, and this run intentionally avoids assigning a synthetic caption score.

## Remaining Errors That Are Not Clearly Prompt Problems

- Label-space mismatch: `ship` vs `boat`, `Parking area` vs `parking lot`, `suburban` vs `urban` are exact-match failures that may need label mapping, training, or dataset policy rather than more prompt wording.
- Visual recognition failures: examples include `Greenery` predicted as `forest`, relative airplane size, and vehicle/ship/category confusion.
- Ambiguous or underspecified spatial answers: object ID references and top-left/bottom-left style answers remain brittle.
- Grounding capability limits: several failures are zero-IoU or severe localization misses for ships, harbors, windmills, storage tanks, and small/adjacent vehicles. These look more like model localization limitations than formatting failures.

## Recommendation

Do not freeze Prompt v2 as the default prompt for subsequent validation based on this 90-sample result. The 30-sample improvement does not generalize: v2 is faster, but it is worse on VQA and Grounding metrics.

Prompt v2 performed well on the 30-sample mini_eval, but regressed on the independently expanded 90-sample internal validation set. This indicates overfitting risk to the small development set, so later experiments will fix Prompt v1 as the default evaluation protocol.

The project is ready to start LoRA experimentation from an evaluation-infrastructure standpoint: the 90-sample internal validation set is repaired and validated, A/B outputs are saved, metrics are reproducible, and remaining errors include cases unlikely to be fixed by prompt wording alone. Before starting LoRA, freeze a single evaluation prompt for comparability; based on this A/B, Prompt v1 is the safer default than Prompt v2.
