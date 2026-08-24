# LoRA Model Selection Report

Generated: 2026-08-24T21:51:40

## Decision

LoRA v1 is the current main model. Base remains the original zero-shot baseline. LoRA v2 is retained as a grounding-enhancement ablation, not promoted.

## Mini Eval 30

| Metric | Base | LoRA v1 | LoRA v2 |
|---|---:|---:|---:|
| VQA accuracy | 0.5 | 0.7 | 0.7 |
| Grounding mean IoU | 0.565188 | 0.617452 | 0.639133 |
| Grounding Acc@0.5 | 0.7 | 0.8 | 0.8 |
| Grounding Acc@0.7 | 0.4 | 0.2 | 0.4 |
| Grounding parse failures | 0 | 0 | 0 |
| Mean latency s | 4.203245 | 1.785099 | 1.323589 |
| P95 latency s | 12.125152 | 4.818842 | 3.401439 |
| Peak VRAM MB | 4184.17 | 4201.32 | 4201.32 |

## VRSBench Eval v1 90

| Metric | Base | LoRA v1 | LoRA v2 |
|---|---:|---:|---:|
| VQA accuracy | 0.6 | 0.7 | 0.733333 |
| Grounding mean IoU | 0.436447 | 0.443984 | 0.382554 |
| Grounding Acc@0.5 | 0.466667 | 0.466667 | 0.4 |
| Grounding Acc@0.7 | 0.3 | 0.266667 | 0.266667 |
| Grounding parse failures | 0 | 1 | 0 |
| Mean latency s | 3.492092 | 1.768474 | 1.518055 |
| P95 latency s | 10.71638 | 5.11779 | 4.226751 |
| Peak VRAM MB | 4184.17 | 4203.17 | 4203.17 |

## Why v1 Is Main

- v1 gives the best 90-sample grounding mean IoU among the LoRA adapters: 0.443984.
- v1 keeps VQA materially above Base: 0.60 -> 0.70 on the 90-sample eval.
- v1 is the better balanced adapter for the current VQA + Grounding objective.

## Why v2 Is Not Main

- v2 improves 90-sample VQA to 0.733333 and removes the v1 grounding parse failure.
- v2 regresses 90-sample Grounding mean IoU from 0.443984 to 0.382554.
- v2 regresses 90-sample Grounding Acc@0.5 from 0.466667 to 0.40.
- The regression is on localization quality, not parser compatibility.

## v2 Positive Value

- Grounding parse failures drop from 1 to 0 on the 90-sample eval.
- VQA is maintained and improves from 0.70 to 0.733333 on the 90-sample eval.
- The ablation confirms Prompt-v1-style grounding format augmentation can improve output-format reliability.

## Current Conclusion

Simply increasing Grounding samples and task weight does not stably improve precise localization. The next grounding work should target object selection and bbox boundary precision, not just more Grounding sampling.
