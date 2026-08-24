# XLRS-Bench Report

Status: NOT RUN

Reason: local XLRS-Bench data was not found under `/home/lhs/spacers_m3`, `/home/lhs`, or `/mnt/g`.

Planned model entry:

- Model: Qwen3-VL-2B-Instruct
- Adapter: LoRA v1
- First precision: INT4 NF4
- Fallback precision: BF16 if INT4 has compatibility or output-format failures

Protocol notes:

- XLRS-Bench is an ultra-high-resolution remote-sensing MLLM benchmark.
- Official resources include VQA, visual grounding, and detailed captioning tasks.
- VQA prompt uses image + question + A-D options; one task variant may require multiple letters.
- Official README points to lmms-eval for `xlrs-lite` and task-specific evaluation resources for grounding/captioning.

Current support gap:

- No local dataset.
- No ultra-high-resolution image policy or tiling/cropping adapter.
- No XLRS-specific A-D/multi-answer parser.
- No official grounding/caption evaluator integration.

Result classification: missing-data blocker, not smoke, not internal validation, not official protocol result.
