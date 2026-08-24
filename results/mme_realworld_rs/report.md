# MME-RealWorld-RS Report

Status: NOT RUN

Reason: local MME-RealWorld-RS data was not found under `/home/lhs/spacers_m3`, `/home/lhs`, or `/mnt/g`.

Planned model entry:

- Model: Qwen3-VL-2B-Instruct
- Adapter: LoRA v1
- First precision: INT4 NF4
- Fallback precision: BF16 if INT4 has compatibility or output-format failures

Protocol notes:

- Official task is multiple-choice VQA.
- Official prompt uses image + question + A-E options and asks for only the answer letter.
- Official scoring is accuracy across categories/subtasks and task types.

Current support gap:

- No local manifest/images.
- No RS-domain extraction if using the full MME-RealWorld package.
- No adapter from official JSON to the project prediction format.
- No official result JSON writer or `eval_your_results.py` integration.

Result classification: missing-data blocker, not smoke, not internal validation, not official protocol result.
