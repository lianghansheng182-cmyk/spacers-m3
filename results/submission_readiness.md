# Submission Readiness

Created: 2026-08-25

## READY

- Python environment pinning: `requirements.txt` exists and includes Torch CUDA 13.0, Transformers, PEFT, Accelerate, Qwen VL utils, and bitsandbytes.
- Local base model path convention: `models/Qwen3-VL-2B-Instruct`.
- LoRA v1 training and evaluation artifacts are preserved.
- LoRA v1 BF16, INT8, and INT4 NF4 VRSBench 90-sample internal validation results exist.
- Quantized LoRA v1 loading path is implemented in `scripts/lora/evaluate_lora_adapter.py`.
- VRSBench internal evaluation uses one fixed evaluator for BF16/INT8/INT4 comparisons.
- Resource logging includes peak VRAM and latency summaries for existing VRSBench runs.

## PARTIAL

- README exists, but it still mainly documents single-image smoke testing and does not describe the full competition workflow.
- One-click inference exists for single-image use, but the LoRA/precision benchmark entry is script-level rather than a single documented command.
- One-click evaluation exists for VRSBench-style manifests, but not for MME-RealWorld-RS, XLRS-Bench, or LEVIR-CC.
- Model/LoRA adapter loading supports BF16/INT8/INT4 for the current Qwen3-VL path, but this is not exposed through a benchmark-agnostic CLI.
- Benchmark results and technical report material are strong for VRSBench, partial for quantization, and missing for the remaining three benchmarks.

## MISSING

- MME-RealWorld-RS local dataset.
- MME-RealWorld-RS adapter and official evaluator integration.
- XLRS-Bench local dataset.
- XLRS-Bench adapter, high-resolution handling policy, and official evaluator integration.
- LEVIR-CC local dataset.
- LEVIR-CC dual-image Qwen3-VL input adapter.
- LEVIR-CC caption metrics pipeline.
- Unified benchmark runner covering VRSBench, MME-RealWorld-RS, XLRS-Bench, and LEVIR-CC.
- Final submission README with exact reproduction commands for every claimed result.
- Packaging checklist for adapter path, config path, dependency install, INT4 startup, and benchmark data path assumptions.

## Next Highest Priority

Acquire or mount the three remaining official benchmark datasets in stable local paths, then implement benchmark adapters in this order:

1. MME-RealWorld-RS, because it is closest to the current single-image multiple-choice/VQA pipeline.
2. XLRS-Bench lite or official selected split, because it can reuse single-image loading but needs XLRS-specific prompts and possibly ultra-high-resolution handling.
3. LEVIR-CC, because it requires a separate dual-image input path and caption metric stack.
