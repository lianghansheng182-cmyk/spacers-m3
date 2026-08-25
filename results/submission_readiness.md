# Submission Readiness

Created: 2026-08-25

## READY

- Frozen model policy: Base Qwen3-VL-2B-Instruct, main model LoRA v1, deployment candidate LoRA v1 INT4 NF4.
- Python dependency file: `requirements.txt` exists and includes Torch CUDA 13.0, Transformers, PEFT, Accelerate, Qwen VL utils, bitsandbytes, and pyarrow.
- Base model path convention: `models/Qwen3-VL-2B-Instruct`.
- LoRA v1 training, smoke, dry-run, model selection, and quantization artifacts are preserved.
- VRSBench internal validation results exist for Base BF16, LoRA v1 BF16, LoRA v1 INT8, and LoRA v1 INT4 NF4.
- MME-RealWorld-Lite Remote Sensing subset has a true INT4 result with official evaluator stdout.
- XLRS-Bench-lite has a true INT4 smoke run and a 44-shard partial run over all locally available Arrow shards.
- LEVIR-CC has a true INT4 official test split run for BLEU, ROUGE-L, and CIDEr using official RSICC scorer code.
- Benchmark adapters now exist under `scripts/benchmarks/` for MME-RealWorld-RS Lite, XLRS-Bench Lite, and LEVIR-CC.
- Raw benchmark data is kept under `data/raw/`, which is excluded by `.gitignore`.
- Model weights, adapter checkpoints, and large raw datasets remain excluded by `.gitignore`.
- Result outputs are isolated under benchmark-specific `results/` subdirectories.
- Resource logging includes load time, mean latency, P95 latency, and peak VRAM for the new benchmark runs.

## PARTIAL

- XLRS-Bench-lite full run is not complete because only 44 / 74 Arrow shards were available locally after the HuggingFace download failed. The current 1848-sample result is reproducible from local data but must be reported as internal partial.
- LEVIR-CC metric set is incomplete because METEOR is unavailable without Java. BLEU, ROUGE-L, and CIDEr are complete.
- One-click inference exists for core Qwen3-VL usage, but the new benchmark adapters are script-level commands rather than a single unified launcher.
- One-click benchmark coverage is split across separate scripts instead of a benchmark-agnostic CLI.
- README still needs final competition-oriented reproduction commands for VRSBench, MME-RealWorld-RS Lite, XLRS-Bench Lite, and LEVIR-CC.
- BF16 fallback is supported by the shared precision backend, but the new benchmark reports currently record INT4 as the completed path.
- Final technical report still needs narrative integration of the new benchmark outcomes.

## BLOCKED

- Complete XLRS-Bench-lite official Lite score: blocked by incomplete dataset download from HuggingFace in the current environment.
- LEVIR-CC METEOR: blocked by missing Java runtime; `sudo -n true` fails because a password is required.
- Full MME-RealWorld leaderboard result: intentionally not run; current result is the official Lite Remote Sensing subset only.
- Full XLRS-Bench leaderboard result: intentionally not run; current priority is Lite/partial reproducible coverage.
- Any additional LoRA or prompt tuning: blocked by project freeze policy, not a technical task.

## Current Benchmark Status

| Benchmark | Status | Result type | Output |
| --- | --- | --- | --- |
| VRSBench | ready | internal validation | `results/final_benchmark_summary.md` |
| MME-RealWorld-RS Lite | ready | official Lite RS subset | `results/mme_realworld_rs/` |
| XLRS-Bench Lite | partial | internal partial, 44 / 74 shards | `results/xlrs_bench/partial_44shards_int4/` |
| LEVIR-CC | ready with metric caveat | official test split, METEOR blocked | `results/levir_cc/` |

## Next Highest Priority

1. Resume XLRS-Bench-lite download when HuggingFace network access is stable, then rerun the full 3080-sample Lite split with the existing adapter.
2. Install or provide a Java runtime without disrupting the Python environment, then rerun LEVIR-CC metric aggregation to add METEOR.
3. Write final README commands for the frozen model, INT4 loading, and every benchmark result claimed in `results/final_benchmark_summary.md`.
4. Add a thin unified benchmark launcher that dispatches to the existing benchmark-specific scripts without changing model behavior.
