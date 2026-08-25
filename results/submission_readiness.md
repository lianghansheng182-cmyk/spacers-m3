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
- Unified benchmark dispatcher exists at `scripts/benchmarks/run_benchmark.py`.
- README now records the frozen model policy, environment setup, INT4 benchmark commands, result locations, and data policy.
- Technical report draft exists at `docs/competition_technical_report.md`.
- Raw benchmark data is kept under `data/raw/`, which is excluded by `.gitignore`.
- Model weights, adapter checkpoints, and large raw datasets remain excluded by `.gitignore`.
- Result outputs are isolated under benchmark-specific `results/` subdirectories.
- Resource logging includes load time, mean latency, P95 latency, and peak VRAM for the new benchmark runs.

## PARTIAL

- XLRS-Bench-lite full run is not complete because only 44 / 74 Arrow shards were available locally after the HuggingFace download failed. The current 1848-sample result is reproducible from local data but must be reported as internal partial.
- LEVIR-CC metric set is incomplete because METEOR is unavailable without Java. BLEU, ROUGE-L, and CIDEr are complete.
- One-click inference exists for core Qwen3-VL usage; final packaging may still need a shorter top-level shell wrapper.
- One-click benchmark coverage now has a unified Python dispatcher, but there is not yet a top-level shell wrapper that checks all data prerequisites before dispatch.
- BF16 fallback is supported by the shared precision backend, but the new benchmark reports currently record INT4 as the completed path.
- Final technical report may still need formatting for the competition template, but the core narrative and metrics are now collected.

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
3. Add a top-level shell wrapper for the frozen benchmark dispatcher if the submission format expects shell commands instead of Python commands.
4. Format `docs/competition_technical_report.md` into the final competition template if a required template is provided.
