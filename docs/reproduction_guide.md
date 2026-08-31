# SPACERS-M3 Reproduction Guide

This guide describes the validated cross-machine setup for the final submission branch.

## 1. Recommended Hardware

- NVIDIA GPU with enough VRAM for Qwen3-VL-2B inference.
- Verified device: RTX 5060 Laptop GPU, compute capability sm_120, 8 GB VRAM.
- INT4 NF4 is the default deployment path. BF16 is the fallback path.

## 2. Recommended System

- WSL2 / Linux
- Ubuntu 24.04
- Python 3.12
- CUDA 13.0 runtime

Use WSL2 Ubuntu or Linux for formal reproduction. Native Windows Python is limited to static checks or a small single-image fallback path when users install their own compatible Windows wheels; it is not the validated full inference or benchmark smoke environment.

## 3. Checkout

```bash
git clone -b release/submission-final https://github.com/lianghansheng182-cmyk/spacers-m3.git spacers_m3
cd spacers_m3
```

The `main` branch is not the frozen submission branch.

## 4. Create The Python Environment

Verified Linux/WSL CUDA 13 setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-verified.txt
```

`requirements-verified.txt` and `requirements-linux-cu130.txt` contain Linux/Python 3.12/CUDA 13 wheel pins. Do not install them directly in native Windows Python 3.14.

`requirements.txt` is a generic package list. If you use it, install a platform-compatible PyTorch build first.

## 5. Place The Base Model

Expected path:

```text
models/Qwen3-VL-2B-Instruct/
```

The directory should contain files such as:

```text
config.json
generation_config.json
tokenizer_config.json
preprocessor_config.json
model.safetensors
```

The inference code expects the local model directory. Do not rely on a hidden Hugging Face cache for reproducible runs.

## 6. LoRA Adapter

In the official submission package, the LoRA v1 adapter is already included at:

```text
03_model/qwen3_vl_2b_lora_v1_adapter/
```

When commands are run from `02_source_code/`, `scripts/check_artifacts.py` detects `../03_model/qwen3_vl_2b_lora_v1_adapter/` directly. `scripts/smoke_test.sh` links it into the code default path automatically when needed.

Git-only clone fallback path:

```text
results/lora/qwen3_vl_2b_lora_v1/adapter_final/
```

Expected files:

```text
adapter_config.json
adapter_model.safetensors
README.md
```

Expected adapter size: about 34 MB.

SHA256 reference:

```text
docs/lora_v1_adapter_manifest.sha256
```

## 7. Place Benchmark Data

Tracked with Git:

```text
datasets/mini_eval/
datasets/vrsbench_eval_v1/
```

External raw data to provide separately:

```text
data/raw/mme_realworld_lite/
data/raw/xlrs_bench_lite/
data/raw/levir_cc/
```

Notes:

- MME-RealWorld-RS uses the Remote Sensing subset of MME-RealWorld-Lite.
- XLRS-Bench Lite experiments in this repo are labeled PARTIAL when only 44 / 74 shards are present.
- LEVIR-CC uses the official split structure with paired A/B images.

## 8. Artifact Precheck

Run a fast path check before model loading:

```bash
python scripts/check_artifacts.py
```

Expected output includes:

```text
[PASS] mini_eval manifest
[PASS] VRSBench 90 manifest
[PASS] base model directory
[PASS] Detected LoRA v1 adapter directory
```

For static source-only checks without public raw data:

```bash
python scripts/check_artifacts.py --skip-raw
```

If a required artifact is missing, the script prints the exact expected path.

## 9. Environment Check

```bash
python scripts/check_environment.py
```

The script reports:

- OS
- Python version
- torch version
- CUDA runtime
- GPU name
- GPU compute capability
- transformers
- peft
- bitsandbytes
- PyYAML

Warnings are diagnostic. They do not hide the original failure.

## 10. INT4 Smoke

For the official submission package, run the complete smoke sequence from the source-code directory:

```bash
cd 02_source_code
bash scripts/setup_linux.sh
python scripts/check_artifacts.py
bash scripts/smoke_test.sh
```

Expected output:

- environment profile saved to `logs/environment_info.txt`
- LoRA v1 adapter detected either at `results/lora/qwen3_vl_2b_lora_v1/adapter_final/` or at packaged path `../03_model/qwen3_vl_2b_lora_v1_adapter/`
- INT4 and BF16 VRSBench smoke each report `success_count: 1` and `failure_count: 0`
- SpaceBudget AUTO smoke reports one MME-RS prediction and official evaluator metadata
- HiRes grid_2x2 smoke reports one XLRS prediction and visual-token statistics
- OrbitGuard integrity smoke writes `orbitguard_summary.json`

The latest WSL2 Ubuntu validation logs are included in the submission package under `05_reproduction/logs/`.

Optional single-image smoke through the Linux/WSL bash wrapper:

```bash
bash scripts/run_single.sh
```

If bash is unavailable, run the same Python entry directly:

```bash
python src/inference/run_single.py \
  --model-path models/Qwen3-VL-2B-Instruct \
  --image datasets/mini_eval/images/00002_0000.png \
  --question "Describe this remote sensing image." \
  --output results/smoke_test.json \
  --max-pixels 1048576
```

This single-image smoke validates local Qwen3-VL model loading and image inference. It is not the LoRA adapter validation path.

```bash
python scripts/benchmarks/run_benchmark.py vrsbench \
  --precision int4 \
  --limit 1 \
  --output-dir results/smoke/vrsbench_int4
```

Expected output: one prediction, `failure_count: 0`, and files under `results/smoke/vrsbench_int4/`.

## 11. BF16 Smoke

```bash
python scripts/benchmarks/run_benchmark.py vrsbench \
  --precision bf16 \
  --limit 1 \
  --output-dir results/smoke/vrsbench_bf16
```

Expected output: one prediction, `failure_count: 0`, and files under `results/smoke/vrsbench_bf16/`.

## 12. SpaceBudget AUTO

```bash
python scripts/benchmarks/run_benchmark.py mme-rs-lite \
  --precision int4 \
  --resource-mode auto \
  --limit 1 \
  --output-dir results/smoke/mme_spacebudget_auto
```

Expected output: one MME-RS prediction, resource-mode metadata, visual token statistics, and summary files.

## 13. HiRes-View

```bash
python scripts/benchmarks/run_benchmark.py xlrs-lite \
  --precision int4 \
  --tiling-mode grid_2x2 \
  --limit 1 \
  --output-dir results/smoke/xlrs_highres_grid_2x2
```

Expected output: one XLRS prediction, tiling metadata, visual token statistics, and summary files.

## 14. OrbitGuard-RS

```bash
python scripts/reliability/run_fault_injection.py \
  --config configs/reliability/orbitguard_v1.yaml \
  --max-fault-runs 1 \
  --output-dir results/smoke/orbitguard_integrity
```

Expected output:

```text
clean integrity PASS
corruption DETECTED
recovery PASS
```

## 15. Full Benchmark Commands

VRSBench internal validation:

```bash
python scripts/benchmarks/run_benchmark.py vrsbench --precision int4
```

MME-RealWorld-RS Lite:

```bash
python scripts/benchmarks/run_benchmark.py mme-rs-lite --precision int4
```

XLRS-Bench Lite:

```bash
python scripts/benchmarks/run_benchmark.py xlrs-lite --precision int4
```

LEVIR-CC:

```bash
python scripts/benchmarks/run_benchmark.py levir-cc --precision int4
```

## 16. Common Failures

### A. Windows Native Python 3.14 Installs Linux Wheels

Symptom:

```text
No matching distribution found
```

or package installation fails on the pinned `torchvision` Linux wheel URL.

Correct action:

- Use WSL2 Ubuntu with Python 3.12 for the verified path.
- Do not install `requirements-verified.txt` in native Windows Python 3.14.
- If using native Windows, install a Windows-compatible PyTorch build first and treat the setup as unverified. Native Windows is not the formal full-smoke validation environment for this submission.

### B. PyTorch Build Does Not Support sm_120

Symptom:

```text
CUDA error
no kernel image is available for execution on the device
```

Correct action:

- Run `python scripts/check_environment.py`.
- Check GPU compute capability and `torch.version.cuda`.
- For RTX 50-series sm_120 GPUs, use the verified Linux/WSL CUDA 13 environment used by this submission.
- Do not downgrade or upgrade torch blindly during final reproduction.

### C. Missing `peft` Or `yaml`

Symptom:

```text
ModuleNotFoundError: No module named 'peft'
ModuleNotFoundError: No module named 'yaml'
```

Correct action:

```bash
source .venv/bin/activate
python -m pip install -r requirements-verified.txt
python scripts/check_environment.py
```

### D. Missing Base Model Or Adapter

Symptom:

```text
FileNotFoundError
HFValidationError
adapter_config.json not found
```

Correct action:

```bash
python scripts/check_artifacts.py --skip-raw
```

Then place the missing artifact at the path printed by the checker.
