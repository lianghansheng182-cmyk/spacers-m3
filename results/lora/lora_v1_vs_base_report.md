# LoRA v1 vs Base Report

Generated: 2026-08-23T23:59:03

## Verdict

LoRA v1 training completed and adapter evaluation finished on Prompt v1.

- Dry-run: PASS
- Smoke test: PASS
- Full training: PASS
- Mini eval inference: 30/30 success
- VRSBench eval v1 inference: 90/90 success
- Baseline outputs were not modified.

## Training Artifact

| Item | Value |
|---|---|
| Base model | `models/Qwen3-VL-2B-Instruct` |
| Adapter | `results/lora/qwen3_vl_2b_lora_v1/adapter_final` |
| Adapter size | 33.311 MB |
| Git commit recorded at training start | `916f1197fa482d6c6cb5bdb9a3574923e63e4426` |
| Prompt | v1 |

## Dry-Run

| Metric | Value |
|---|---:|
| Status | PASS |
| Data samples collated | 10 |
| Total parameters | 2136248320 |
| Trainable parameters | 8716288 |
| Trainable ratio | 0.408% |
| Dry-run report | `results/lora/lora_dry_run_report.md` |

## Smoke Test

| Metric | Value |
|---|---:|
| Optimizer steps | 50 |
| Micro steps | 400 |
| Loss first window | 3.256994 |
| Loss last window | 0.829499 |
| NaN loss | False |
| OOM count | 0 |
| Peak VRAM | 4.978 GiB |
| Checkpoints | 2 |

## Full Training

| Metric | Value |
|---|---:|
| Optimizer steps | 1125 |
| Micro steps | 9000 |
| Loss first window | 3.258497 |
| Loss last window | 0.849381 |
| Min loss | 0.005162 |
| Max loss | 7.038500 |
| NaN loss | False |
| OOM count | 0 |
| Elapsed seconds | 4567.74 |
| Peak VRAM | 5.077 GiB |
| Checkpoints | 45 |

## Mini Eval 30

| Metric | Base | LoRA v1 | Delta |
|---|---:|---:|---:|
| VQA accuracy | 0.5 | 0.7 | +0.2 |
| Grounding mean IoU | 0.565188 | 0.617452 | +0.052264 |
| Grounding Acc@0.5 | 0.7 | 0.8 | +0.1 |
| Grounding Acc@0.7 | 0.4 | 0.2 | -0.2 |
| Latency mean seconds | 4.20324 | 1.7851 | -2.41815 |
| Latency P95 seconds | 12.1252 | 4.81884 | -7.30631 |
| Peak VRAM MB | 4184.17 | 4201.32 | +17.15 |

## VRSBench Eval v1 90

| Metric | Base | LoRA v1 | Delta |
|---|---:|---:|---:|
| VQA accuracy | 0.6 | 0.7 | +0.1 |
| Grounding mean IoU | 0.436447 | 0.443984 | +0.007537 |
| Grounding Acc@0.5 | 0.466667 | 0.466667 | +0 |
| Grounding Acc@0.7 | 0.3 | 0.266667 | -0.033333 |
| Latency mean seconds | 3.49209 | 1.76847 | -1.72362 |
| Latency P95 seconds | 10.7164 | 5.11779 | -5.59859 |
| Peak VRAM MB | 4184.17 | 4203.17 | +19 |

## Issues Found

- LoRA improves VQA accuracy on both eval sets.
- Grounding mean IoU improves slightly on both eval sets, but Acc@0.7 drops on both sets.
- One grounding prediction failed bbox parsing on the 90-sample eval.
- Training used a constant learning rate; no scheduler or validation loop was added in this v1 run.
- The training objective masks the prompt and trains only assistant tokens; caption/VQA/grounding are mixed by shuffled sample order.

## Config

```yaml
run:
  name: qwen3_vl_2b_lora_v1
  seed: 20260823
  output_dir: results/lora/qwen3_vl_2b_lora_v1
  smoke_output_dir: results/lora/qwen3_vl_2b_lora_v1_smoke

model:
  name: Qwen3-VL-2B-Instruct
  path: models/Qwen3-VL-2B-Instruct
  dtype: bf16
  local_files_only: true
  gradient_checkpointing: true
  use_cache: false

data:
  train: datasets/vrsbench_lora_v1/train.jsonl
  dev: datasets/vrsbench_lora_v1/train_dev.jsonl
  image_root: datasets/vrsbench_lora_v1
  prompt_version: v1

lora:
  r: 8
  alpha: 16
  dropout: 0.05
  bias: none
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  save_adapter_only: true

training:
  bf16: true
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.0002
  weight_decay: 0.0
  max_grad_norm: 1.0
  num_epochs: 1
  max_steps: 1125
  smoke_max_steps: 50
  log_every: 1
  save_every: 25
  shuffle: true

reporting:
  record_git_commit: true
  record_environment: true
  save_training_log: true
```
