# SPACERS-M3 Technical Report Draft

Date: 2026-08-25

## Model Decision

The final main model is Qwen3-VL-2B-Instruct with the LoRA v1 adapter.

- Base model: Qwen3-VL-2B-Instruct
- Main model: LoRA v1
- Deployment candidate: LoRA v1 INT4 NF4
- Effect reference: LoRA v1 BF16
- LoRA v2: retained as an ablation experiment only

LoRA v1 is kept as the main model because it improved VQA and preserved the best overall Grounding behavior among the stable LoRA experiments. LoRA v2 improved VQA and removed the single Grounding parse failure, but its 90-sample Grounding localization metrics regressed.

## Training Data

The VRSBench LoRA training data was rebuilt from official train data with image-level exclusion against the evaluation sets.

- Final dataset: `datasets/vrsbench_lora_v1/`
- Train samples: 9000
- Train-dev samples: 600
- Task balance: Caption 3000, VQA 3000, Grounding 3000 in train
- Image paths are relative and images are materialized only for selected samples
- Evaluation images are blacklisted at image level
- Leakage checks use basename and SHA256 hash

Validator and audit outputs:

- `results/data_audit/vrsbench_official_train_structure.md`
- `results/data_audit/lora_final_audit.md`
- `results/data_audit/lora_training_readiness.md`
- `scripts/data/validate_vrsbench_lora_v1.py`

## LoRA Experiments

### LoRA v1

LoRA v1 is the current main model.

VRSBench 90-sample internal validation:

| Metric | Base BF16 | LoRA v1 BF16 |
| --- | ---: | ---: |
| VQA accuracy | 0.600000 | 0.700000 |
| Grounding mean IoU | 0.436447 | 0.443984 |
| Grounding Acc@0.5 | 0.466667 | 0.466667 |
| Grounding Acc@0.7 | 0.300000 | 0.266667 |
| Parse failures | 0 | 1 |

The Grounding SFT audit confirmed that the training, Prompt v1 inference, and evaluator bbox protocols are basically aligned. The main Grounding errors are localization precision problems rather than an obvious train/inference protocol mismatch.

### LoRA v2

LoRA v2 is retained as a Grounding-enhancement ablation.

VRSBench 90-sample internal validation:

- VQA improved to 0.733333
- Grounding parse failures decreased from 1 to 0
- Grounding mean IoU decreased to 0.382554
- Grounding Acc@0.5 decreased to 0.400000

Conclusion: simply increasing Grounding samples and Grounding task weight did not reliably improve precise localization.

## Quantization

bitsandbytes 0.50.1 was installed and validated without upgrading Torch, Transformers, PEFT, or Accelerate. INT8 and INT4 were both tested with the LoRA v1 adapter.

VRSBench 90-sample internal validation:

| Model | Precision | VQA Acc | Grounding mean IoU | Acc@0.5 | Acc@0.7 | Parse failures | Peak VRAM MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA v1 | BF16 | 0.700000 | 0.443984 | 0.466667 | 0.266667 | 1 | 4203.17 |
| LoRA v1 | INT8 | 0.733333 | 0.477298 | 0.466667 | 0.400000 | 0 | 2545.99 |
| LoRA v1 | INT4 NF4 | 0.733333 | 0.509680 | 0.600000 | 0.466667 | 1 | 1659.81 |

INT4 NF4 is selected as the current deployment candidate because it has the lowest measured VRAM and no obvious accuracy loss in the internal 90-sample comparison.

## Benchmark Results

The consolidated benchmark table is in `results/final_benchmark_summary.md`.

### VRSBench

Protocol type: internal validation.

Best deployment candidate result:

- LoRA v1 INT4 NF4
- VQA accuracy: 0.733333
- Grounding mean IoU: 0.509680
- Grounding Acc@0.5: 0.600000
- Grounding Acc@0.7: 0.466667
- Parse failures: 1
- Peak VRAM: 1659.81 MB

### MME-RealWorld-RS Lite

Protocol type: official Lite Remote Sensing subset.

- Data: MME-RealWorld-Lite Remote Sensing subset
- Samples: 150
- Model: LoRA v1 INT4 NF4
- Accuracy: 0.466667
- Parse failures: 0
- Peak VRAM: 1850.04 MB
- Official evaluator returncode: 0

Category accuracy:

- Position: 0.6200
- Color: 0.5800
- Count: 0.2000

This is not a full MME-RealWorld leaderboard result.

### XLRS-Bench Lite

Protocol type: internal partial.

- Official Lite metadata: 3080 samples, 74 Arrow shards
- Locally available data: 1848 samples, 44 / 74 shards
- Model: LoRA v1 INT4 NF4
- Micro accuracy: 0.389069
- Macro category accuracy: 0.478757
- Parse failures: 0
- Peak VRAM: 1901.08 MB

The complete XLRS-Bench-lite result is blocked by incomplete HuggingFace data download. The current result must be reported as a partial local result only.

### LEVIR-CC

Protocol type: official test split with METEOR blocked.

- Data: LEVIR-CC official test split
- Samples: 1929
- Model: LoRA v1 INT4 NF4
- Failures: 0
- BLEU-1: 0.324558
- BLEU-2: 0.246340
- BLEU-3: 0.187795
- BLEU-4: 0.144016
- ROUGE-L: 0.383925
- CIDEr: 0.624115
- METEOR: unavailable because Java is missing

BLEU, ROUGE-L, and CIDEr use the official RSICC scorer files. METEOR requires a Java runtime that is not installed on the current host.

## High-Resolution Multi-View Tiling Ablation

The competition innovation experiment tested a resource-aware multi-scale inference mechanism for ultra-high-resolution remote-sensing scenes.

Method:

- Single global overview of the original image
- Four local 2x2 crops: top-left, top-right, bottom-left, bottom-right
- Fixed input order and a wrapper prompt explaining that all views are from the same scene
- Same Qwen3-VL-2B-Instruct base model, LoRA v1 adapter, INT4 NF4 precision, labels, and evaluators

Input audit:

- XLRS local 44-shard partial contains 1848 samples; 1837 are >16MP.
- XLRS median image size is 100MP, commonly compressed to `image_grid_thw=[[1, 64, 64]]` in single-view inference.
- MME-RS contains 150 samples; 135 are >16MP and the largest image is 11500x7500.

A/B result:

| Benchmark | Single acc | Global+2x2 acc | Delta | Mean latency multiplier | Peak VRAM increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| XLRS local 44-shard partial | 0.389069 | 0.405303 | +0.016234 | 4.88x | +1019.78 MB |
| MME-RS Lite | 0.466667 | 0.473333 | +0.006666 | 4.54x | +1019.14 MB |

Category-level observations:

- XLRS improved most on Object motion state (+0.066667) and Regional land use classification (+0.065000).
- XLRS regressed on Environmental condition reasoning (-0.050000), Anomaly Detection and Interpretation (-0.030000), and Counting with complex reasoning (-0.020000).
- MME-RS improved on color (+0.040000) and count (+0.020000), but regressed on position (-0.040000).

Decision:

- The mechanism is technically feasible on the RTX 5060 Laptop GPU 8GB with INT4 NF4.
- The accuracy gains are small compared with the latency cost.
- A simple megapixel-threshold adaptive route is not justified by this ablation because the benefit is category-dependent and not cleanly explained by resolution alone.
- The module should be kept as a reproducible innovation ablation and optional high-resolution fallback, not as the default deployment path.

Detailed reports:

- `results/highres/highres_input_audit.md`
- `results/highres/highres_tiling_ablation.md`

## OrbitGuard-RS

OrbitGuard-RS is the reliability module for resource-aware onboard inference:

Resource-Aware Soft-Error Resilience for Onboard Remote-Sensing Multimodal Models.

Chinese title:

面向星载多模态遥感大模型的资源感知软错误容错框架。

Traditional multimodal remote-sensing model work mainly emphasizes task accuracy. Space-oriented deployment also has to account for radiation-induced soft errors and limited redundancy resources. This project therefore moves from "can the model be deployed" toward "can the model keep running reliably under resource constraints and possible soft-error events."

Implemented loop:

1. Software-level SEU-equivalent fault injection
2. Task-driven LoRA parameter sensitivity analysis
3. Criticality-aware protection tiering
4. Model asset integrity verification
5. Automatic fault recovery
6. Accuracy, reliability, and resource joint evaluation

Important limitation:

This work is software-level equivalent fault-injection validation only. It has not performed real radiation testing, on-orbit testing, or space-qualified hardware certification.

### Fault Model

The experiment covers:

- LoRA adapter parameter bit flips in temporary safetensors copies
- Static adapter/config file corruption with SHA256 detection
- Automatic recovery from a trusted copy after corruption is detected

bitsandbytes INT4 internal weight/metadata mutation was not attempted because modifying quantized backend internals could destabilize the frozen working environment. The report states this as a stopped high-risk subtask, not as unsupported hardware.

Fault levels:

| Level | Faults per adapter copy | Seeds |
| --- | ---: | --- |
| Level 1 | 1 | 101, 102, 103 |
| Level 2 | 8 | 201, 202, 203 |
| Level 3 | 32 | 301, 302, 303 |

Each manifest records seed, affected tensor, tensor group, element index, bit position, original value/hash, and corrupted value/hash.

### Criticality Score

Criticality is task-driven rather than manually assigned:

Score = 0.35 * max(VQA drop, 0) + 0.35 * max(Grounding mean IoU drop, 0) + 0.15 * max(Acc@0.5 drop, 0) + 0.10 * parse-failure delta + 0.05 * inference-failure rate.

Top sensitive groups in the current ablation:

| Rank | Module group | Score | Class |
| ---: | --- | ---: | --- |
| 1 | q_proj / LoRA-B | 0.099005 | Critical |
| 2 | q_proj / LoRA-A | 0.096575 | Critical |
| 3 | v_proj / LoRA-A | 0.096575 | Critical |
| 4 | o_proj / LoRA-B | 0.042120 | Important |
| 5 | v_proj / LoRA-B | 0.042120 | Important |

### End-to-End Check

The real corrupted-adapter check uses `datasets/mini_eval/mini_eval.jsonl` with LoRA v1 INT4 NF4.

| State | VQA | Grounding IoU | Acc@0.5 | Acc@0.7 | Parse failures | Inference failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Healthy INT4 | 0.600000 | 0.484756 | 0.600000 | 0.300000 | 1 | 0 |
| Fault injected Level-1 mean | 0.733333 | 0.573159 | 0.766667 | 0.300000 | 0 | 0 |
| Fault + OrbitGuard recovery | 0.600000 | 0.484756 | 0.600000 | 0.300000 | 1 | 0 |

The Level-1 real bit flips did not produce monotonic degradation on this small subset. The result is kept unchanged; the broader ranking uses three fault levels and multiple seeds.

### Integrity And Recovery

Startup integrity check validates the deployed LoRA adapter, adapter config, inference config, precision config, and key runtime files with SHA256 and file size.

Measured results:

- Clean startup check: PASS
- Corrupted file check: DETECTED
- Post-recovery check: PASS
- Detection rate in the corruption test: 1.0
- Recovery success rate: 1.0
- False positive rate on clean startup: 0.0
- Integrity check latency: 0.021068 s
- Recovery latency: 0.078948 s

Protection modes:

| Mode | Detection Rate | Recovery Rate | False Positive Rate | Storage Overhead | Check Latency s | Recovery Latency s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No Protection | 0.0 | 0.0 | 0.0 | 0.0% | 0.0 | 0.0 |
| Full Adapter Protection | 1.0 | 1.0 | 0.0 | 100.0% | 0.021068 | 0.078948 |
| Criticality-Aware Selective Protection | 1.0 | 1.0 | 0.0 | 99.843% | 0.021068 | 0.078948 |

Mode 2 currently uses safe file-level critical asset protection. Tensor-level selective recovery would reduce storage further, but was not implemented because rewriting partial safetensors ranges is higher risk at submission time.

### SpaceRS-Agent Four-Layer Innovation Structure

The final system is packaged as **SpaceRS-Agent**. These module names are project-internal names, not claims of external products.

I. **RemoteLoRA**: remote-sensing Caption/VQA/Grounding capability adaptation with LoRA v1.

II. **OrbitQuant**: INT4 NF4 resource-constrained deployment for Qwen3-VL-2B-Instruct + LoRA v1.

III. **HiRes-View**: optional Global + 2x2 high-resolution multi-view fallback.

IV. **OrbitGuard-RS**: software-level soft-error integrity detection, criticality-aware protection, and automatic recovery.

Report-ready figures are generated under `results/reliability/figures/`:

- `orbitguard_architecture.svg`
- `fault_recovery_pipeline.svg`
- `criticality_ranking.svg`
- `fault_level_vqa.svg`
- `fault_level_grounding.svg`
- `healthy_faulted_recovered.svg`
- `protection_tradeoff.svg`

Detailed outputs:

- `results/reliability/satellite_fault_tolerance_ablation.md`
- `results/reliability/orbitguard_summary.json`
- `results/reliability/criticality_scores.csv`
- `results/reliability/fault_level_metrics.csv`
- `results/reliability/protection_modes.csv`

## Reproducibility

Use the unified dispatcher:

```bash
cd /home/lhs/spacers_m3
source .venv/bin/activate
python scripts/benchmarks/run_benchmark.py <benchmark> --precision int4
```

Benchmark names:

- `vrsbench`
- `mme-rs-lite`
- `xlrs-lite`
- `levir-cc`

Full command examples are documented in `README.md`.

## Known Limitations

- VRSBench numbers are internal 30/90 validation results, not official full benchmark scores.
- MME result is official Lite Remote Sensing subset only, not the full MME-RealWorld leaderboard.
- XLRS result is a 44 / 74 shard partial run because the remaining shards could not be downloaded in the current environment.
- LEVIR-CC METEOR is missing because Java is unavailable.
- No additional LoRA or prompt tuning should be run for this delivery branch.

## Key Commits

- LoRA v1 experiment: `e256c33ede496f50074ca8f46591f79d7b6deaf6`
- LoRA v2 ablation: `c0eff8fdbaf531a6e1bb2cd4f71e65e12da521a5`
- Quantization validation: `1d26b35c7ba27ba43fcc4a60d5c9c5dcfeb1a30f`
- Remaining benchmark adapters/results: `78c6d2a9d9e62d0c6d37c929a48237a8f114a753`
- Benchmark dispatcher and reproduction README: `5755253a2111e2714625d766f447cf6accc148dc`
