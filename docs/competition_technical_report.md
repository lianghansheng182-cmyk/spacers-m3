# SPACERS-M3 Technical Report

Date: 2026-08-26

## Model Decision

The final main model is Qwen3-VL-2B-Instruct with the LoRA v1 adapter.

- Base model: Qwen3-VL-2B-Instruct
- Main model: LoRA v1
- Deployment candidate: LoRA v1 INT4 NF4
- Effect reference: LoRA v1 BF16
- LoRA v2: retained as an ablation experiment only

LoRA v1 is kept as the main model because it improved VQA and preserved the best overall Grounding behavior among the stable LoRA experiments. LoRA v2 improved VQA and removed the single Grounding parse failure, but its 90-sample Grounding localization metrics regressed.

## Final Core Results

The final one-page result summary is maintained in `results/final_core_results.md`. The key representative results are:

| Layer | Protocol | Result |
| --- | --- | --- |
| RemoteLoRA | INTERNAL, VRSBench 90 | Base VQA 0.600000 -> LoRA v1 VQA 0.700000 |
| OrbitQuant | INTERNAL, VRSBench 90 | BF16 peak VRAM 4203 MB -> INT4 peak VRAM 1660 MB; INT4 VQA 0.733333 and Grounding mean IoU 0.509680 |
| HiRes-View | PARTIAL, XLRS-Bench Lite local 1848 samples | Single-view 0.389069 -> global+2x2 0.405303 |
| SpaceBudget-RS | PARTIAL, XLRS-Bench Lite local 1848 samples | Fixed BALANCED 0.389069 -> AUTO 0.397186 with 2.46% lower mean visual tokens |
| SpaceBudget-RS | OFFICIAL LITE, MME-RealWorld-RS Lite 150 | Fixed BALANCED 0.473333 -> AUTO 0.460000 with 25.14% lower mean visual tokens |
| OrbitGuard-RS | INTERNAL, software equivalent fault injection | Detection 100%, recovery 100%, false positive 0%, integrity check about 21 ms, recovery about 79 ms |

These results support one coherent system story: onboard remote-sensing MLLM inference is improved through coordinated design across task adaptation, low-bit deployment, high-resolution perception, dynamic compute budgeting, and software-level reliability protection.

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

## SpaceBudget-RS

SpaceBudget-RS adds task-aware dynamic compute budgeting for onboard remote-sensing MLLM inference. It does not retrain the model, modify the LoRA adapter, change evaluation sets, or tune prompts. The module changes only the pre-inference visual compute budget and optional high-resolution route.

Traditional fixed-budget inference applies the same visual token budget to all images and questions. That is poorly matched to satellite-side deployment, where energy, memory, latency, and task priority can change at runtime. SpaceBudget-RS uses deterministic pre-inference metadata to select a compute profile:

| Profile | max_pixels | Mean language visual tokens on MME-RS | Mean language visual tokens on XLRS |
| --- | ---: | ---: | ---: |
| ECO | 262144 | 240.146667 | 251.334957 |
| BALANCED | 1048576 | 981.946667 | 1011.769481 |
| PRECISION | 4194304 | 3963.646667 | 4086.888528 |

The AUTO policy is fixed and reproducible. It uses task/category text, question text, image metadata, and resource mode only; it never uses ground-truth answers or model predictions for routing.

Measured INT4 NF4 results:

| Benchmark | Budget | Accuracy | Mean visual tokens | Mean latency s | Peak VRAM MB | Parse failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MME-RS Lite | ECO | 0.360000 | 240.146667 | 0.338364 | 1667.78 | 0 |
| MME-RS Lite | BALANCED | 0.473333 | 981.946667 | 0.599638 | 1850.04 | 0 |
| MME-RS Lite | PRECISION | 0.493333 | 3963.646667 | 2.834247 | 2597.24 | 0 |
| MME-RS Lite | AUTO | 0.460000 | 735.126667 | 0.561876 | 1850.04 | 0 |
| XLRS Lite current set | ECO | 0.374459 | 251.334957 | 0.294910 | 1715.27 | 0 |
| XLRS Lite current set | BALANCED | 0.389069 | 1011.769481 | 0.555370 | 1901.08 | 0 |
| XLRS Lite current set | PRECISION | 0.388528 | 4086.888528 | 3.429975 | 2646.94 | 1 |
| XLRS Lite current set | AUTO | 0.397186 | 986.916667 | 0.711017 | 1901.08 | 0 |

AUTO reduced MME-RS mean visual tokens by 25.1358% versus fixed BALANCED with a 0.013333 accuracy drop. On XLRS, AUTO reduced mean visual tokens by 2.4564% and improved accuracy by 0.008117 versus fixed BALANCED. PRECISION is not selected as the default deployment path because its token and latency cost is much higher and did not consistently improve XLRS accuracy.

Resource scenarios:

- Power-Limited: force ECO.
- Normal: use AUTO or BALANCED.
- Urgent Precision: use PRECISION and enable HiRes fallback only for categories where tiling is justified.

Pareto interpretation:

- On MME-RS Lite, PRECISION reaches the highest accuracy but increases mean visual tokens from 981.95 to 3963.65 and mean latency from 0.60 s to 2.83 s compared with BALANCED. It is therefore a special-purpose precision mode rather than the default.
- On XLRS-Lite local samples, PRECISION is dominated by AUTO: AUTO has higher accuracy, lower visual token count, lower latency, lower peak VRAM, and no parse failure.
- ECO is the lowest-cost mode and remains useful for power-limited operation, but it loses accuracy on both MME-RS and XLRS.
- AUTO is the preferred dynamic policy for XLRS and a useful MME resource-saving ablation; it should not be described as universally better than fixed BALANCED.

Detailed outputs:

- `results/spacebudget/visual_token_audit.md`
- `results/spacebudget/spacebudget_ablation.md`
- `results/spacebudget/spacebudget_summary.json`
- `results/spacebudget/figures/`

## Ablation Summary

The final branch contains only five innovation modules with experimental evidence:

| Module | Ablation conclusion |
| --- | --- |
| RemoteLoRA | LoRA v1 improves VQA on VRSBench 90 and remains the main adapter; LoRA v2 is not selected because Grounding localization regressed. |
| OrbitQuant | INT4 NF4 substantially reduces measured VRAM. The internal VRSBench result does not show obvious accuracy loss, but the report does not claim INT4 universally improves accuracy. |
| HiRes-View | Global+2x2 tiling improves the XLRS local shard subset, but it is slower and is kept as an optional fallback instead of a default path. |
| SpaceBudget-RS | Dynamic AUTO routing can reduce visual token use and gives a better XLRS Pareto point, but MME shows a small accuracy tradeoff. |
| OrbitGuard-RS | Software-level equivalent fault injection validates integrity detection and recovery behavior; it is not a real radiation or space-qualified hardware test. |

## Domestic AI Chip Migration Plan

The current implementation targets PyTorch, Transformers, PEFT, and bitsandbytes on an NVIDIA RTX 5060 Laptop GPU. Migration to domestic AI accelerators should be staged and measured rather than assumed:

1. Replace bitsandbytes INT4 with the target accelerator's supported low-bit runtime, keeping the same LoRA v1 adapter and evaluation manifests.
2. Port the unified dispatcher and benchmark adapters first, because they isolate model loading, precision backend, prompts, datasets, and output formats.
3. Revalidate processor behavior and visual token accounting. SpaceBudget-RS depends on actual Qwen3-VL processor outputs, so token counts must be re-audited after runtime migration.
4. Re-measure latency, P95, peak memory, parse failures, and accuracy for VRSBench, MME-RS Lite, XLRS-Lite, and LEVIR-CC before claiming deployment parity.
5. Keep OrbitGuard-RS file-level integrity checks hardware-independent. Device-specific memory fault injection should be added only after the target runtime exposes safe low-level hooks.

### SpaceRS-Agent Five-Layer Innovation Structure

The final system is packaged as **SpaceRS-Agent**. These module names are project-internal names, not claims of external products.

Final architecture figure:

- `docs/figures/spacers_agent_architecture.svg`

I. **SpaceBudget-RS**: task-aware dynamic control of visual token budgets and high-resolution fallback.

II. **RemoteLoRA**: remote-sensing Caption/VQA/Grounding capability adaptation with LoRA v1.

III. **OrbitQuant**: INT4 NF4 resource-constrained deployment for Qwen3-VL-2B-Instruct + LoRA v1.

IV. **HiRes-View**: optional Global + 2x2 high-resolution multi-view fallback.

V. **OrbitGuard-RS**: software-level soft-error integrity detection, criticality-aware protection, and automatic recovery.

The five modules are not presented as independent utilities. They form a deployment-oriented chain for satellite remote-sensing scenarios: RemoteLoRA supplies task capability, HiRes-View handles high-resolution details when needed, SpaceBudget-RS controls visual compute under resource modes, OrbitQuant makes the frozen model practical under limited memory, and OrbitGuard-RS protects the deployed assets against software-level equivalent faults.

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

For SpaceBudget-RS on MME-RS Lite and XLRS-Lite:

```bash
python scripts/benchmarks/run_benchmark.py mme-rs-lite --precision int4 --resource-mode auto
python scripts/benchmarks/run_benchmark.py xlrs-lite --precision int4 --resource-mode auto
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
- HiRes-View is an optional fallback. It improves the XLRS local subset but increases latency and is not the default main path.
- INT4 NF4 is selected for deployment efficiency. The internal result is favorable, but this should not be generalized as a guaranteed precision improvement across all datasets.
- SpaceBudget-RS AUTO is deterministic and uses only pre-inference metadata. It does not use ground-truth answers or model outputs for routing.
- OrbitGuard-RS is a software-level equivalent fault-injection and integrity-recovery module, not a real radiation, on-orbit, or space-qualified hardware test.
- No additional LoRA or prompt tuning should be run for this delivery branch.

## Key Commits

- LoRA v1 experiment: `e256c33ede496f50074ca8f46591f79d7b6deaf6`
- LoRA v2 ablation: `c0eff8fdbaf531a6e1bb2cd4f71e65e12da521a5`
- Quantization validation: `1d26b35c7ba27ba43fcc4a60d5c9c5dcfeb1a30f`
- Remaining benchmark adapters/results: `78c6d2a9d9e62d0c6d37c929a48237a8f114a753`
- Benchmark dispatcher and reproduction README: `5755253a2111e2714625d766f447cf6accc148dc`
- SpaceBudget-RS dynamic compute budgeting: `2acffcde76c508218e908ac6355ae1ffcfda8b7e`
