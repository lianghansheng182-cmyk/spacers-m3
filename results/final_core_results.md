# SPACERS-M3 Final Core Results

This page summarizes the representative, reproducible results for the frozen delivery line. Protocol labels are explicit: INTERNAL results are local validation, OFFICIAL LITE results use official Lite subsets, PARTIAL results use the available local benchmark shard subset, and OFFICIAL TEST results use the benchmark test split available locally.

| Module | Protocol | Setting | Key Result | Resource / Reliability Result | Evidence |
| --- | --- | --- | --- | --- | --- |
| RemoteLoRA | INTERNAL, VRSBench 90 | Base Qwen3-VL-2B vs LoRA v1 BF16 | VQA: 0.600000 -> 0.700000 | LoRA v1 retained as main model; LoRA v2 kept only as ablation | `results/lora/lora_model_selection_report.md` |
| OrbitQuant | INTERNAL, VRSBench 90 | LoRA v1 BF16 vs INT4 NF4 | INT4 VQA: 0.733333; Grounding mean IoU: 0.509680; Acc@0.5: 0.600000 | Peak VRAM: 4203 MB -> 1660 MB, about 60% lower | `results/quantization/lora_v1_quantization_report.md` |
| HiRes-View | PARTIAL, XLRS-Bench Lite 1848 local samples | Single-view INT4 vs global+2x2 tiling | Accuracy: 0.389069 -> 0.405303 | Optional high-resolution fallback, not default main path | `results/highres/highres_tiling_ablation.md` |
| SpaceBudget-RS | PARTIAL, XLRS-Bench Lite 1848 local samples | Fixed BALANCED vs AUTO | Accuracy: 0.389069 -> 0.397186 | Mean visual tokens: 1011.77 -> 986.92, 2.46% lower | `results/spacebudget/spacebudget_ablation.md` |
| SpaceBudget-RS | OFFICIAL LITE, MME-RealWorld-RS Lite 150 | Fixed BALANCED vs AUTO | Accuracy: 0.473333 -> 0.460000 | Mean visual tokens: 981.95 -> 735.13, 25.14% lower | `results/spacebudget/spacebudget_ablation.md` |
| OrbitGuard-RS | INTERNAL, software equivalent fault injection | Criticality-aware integrity check and recovery | Detection rate: 100%; recovery success rate: 100%; false positive rate: 0% | Integrity check: about 21 ms; recovery: about 79 ms | `results/reliability/satellite_fault_tolerance_ablation.md` |

## Main Frozen System

- Base model: Qwen3-VL-2B-Instruct
- Main adapter: LoRA v1
- Deployment candidate: LoRA v1 + INT4 NF4
- Dynamic compute: SpaceBudget-RS AUTO is available for MME-RS Lite and XLRS-Lite
- High-resolution path: HiRes-View is an optional fallback, not enabled globally
- Reliability guard: OrbitGuard-RS protects adapter and critical inference assets through software-level integrity detection and recovery

## Unified Story

SPACERS-M3 targets onboard remote-sensing multimodal inference through coordinated design across five experimentally supported layers: task capability adaptation, low-bit deployment, high-resolution perception, dynamic visual compute budgeting, and software-level reliability protection.
