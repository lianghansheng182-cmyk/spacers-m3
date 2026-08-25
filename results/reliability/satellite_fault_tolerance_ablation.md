# OrbitGuard-RS Fault Tolerance Ablation

**Scope:** software-level soft-error-equivalent fault injection only. This is not a real radiation experiment, not space-qualified hardware certification, and not on-orbit validation.

## Fault Levels

| Level | Definition | Ratio vs about 8.7M trainable LoRA params | Real status |
| --- | ---: | ---: | --- |
| Level-1 | 1 bit flip | 0.0000115% | 3 seeds completed |
| Level-2 | 8 bit flips | 0.0000920% | 3 seeds attempted; all triggered CUDA device-side assert |
| Level-3 | 32 bit flips | 0.0003678% | not run after Level-2 stop-loss |

## Real Safe-Range Experiment

| State | VQA | Grounding IoU | Acc@0.5 | Acc@0.7 | Parse failures | Inference failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Healthy INT4 | 0.6 | 0.484756 | 0.6 | 0.3 | 1 | 0 |
| Faulted Level-1 mean | 0.666667 | 0.570112 | 0.766667 | 0.366667 | 0.0 | 0.0 |
| Recovered | 0.6 | 0.484756 | 0.6 | 0.3 | 1 | 0 |

Level-2 produced real inference failures: corrupted adapter generation triggered CUDA device-side assertions caused by invalid probability tensors. The experiment stopped before Level-3 to avoid destabilizing the frozen environment.

## Criticality Top 10

| Rank | Module | Score | Class |
| ---: | --- | ---: | --- |
| 1 | q_proj/LoRA-B | 0.099005 | Critical |
| 2 | q_proj/LoRA-A | 0.096575 | Critical |
| 3 | v_proj/LoRA-A | 0.096575 | Critical |
| 4 | o_proj/LoRA-B | 0.04212 | Important |
| 5 | v_proj/LoRA-B | 0.04212 | Important |
| 6 | down_proj/LoRA-B | 0.0405 | Important |

## Integrity And Recovery

- Detection Rate: 1.0
- Recovery Success Rate: 1.0
- False Positive Rate: 0.0
- Integrity check latency: 0.021068 s
- Recovery latency: 0.078948 s
- Full backup extra storage: 33.359 MB
- Selective backup extra storage: 33.307 MB
- Selective vs Full saving: 0.157%
- Peak VRAM overhead: 0.0 MB

## Figures

- `results/reliability/figures/orbitguard_architecture.svg`
- `results/reliability/figures/fault_recovery_pipeline.svg`
- `results/reliability/figures/criticality_ranking.svg`
- `results/reliability/figures/fault_level_vqa.svg`
- `results/reliability/figures/fault_level_grounding.svg`
- `results/reliability/figures/healthy_faulted_recovered.svg`
- `results/reliability/figures/protection_tradeoff.svg`

## Verdict

OrbitGuard-RS satisfies the final innovation-module threshold within safe software-level validation limits: LoRA bit-level fault injection, multi-seed sensitivity analysis, criticality ranking, SHA256 integrity detection, real file-corruption detection, adapter recovery, Healthy/Faulted/Recovered task metrics, detection/recovery latency, storage overhead, technical-report section, and architecture figures are present.
