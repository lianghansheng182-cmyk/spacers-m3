# SpaceBudget-RS Ablation

SpaceBudget-RS evaluates fixed visual-compute budgets and a deterministic task-aware AUTO policy for onboard remote-sensing MLLM inference.

All runs use the frozen main model: Qwen3-VL-2B-Instruct + LoRA v1 + INT4 NF4. Prompts, datasets, and evaluators are unchanged.

## Budget Profiles

| Profile | max_pixels | Intended use |
| --- | ---: | --- |
| ECO | 262144 | Power-limited coarse recognition and low-sensitivity tasks |
| BALANCED | 1048576 | Existing fixed single-view benchmark budget |
| PRECISION | 4194304 | Detail-sensitive single-view runs; high latency cost on RTX 5060 Laptop 8GB |

## Results

| Benchmark | Budget | Samples | Accuracy | Mean visual tokens | P95 visual tokens | Mean latency s | P95 latency s | Peak VRAM MB | Run failures | Parse failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MME-RS | ECO | 150 | 0.36 | 240.146667 | 256.0 | 0.338364 | 0.456903 | 1667.78 | 0 | 0 |
| MME-RS | BALANCED | 150 | 0.473333 | 981.946667 | 1024.0 | 0.599638 | 0.716166 | 1850.04 | 0 | 0 |
| MME-RS | PRECISION | 150 | 0.493333 | 3963.646667 | 4096.0 | 2.834247 | 3.034765 | 2597.24 | 0 | 0 |
| MME-RS | AUTO | 150 | 0.46 | 735.126667 | 1024.0 | 0.561876 | 0.815746 | 1850.04 | 0 | 0 |
| XLRS | ECO | 1848 | 0.374459 | 251.334957 | 256.0 | 0.29491 | 0.432953 | 1715.27 | 0 | 0 |
| XLRS | BALANCED | 1848 | 0.389069 | 1011.769481 | 1024.0 | 0.55537 | 0.681128 | 1901.08 | 0 | 0 |
| XLRS | PRECISION | 1848 | 0.388528 | 4086.888528 | 4096.0 | 3.429975 | 4.174926 | 2646.94 | 0 | 1 |
| XLRS | AUTO | 1848 | 0.397186 | 986.916667 | 1024.0 | 0.711017 | 0.939976 | 1901.08 | 0 | 0 |

## AUTO vs Fixed BALANCED

| Benchmark | BALANCED accuracy | AUTO accuracy | Accuracy delta | BALANCED mean tokens | AUTO mean tokens | Token saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MME-RS | 0.473333 | 0.46 | -0.013333 | 981.946667 | 735.126667 | 0.251358 |
| XLRS | 0.389069 | 0.397186 | 0.008117 | 1011.769481 | 986.916667 | 0.024564 |

## AUTO Rule

AUTO is deterministic and uses only pre-inference metadata: benchmark task/category text, question text, image dimensions, and resource mode.

- MME count and XLRS overall-counting use ECO because higher budgets did not improve accuracy reliably.
- Spatial, land-use, motion-state, and complex-reasoning categories use BALANCED because they show measurable ECO-to-BALANCED gains.
- Color/small-object questions use BALANCED when category/question metadata indicates color detail; PRECISION did not improve MME color over BALANCED.
- PRECISION+ HiRes tiling is available as an explicit fallback for selected regional-recognition tasks, but is not enabled globally.

## Pareto Frontier

### MME-RS

- Accuracy vs visual tokens frontier: ECO, BALANCED, PRECISION, AUTO
- Accuracy vs latency frontier: ECO, BALANCED, PRECISION, AUTO
- Accuracy vs peak VRAM frontier: ECO, BALANCED, PRECISION

### XLRS

- Accuracy vs visual tokens frontier: ECO, AUTO
- Accuracy vs latency frontier: ECO, BALANCED, AUTO
- Accuracy vs peak VRAM frontier: ECO, AUTO

## Resource Scenarios

- Power-Limited: force ECO.
- Normal: use AUTO or BALANCED when reproducibility requires a fixed budget.
- Urgent Precision: use PRECISION, and enable HiRes fallback only for categories where tiling has demonstrated value.

This is a compute-resource strategy simulation, not a real satellite power measurement.
