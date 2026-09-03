# SPACERS-M3 Submission Manifest

Date: 2026-08-29

## Submission Status

| Item | Status | Path / Note |
| --- | --- | --- |
| README | READY | `README.md` |
| Requirements | READY | `requirements.txt`, `requirements-verified.txt`, `requirements-linux-cu130.txt` |
| Configs | READY | `configs/` |
| Source code | READY | `src/` |
| Benchmark scripts | READY | `scripts/benchmarks/` |
| Environment checker | READY | `scripts/check_environment.py` |
| Artifact checker | READY | `scripts/check_artifacts.py` |
| One-click benchmark dispatcher | READY | `scripts/benchmarks/run_benchmark.py` |
| One-click single-image inference | READY | `scripts/run_single.sh`, `src/inference/run_single.py` |
| LoRA v1 adapter | READY IN SUBMISSION PACKAGE | `03_model/qwen3_vl_2b_lora_v1_adapter/`; Git fallback path `results/lora/qwen3_vl_2b_lora_v1/adapter_final/` |
| LoRA v1 adapter SHA256 manifest | READY | `docs/lora_v1_adapter_manifest.sha256` |
| Technical report | READY | `docs/competition_technical_report.md` |
| Reproduction guide | READY | `docs/reproduction_guide.md` |
| Final core result summary | READY | `results/final_core_results.md` |
| Final architecture figure | READY | `docs/figures/spacers_agent_architecture.svg` |
| Benchmark result summaries | READY | `results/final_benchmark_summary.md`, `results/*/report.md`, selected `summary.json` files |
| SpaceBudget reports | READY | `results/spacebudget/` |
| OrbitGuard reports | READY | `results/reliability/` |

## Required Runtime Assets

These are required to reproduce local inference but are intentionally not committed to Git when large:

| Asset | Path | Size / Note |
| --- | --- | --- |
| Base model | `models/Qwen3-VL-2B-Instruct/` | Excluded from Git; can be restored from official model source or local artifact storage |
| LoRA v1 adapter | `03_model/qwen3_vl_2b_lora_v1_adapter/` | About 34 MB; included in the main submission package, not an external artifact |
| Benchmark raw data | `data/raw/` | Excluded from Git; benchmark downloads or local raw data required for full reruns |
| VRSBench internal eval manifests | `datasets/mini_eval/`, `datasets/vrsbench_eval_v1/` | Kept as allowed small evaluation data in repository policy |

## Baidu Netdisk External Uploads

Large files required for full reproduction are provided through Baidu Netdisk:

- Link: https://pan.baidu.com/s/1GrAYgwE0YkL8rRalwX4g4Q?pwd=r0k4
- Extraction code: `r0k4`
- Shared folder: `external_uploads`
- Access policy: public share link with extraction code, intended to remain permanently valid.

| File | SHA256 | Placement after extraction |
| --- | --- | --- |
| `Qwen3-VL-2B-Instruct__SpaceRS-M3__base_model.zip` | `f97c7387001f1d371a0aa12e5232abaf42cb1c84f73fd9c7b4883bba56af70a7` | `submission_final/02_source_code/models/Qwen3-VL-2B-Instruct/` |
| `raw_benchmark_data__SpaceRS-M3.zip` | `9fe29f2dad8455de88dee725122523e50d9537cf64d1db5ccf17ea5ecd651208` | `submission_final/02_source_code/data/raw/` |

## Reproduction Environment

- Recommended formal environment: WSL2 Ubuntu / Linux.
- Windows native is limited to static checks or a small single-image fallback path and is not the validated full inference or benchmark smoke environment.

## Excluded From Submission Package

| Exclusion | Reason |
| --- | --- |
| `.git/` | VCS metadata |
| `.venv/`, `venv/` | Local Python environment |
| `models/` | Large base model weights |
| `data/raw/` | Raw benchmark datasets |
| training checkpoints | Large and not needed for frozen inference |
| LoRA v2 adapter | Ablation only, not main model |
| `__pycache__/`, `*.pyc` | Python cache |
| `logs/`, temporary downloads, local cache | Nonessential runtime artifacts |
| `results/reliability/**/protected_cache/`, `corrupted_adapters/`, `tmp/` | Fault-injection temporary outputs |

## Estimated Size

| Package Component | Estimated Size |
| --- | ---: |
| Git-tracked code, configs, docs, and selected result summaries | 68.71 MB |
| LoRA v1 adapter included in main package | 33.30 MB |
| Estimated code/results/adapter package | 102.01 MB |

This estimate excludes raw datasets, the base Qwen3-VL model, virtual environments, Git metadata, and local caches.

## Reproduction Entry Points

```bash
source .venv/bin/activate
python scripts/check_environment.py
python scripts/check_artifacts.py --skip-raw
python scripts/benchmarks/run_benchmark.py --help
python scripts/benchmarks/run_benchmark.py mme-rs-lite --precision int4 --resource-mode auto --limit 3
python scripts/benchmarks/run_benchmark.py xlrs-lite --precision int4 --resource-mode auto --limit 3
```

OrbitGuard integrity check:

```bash
python scripts/reliability/run_fault_injection.py --config configs/reliability/orbitguard_v1.yaml
```

## Delivery Notes

- No new LoRA training is part of the final branch.
- No LoRA v3, new prompt, new model, or benchmark protocol modification is included.
- XLRS numbers must be reported as PARTIAL because only the local 44 / 74 shard subset was available.
- VRSBench numbers must be reported as INTERNAL validation.
- MME-RealWorld-RS result is OFFICIAL LITE, not the full leaderboard.
- OrbitGuard-RS results are software-level equivalent fault injection and recovery checks, not real space-radiation hardware tests.
