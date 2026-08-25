# Remaining Benchmarks Readiness Audit

Updated: 2026-08-25

Scope: MME-RealWorld-RS, XLRS-Bench, LEVIR-CC. Model state is frozen: Base is Qwen3-VL-2B-Instruct, main model is LoRA v1, first deployment candidate is LoRA v1 INT4 NF4. No LoRA v3, no VRSBench prompt tuning, and no modification to the 30/90 VRSBench eval sets.

## Local Search Summary

- Project data directories inspected: `datasets/`, `data/`, `scripts/`, `src/`, `configs/`, `results/`.
- Raw benchmark data is stored under `data/raw/`, which is excluded by `.gitignore`.
- Current local supported model path: Qwen3-VL-2B + LoRA v1 through the shared precision backend.
- New benchmark adapters exist under `scripts/benchmarks/`:
  - `evaluate_mme_realworld_rs_lite.py`
  - `evaluate_xlrs_bench_lite.py`
  - `evaluate_levir_cc.py`

## MME-RealWorld-RS

| Item | Status |
| --- | --- |
| Data exists locally | READY |
| Data position | `data/raw/mme_realworld_lite/` |
| Manifest / metadata | `data/raw/mme_realworld_lite/extracted/data/MME-RealWorld-Lite.json` |
| Official evaluator | `data/raw/mme_realworld_official_eval/eval_your_results.py` |
| Local sample scale | 1919 Lite samples total; 150 Remote Sensing samples |
| Official task | Multiple-choice VQA over real-world scenarios; RS is the Remote Sensing domain. |
| Official split | MME-RealWorld-Lite QA set. |
| Official input format | Image plus question and A-E choices. The official prompt asks the model to output only one letter. |
| Official metric | Accuracy, reported by task/domain/category and overall. |
| Current code support | READY for Lite Remote Sensing subset. |
| Result status | COMPLETE: `results/mme_realworld_rs/` |

Actual data scale:

- Remote Sensing samples: 150
- Perception / position: 50
- Perception / color: 50
- Perception / count: 50
- Missing images: 0
- Image resolution range in RS subset: min width/height 921/1062, max width/height 11500/7500

Completed result:

- Model: LoRA v1 INT4 NF4
- Accuracy: 0.466667
- Parse failures: 0
- Official evaluator returncode: 0

Official references:

- https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite
- https://github.com/MME-Benchmarks/MME-RealWorld
- https://mme-realworld.github.io/home_page.html

## XLRS-Bench

| Item | Status |
| --- | --- |
| Data exists locally | PARTIAL |
| Data position | `data/raw/xlrs_bench_lite/` |
| Dataset metadata | `data/raw/xlrs_bench_lite/train/dataset_info.json` |
| Local sample scale | 1848 samples from 44 / 74 Arrow shards |
| Official Lite scale | 3080 samples in 74 Arrow shards |
| Official task | XLRS-Bench-lite VQA / multiple-choice benchmark over ultra-high-resolution remote-sensing images. |
| Official split | Hugging Face `train` split for Lite. |
| Official input format | Image plus question and A-D choices; Overall Land Use Classification can require multiple letters. |
| Official metric | VQA accuracy; the local adapter reports micro accuracy and macro category accuracy. |
| Current code support | READY for available Lite VQA shards; no grounding/caption XLRS support claimed. |
| Result status | PARTIAL: `results/xlrs_bench/partial_44shards_int4/` |

Actual local data scale:

- Available shards: 44 / 74
- Available samples: 1848 / 3080
- Data fields: `path`, `index`, `question`, `multi-choice options`, `answer`, `category`, `l2-category`, `image`
- Observed resolution range in partial run: min width/height 1361/1196, max width/height 11500/10000

Covered local categories:

| Category | Samples |
| --- | ---: |
| Complex reasoning / Anomaly Detection and Interpretation | 100 |
| Complex reasoning / Environmental condition reasoning | 100 |
| Counting / Counting with complex reasoning | 100 |
| Counting / Overall counting | 60 |
| Land use classification / Regional Land use classification | 200 |
| Object properties / Object color | 728 |
| Object properties / Object motion state | 60 |
| Object spatial relationship / Object spatial relationship | 500 |

Completed partial result:

- Model: LoRA v1 INT4 NF4
- Samples: 1848
- Micro accuracy: 0.389069
- Macro category accuracy: 0.478757
- Parse failures: 0
- Inference failures: 0

Remaining blocker:

- Full XLRS-Bench-lite is blocked by incomplete HuggingFace data download. Network checks timed out / returned unreachable before the remaining 30 shards could be downloaded.
- The current result must be reported as internal partial, not official complete Lite.

Official references:

- https://huggingface.co/datasets/initiacms/XLRS-Bench-lite
- https://github.com/AI9Stars/XLRS-Bench

## LEVIR-CC

| Item | Status |
| --- | --- |
| Data exists locally | READY |
| Data position | `data/raw/levir_cc/` |
| Caption JSON | `data/raw/levir_cc/extracted/LevirCCcaptions.json` |
| Image tree | `data/raw/levir_cc/extracted/images/{train,val,test}/{A,B}` |
| Official scorer code | `data/raw/rsicc_official_code/eval_func/` |
| Local sample scale | 10077 image pairs total |
| Official task | Bi-temporal remote-sensing image change captioning. |
| Official split | train 6815, val 1333, test 1929 |
| Official input format | Paired pre-phase/post-phase images; `A` is before/pre-phase and `B` is after/post-phase. |
| Official metric | BLEU, METEOR, ROUGE-L, CIDEr. |
| Current code support | READY for dual-image Qwen3-VL inference and BLEU/ROUGE-L/CIDEr scoring. |
| Result status | COMPLETE WITH METEOR BLOCKED: `results/levir_cc/` |

Actual data scale:

- Total pairs: 10077
- Train: 6815
- Val: 1333
- Test: 1929
- References per pair: 5
- Missing A/B image files: 0
- Image size: 256 x 256

Completed test split result:

- Model: LoRA v1 INT4 NF4
- Samples: 1929
- Failures: 0
- BLEU-4: 0.144016
- ROUGE-L: 0.383925
- CIDEr: 0.624115
- METEOR: unavailable because Java is missing

Remaining blocker:

- METEOR requires Java through the official scorer. `java` is not installed and `sudo -n true` fails because a password is required.

Official references:

- https://huggingface.co/datasets/lcybuaa/LEVIR-CC/tree/main
- https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset
- https://github.com/Chen-Yang-Liu/RSICC
- https://levir.buaa.edu.cn/datasets/index.html

## Readiness Decision

- MME-RealWorld-RS Lite: ready for the official Lite RS subset result.
- XLRS-Bench Lite: partial ready; full Lite result blocked by incomplete data download.
- LEVIR-CC: ready for official test split BLEU/ROUGE-L/CIDEr; METEOR blocked by missing Java.
