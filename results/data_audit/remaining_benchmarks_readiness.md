# Remaining Benchmarks Readiness Audit

Created: 2026-08-25

Scope: MME-RealWorld-RS, XLRS-Bench, LEVIR-CC. Model state is frozen: Base is Qwen3-VL-2B-Instruct, main model is LoRA v1, first deployment candidate is LoRA v1 INT4 NF4. No LoRA v3, no VRSBench prompt tuning, and no modification to the 30/90 VRSBench eval sets.

## Local Search Summary

- Project data directories inspected: `datasets/`, `data/`, `scripts/`, `src/`, `configs/`, `results/`.
- Mounted `G:` equivalent inspected: `/mnt/g`; it is mounted but contains no visible files at the checked depth.
- `/home/lhs` bounded search found no usable MME-RealWorld-RS, XLRS-Bench, or LEVIR-CC dataset directories. Matches for `mme` were unrelated package/cache names such as `comments`, `symmetric`, and `recommended`.
- Git-tracked files matching `mme|realworld|xlrs|levir`: none.
- Current local supported benchmark pipeline: VRSBench-style single-image Caption/VQA/Grounding through `src/inference/run_batch.py`, `src/evaluation/evaluate_mini_eval.py`, and `scripts/lora/evaluate_lora_adapter.py`.

## MME-RealWorld-RS

| Item | Status |
| --- | --- |
| Data exists locally | MISSING |
| Data position | Not found under `/home/lhs/spacers_m3`, `/home/lhs`, or `/mnt/g`. |
| Local sample scale | N/A; no local manifest/images found. |
| Official task | Multiple-choice VQA over real-world scenarios; RS is the remote-sensing domain. |
| Official split | Hugging Face dataset exposes a `train` split; official benchmark is evaluated by generating model responses for the provided QA set. |
| Official input format | Image plus question and five choices A-E. The official prompt asks the model to output only one letter. |
| Official metric | Accuracy, reported across domains/categories/subtasks and perception/reasoning task types. |
| Current code support | PARTIAL. Qwen3-VL + LoRA v1 + BF16/INT8/INT4 loading is available, but there is no MME manifest adapter or official result writer/evaluator integration. |
| Missing | Dataset files, RS-domain filtering/manifest conversion, A-E answer parser, official `eval_your_results.py` compatibility, result report. |

Official references checked:

- https://github.com/MME-Benchmarks/MME-RealWorld
- https://mme-realworld.github.io/home_page.html
- https://huggingface.co/datasets/yifanzhang114/MME-RealWorld

## XLRS-Bench

| Item | Status |
| --- | --- |
| Data exists locally | MISSING |
| Data position | Not found under `/home/lhs/spacers_m3`, `/home/lhs`, or `/mnt/g`. |
| Local sample scale | N/A; no local manifest/images found. |
| Official task | Ultra-high-resolution remote-sensing MLLM benchmark with 16 subtasks across perception and reasoning; includes VQA, visual grounding, and detailed image captioning resources. |
| Official split | Hugging Face resources list VQA lite splits and caption/grounding datasets; grounding resources include train/test. |
| Official input format | VQA uses image plus question and A-D choices; overall land-use classification may allow multiple answers. Grounding/caption require task-specific formats from the official `evaluation` folder. |
| Official metric | VQA average accuracy, with macro/micro averages depending on lite/full reporting. Grounding and captioning metrics are reported separately by the official evaluation resources. |
| Current code support | PARTIAL. Single-image model loading and VRSBench-style bbox parsing exist, but XLRS-specific ultra-high-resolution tiling, A-D/multi-answer parser, grounding/caption protocol, and official evaluator are absent. |
| Missing | Dataset files, official evaluation folder or lmms-eval task wiring, high-resolution image handling policy, per-task adapter, result report. |

Official references checked:

- https://github.com/AI9Stars/XLRS-Bench
- https://arxiv.org/abs/2503.23771

## LEVIR-CC

| Item | Status |
| --- | --- |
| Data exists locally | MISSING |
| Data position | Not found under `/home/lhs/spacers_m3`, `/home/lhs`, or `/mnt/g`. |
| Local sample scale | N/A; no local `LevirCCcaptions.json` or `images/{train,val,test}/{A,B}` tree found. |
| Official task | Bi-temporal remote-sensing image change captioning. |
| Official split | Official dataset layout has `images/train/A`, `images/train/B`, `images/val/A`, `images/val/B`, `images/test/A`, `images/test/B`, plus `LevirCCcaptions.json`. |
| Official input format | Paired pre-phase/post-phase images. `A` is pre-phase and `B` is post-phase. |
| Official metric | Caption generation metrics such as BLEU, METEOR, ROUGE-L, and CIDEr are standard for LEVIR-CC reporting; exact scorer must be fixed once local official code/data is available. |
| Current code support | MISSING for the benchmark. Current inference path is single-image; LEVIR-CC needs a dual-image prompt/input adapter and caption evaluator. |
| Missing | Dataset files, dual-image Qwen3-VL adapter, caption references loader, BLEU/METEOR/ROUGE-L/CIDEr evaluator, result report. |

Official references checked:

- https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset
- https://levir.buaa.edu.cn/datasets/index.html

## Readiness Decision

- MME-RealWorld-RS: not ready. Data and adapter/evaluator are missing.
- XLRS-Bench: not ready. Data and adapter/evaluator are missing.
- LEVIR-CC: not ready. Data and dual-image/evaluator support are missing.

No smoke test or official benchmark run was launched for these three benchmarks because no local benchmark data was available. This avoids producing non-reproducible or mislabeled results.
