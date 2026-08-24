# LEVIR-CC Report

Status: NOT RUN

Reason: local LEVIR-CC data was not found under `/home/lhs/spacers_m3`, `/home/lhs`, or `/mnt/g`.

Planned model entry:

- Model: Qwen3-VL-2B-Instruct
- Adapter: LoRA v1
- First precision: INT4 NF4
- Fallback precision: BF16 if INT4 has compatibility or output-format failures

Protocol notes:

- LEVIR-CC is a bi-temporal remote-sensing change-captioning benchmark.
- Official layout uses `LevirCCcaptions.json` and paired `images/{train,val,test}/A` and `images/{train,val,test}/B` directories.
- `A` is the pre-phase image and `B` is the post-phase image.

Current support gap:

- No local dataset.
- Current Qwen3-VL inference entry is single-image VRSBench-style; LEVIR-CC requires a dual-image adapter.
- Caption metrics such as BLEU/METEOR/ROUGE-L/CIDEr are not wired into this project yet.

Result classification: missing-data blocker, not smoke, not internal validation, not official protocol result.
