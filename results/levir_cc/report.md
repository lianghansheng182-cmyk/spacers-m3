# LEVIR-CC Report

Status: COMPLETE WITH METEOR BLOCKED

Result classification: OFFICIAL SPLIT result on the LEVIR-CC `test` split. BLEU, ROUGE-L, and CIDEr use the official RSICC scorer files. METEOR is unavailable on this host because the official METEOR scorer requires `java`, and `sudo` requires a password.

## Data

- Source: `lcybuaa/LEVIR-CC`
- Local raw path: `data/raw/levir_cc/`
- Archive: `data/raw/levir_cc/Levir-CC-dataset.zip`
- Extracted root: `data/raw/levir_cc/extracted/`
- Caption JSON: `data/raw/levir_cc/extracted/LevirCCcaptions.json`
- Total image pairs: 10077
- Split counts: train 6815, val 1333, test 1929
- References per pair: 5
- Missing A/B image files: 0
- Image size: 256 x 256
- Phase convention: `A` is before / pre-phase, `B` is after / post-phase.

## Protocol

- Input is two images in one Qwen3-VL message: image A first, image B second.
- Prompt states that A is before and B is after, and asks for one concise English change caption describing visible changes only.
- Final evaluation uses the official test split, not train.
- Official scorer source: `data/raw/rsicc_official_code/eval_func/`

## Model

- Base: Qwen3-VL-2B-Instruct
- Adapter: LoRA v1
- Precision: INT4 NF4 bitsandbytes
- Max new tokens: 64
- Max pixels per image: 1048576
- BF16 fallback: not needed

## Smoke

- Output: `results/levir_cc/smoke_int4/`
- Samples: 5
- Inference failures: 0
- Peak VRAM: 1635.11 MB
- Official BLEU/ROUGE-L/CIDEr scorer path works.
- METEOR blocked by missing Java runtime.

## Test Split Result

- Output: `results/levir_cc/`
- Samples: 1929
- Inference failures: 0
- Model load time: 4.372493 s
- Mean latency: 0.859499 s
- P95 latency: 2.390440 s
- Peak VRAM: 1635.11 MB

Official scorer metrics:

| Metric | Value |
| --- | ---: |
| BLEU-1 | 0.324558 |
| BLEU-2 | 0.246340 |
| BLEU-3 | 0.187795 |
| BLEU-4 | 0.144016 |
| ROUGE-L | 0.383925 |
| CIDEr | 0.624115 |
| METEOR | unavailable: Java runtime missing |

Change/no-change classification proxy from official no-change phrase list:

| Split | Count | Accuracy |
| --- | ---: | ---: |
| Change | 964 | 0.562241 |
| No change | 965 | 0.451813 |

## Notes

- This is a true test split run with the official dataset and official BLEU/ROUGE-L/CIDEr scorer code.
- METEOR is the only blocked official metric. The host has no `java`, and `sudo -n true` fails with `sudo: a password is required`, so a system JRE cannot be installed in this run without credentials.
