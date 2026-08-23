# VRSBench LoRA v1 Final Audit

Generated: 2026-08-23T22:10:59

## Verdict

Ready for training: **YES**

No LoRA training was started.

## Counts

| Metric | Count |
|---|---:|
| Official train images | 20262 |
| Source image files | 20264 |
| Evaluation blacklist images | 104 |
| Train samples | 9000 |
| Train-dev samples | 600 |
| Final copied images | 3199 |

| Split | Caption | VQA | Grounding |
|---|---:|---:|---:|
| train | 3000 | 3000 | 3000 |
| train-dev | 200 | 200 | 200 |

## Leakage

| Check | Count |
|---|---:|
| basename overlap with eval | 0 |
| hash overlap with eval | 0 |
| train/dev image overlap | 0 |
| duplicate image hashes in final set | 0 |

## Validator

Validator command:

```bash
python scripts/data/validate_vrsbench_lora_v1.py --dataset-dir datasets/vrsbench_lora_v1 --repo-root . --json
```

Validator status: **PASS**

## Build Notes

- Old `lora_train_v1` was not used as a manifest source.
- Official train data was split at image level before sampling rows.
- Images present in `mini_eval` or `vrsbench_eval_v1` were blacklisted by basename and SHA-256 hash.
- Final image paths are relative `images/*.png` paths.
