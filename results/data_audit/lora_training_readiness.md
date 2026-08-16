# LoRA Training Data Readiness Audit

## Scope And Rules

This audit checks only the current local project state under `/home/lhs/spacers_m3`.

Hard exclusions for LoRA/SFT training:

- Do not train on `datasets/mini_eval/mini_eval.jsonl`.
- Do not train on `datasets/vrsbench_eval_v1/resolved_manifest.jsonl`.
- Do not use `datasets/vrsbench_eval_v1/candidate_manifest.jsonl` for training or evaluation.
- Do not fine-tune on evaluation samples and then evaluate back on the same samples.
- Prefer official VRSBench train data for LoRA/SFT.

## 1. Current Local Data Available For LoRA

Current local annotated data:

| Path | Purpose | Rows | Caption | VQA | Grounding | Training usable? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `datasets/mini_eval/mini_eval.jsonl` | 30-sample dev/eval set | 30 | 10 | 10 | 10 | No |
| `datasets/vrsbench_eval_v1/resolved_manifest.jsonl` | 90-sample internal validation set | 90 | 30 | 30 | 30 | No |
| `datasets/vrsbench_eval_v1/candidate_manifest.jsonl` | Raw candidate source for internal validation | 90 | 30 | 30 | 30 | No |
| `data/test/demo.jpeg` | Demo image only | 0 annotations | 0 | 0 | 0 | No |

Usable LoRA training samples under the current rules:

| Task | Usable samples |
| --- | ---: |
| Caption | 0 |
| VQA | 0 |
| Grounding | 0 |
| Total | 0 |

## 2. Caption / VQA / Grounding Counts

Evaluation-only local counts:

- `mini_eval`: 10 Caption, 10 VQA, 10 Grounding.
- `vrsbench_eval_v1/resolved_manifest.jsonl`: 30 Caption, 30 VQA, 30 Grounding.
- `vrsbench_eval_v1/candidate_manifest.jsonl`: 30 Caption, 30 VQA, 30 Grounding, but this is not the repaired evaluation manifest and must not be used directly for evaluation.

Training-usable counts after excluding both evaluation sets: 0 Caption, 0 VQA, 0 Grounding.

## 3. Complete VRSBench Train Availability

The project does not currently contain a complete official VRSBench train image directory or train annotation manifest suitable for LoRA.

Evidence:

- `datasets/README.md` states that the complete VRSBench dataset is not stored in Git and that only small evaluation subsets are committed.
- Local `datasets/` contains only `mini_eval` and `vrsbench_eval_v1`.
- `src/training/`, `configs/`, and `checkpoints/` do not contain a prepared SFT training manifest.

Current readiness: not ready for LoRA training.

## 4. Overlap With 30-Sample mini_eval

Within the current local evaluation sets:

- Sample ID overlap between `mini_eval` and `vrsbench_eval_v1/resolved_manifest.jsonl`: 0.
- Image basename overlap: 1, `00012_0000.png`.
- Image SHA256 overlap: 1. The overlapping image is:
  - `datasets/mini_eval/images/00012_0000.png`
  - `datasets/vrsbench_eval_v1/images/00012_0000.png`

For any future LoRA training manifest, both sample IDs and image hashes must be checked against `mini_eval`; any overlap must be removed.

## 5. Overlap With 90-Sample vrsbench_eval_v1

No candidate training manifest exists locally, so there is currently no train-vs-90 overlap to measure.

Before training, the future train manifest must be deduplicated against `datasets/vrsbench_eval_v1/resolved_manifest.jsonl` by:

- sample ID;
- image path basename;
- image content SHA256;
- original VRSBench image/sample identifiers if available.

Any overlap with the 90-sample internal validation set must be excluded from training.

## 6. Missing Data

The following are missing before LoRA can start:

- Official VRSBench train images, stored outside the evaluation directories.
- Official VRSBench train annotations for Caption, VQA, and Grounding.
- A project-local SFT manifest built only from training samples, excluding all 30 mini_eval and 90 vrsbench_eval_v1 samples.
- A dedup report proving zero overlap with both evaluation sets by sample ID and image hash.
- Grounding annotation validation confirming `xyxy` boxes in `normalized_0_100` are valid and non-degenerate.
- A train/held-out-train-dev split that is separate from the existing 30 and 90 evaluation sets.

## 7. Recommended First LoRA Training Set Size

Recommended first LoRA/SFT version after official train data is available:

- 9,000 total samples.
- 3,000 Caption.
- 3,000 VQA.
- 3,000 Grounding.

Rationale: this is large enough to test cross-task behavior without immediately building a full-scale training pipeline. If GPU time is constrained, run a smaller pipeline sanity pass first with 300 samples per task, but do not treat that as a real model-quality experiment.

Keep a separate train-dev slice from official train only, for example 300-900 samples, and do not mix it with `mini_eval` or `vrsbench_eval_v1`.

## 8. Recommended Data Format

Use one JSONL row per SFT example:

```json
{
  "id": "vrsbench_train_caption_000001",
  "task": "caption",
  "image": "datasets/vrsbench_train/images/example.png",
  "messages": [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image in detail"}]},
    {"role": "assistant", "content": "A concise factual caption..."}
  ],
  "source": "VRSBench",
  "source_split": "train"
}
```

For VQA, assistant content should be the short ground-truth answer only. For Grounding, assistant content should match the default Prompt v1 evaluation protocol:

```text
{<x1><y1><x2><y2>}
```

Grounding metadata should also be preserved:

```json
{
  "target": "vehicle: The large vehicle located at the bottom-left corner.",
  "bbox": [11, 96, 24, 100],
  "bbox_format": "xyxy",
  "bbox_space": "normalized_0_100"
}
```

## 9. Required Preparation Before Training

Before starting LoRA:

1. Materialize an official VRSBench train-only manifest with Caption, VQA, and Grounding examples.
2. Copy or reference official train images outside `datasets/mini_eval/images/` and `datasets/vrsbench_eval_v1/images/`.
3. Validate all image paths exist.
4. Validate required fields per task.
5. Validate Grounding boxes are `xyxy`, `normalized_0_100`, in range, and non-zero area.
6. Deduplicate by sample ID, image basename, image hash, and original VRSBench identifiers.
7. Emit an overlap report proving zero overlap with the 30-sample and 90-sample evaluation sets.
8. Freeze Prompt v1 as the default evaluation protocol for LoRA comparisons.
9. Define train/dev split, LoRA config, checkpoint naming, and experiment registry entries before launching training.

## Decision

LoRA training data is not ready. The project has evaluation subsets and model/evaluation infrastructure, but it does not currently have a clean, deduplicated, train-only VRSBench SFT dataset.
