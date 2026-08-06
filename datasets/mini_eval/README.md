# VRSBench mini_eval

This directory contains a small VRSBench evaluation subset for smoke testing and
manual inspection.

## Contents

- `mini_eval.jsonl`: 30 UTF-8 JSONL records.
- `images/`: 15 PNG images referenced by the JSONL file.

Current task counts:

- `caption`: 10
- `vqa`: 10
- `grounding`: 10

## Source

The samples were selected from VRSBench training/referring annotations submitted
on `origin/master`. The full raw VRSBench JSON files and source PDFs are not
included in this repository because they are large source artifacts rather than
the curated mini evaluation set.

`source_split` is set to `train` for all records because the submitted metadata
identifies these examples as training-split samples.

## JSONL Schema

Each line is one JSON object. The `task` field is one of `caption`, `vqa`, or
`grounding`.

Caption records contain:

- `id`
- `task`
- `image`
- `question`
- `answer`
- `source_split`

VQA records contain:

- `id`
- `task`
- `image`
- `question`
- `answer`
- `question_type`
- `source_split`

Grounding records contain:

- `id`
- `task`
- `image`
- `question`
- `answer`
- `target`
- `bbox`
- `bbox_format`
- `bbox_space`
- `source_split`

Grounding boxes use `bbox_format: "xyxy"` and
`bbox_space: "normalized_0_100"`, where each coordinate is in the inclusive
range from 0 to 100.

## Validation

Run from the repository root:

```bash
python scripts/data/validate_mini_eval.py
```

Run from another directory with explicit absolute paths:

```bash
python /path/to/spacers_m3/scripts/data/validate_mini_eval.py \
  --repo-root /path/to/spacers_m3 \
  --jsonl /path/to/spacers_m3/datasets/mini_eval/mini_eval.jsonl
```

The validator checks UTF-8 decoding, JSONL parsing, task values, required
fields, duplicate IDs, image existence, expected per-task counts, and grounding
bbox format/range.

## Known Limitations

- This is a 30-record smoke-test subset, not a statistically representative
  benchmark.
- The full VRSBench source annotations are intentionally excluded from the PR.
- Grounding `answer` values are normalized from the submitted VRSBench
  `ground_truth` strings; no extra samples or labels were fabricated.
