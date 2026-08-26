# SpaceBudget-RS Visual Token Audit

This audit is based on actual Qwen3-VL processor outputs recorded during inference.
It does not estimate tokens from the original image resolution alone.

Local processor/model configuration:

- Qwen3-VL image budget is controlled through per-image `max_pixels`.
- `image_grid_thw` is read from processor output.
- Patch visual tokens = `sum(t * h * w)`.
- Language-side visual tokens = patch tokens / 4 because `spatial_merge_size=2`.
- Processed resolution is reconstructed as `grid_h/grid_w * patch_size`; local patch size is 16.

Per-sample audit rows are stored in `results/spacebudget/visual_token_audit_samples.jsonl`.

| Benchmark | Budget | max_pixels | Samples | Mean language tokens | P50 | P95 | Peak VRAM MB | Mean latency s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MME-RS | ECO | 262144 | 150 | 240.146667 | 228.0 | 256.0 | 1667.78 | 0.338364 |
| MME-RS | BALANCED | 1048576 | 150 | 981.946667 | 950.0 | 1024.0 | 1850.04 | 0.599638 |
| MME-RS | PRECISION | 4194304 | 150 | 3963.646667 | 4056.0 | 4096.0 | 2597.24 | 2.834247 |
| MME-RS | AUTO | mixed | 150 | 735.126667 | 950.0 | 1024.0 | 1850.04 | 0.561876 |
| XLRS | ECO | 262144 | 1848 | 251.334957 | 256.0 | 256.0 | 1715.27 | 0.29491 |
| XLRS | BALANCED | 1048576 | 1848 | 1011.769481 | 1024.0 | 1024.0 | 1901.08 | 0.55537 |
| XLRS | PRECISION | 4194304 | 1848 | 4086.888528 | 4096.0 | 4096.0 | 2646.94 | 3.429975 |
| XLRS | AUTO | mixed | 1848 | 986.916667 | 1024.0 | 1024.0 | 1901.08 | 0.711017 |
