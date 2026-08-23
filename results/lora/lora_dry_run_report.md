# Qwen3-VL-2B LoRA Dry-Run Report

Generated: 2026-08-23T22:26:57

## Verdict

Dry-run status: **PASS**

Can proceed to smoke training: **True**

No forward pass, backward pass, optimizer step, or checkpoint save was executed in this dry-run.

## Dependencies

```json
{
  "torch": "2.13.0+cu130",
  "transformers": "4.57.6",
  "peft": "0.20.0",
  "accelerate": "1.14.0",
  "pillow": "12.3.0",
  "qwen-vl-utils": "0.0.14"
}
```

## Data

| Item | Value |
|---|---:|
| Dataset dir | `datasets/vrsbench_lora_v1` |
| Train samples loaded | 10 |
| Task mix | `{'caption': 4, 'vqa': 3, 'grounding': 3}` |
| Field errors | 0 |
| Batch tensor keys | `['attention_mask', 'image_grid_thw', 'input_ids', 'pixel_values']` |
| input_ids shape | `[10, 369]` |
| pixel_values shape | `[10240, 1536]` |

## Model / Processor

| Item | Value |
|---|---|
| Base model | `models/Qwen3-VL-2B-Instruct` |
| Processor class | `Qwen3VLProcessor` |
| Image processor class | `Qwen2VLImageProcessorFast` |
| Tokenizer class | `Qwen2TokenizerFast` |
| Chat template present | True |

## LoRA

Target modules:

```json
{
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
  ],
  "matched_linear_modules": 196,
  "examples": [
    "model.language_model.layers.0.self_attn.q_proj",
    "model.language_model.layers.0.self_attn.k_proj",
    "model.language_model.layers.0.self_attn.v_proj",
    "model.language_model.layers.0.self_attn.o_proj",
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.up_proj",
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.1.self_attn.q_proj",
    "model.language_model.layers.1.self_attn.k_proj",
    "model.language_model.layers.1.self_attn.v_proj",
    "model.language_model.layers.1.self_attn.o_proj",
    "model.language_model.layers.1.mlp.gate_proj",
    "model.language_model.layers.1.mlp.up_proj",
    "model.language_model.layers.1.mlp.down_proj",
    "model.language_model.layers.2.self_attn.q_proj",
    "model.language_model.layers.2.self_attn.k_proj",
    "model.language_model.layers.2.self_attn.v_proj",
    "model.language_model.layers.2.self_attn.o_proj",
    "model.language_model.layers.2.mlp.gate_proj",
    "model.language_model.layers.2.mlp.up_proj",
    "model.language_model.layers.2.mlp.down_proj",
    "model.language_model.layers.3.self_attn.q_proj",
    "model.language_model.layers.3.self_attn.k_proj",
    "model.language_model.layers.3.self_attn.v_proj",
    "model.language_model.layers.3.self_attn.o_proj",
    "model.language_model.layers.3.mlp.gate_proj",
    "model.language_model.layers.3.mlp.up_proj",
    "model.language_model.layers.3.mlp.down_proj",
    "model.language_model.layers.4.self_attn.q_proj",
    "model.language_model.layers.4.self_attn.k_proj",
    "model.language_model.layers.4.self_attn.v_proj",
    "model.language_model.layers.4.self_attn.o_proj",
    "model.language_model.layers.4.mlp.gate_proj",
    "model.language_model.layers.4.mlp.up_proj",
    "model.language_model.layers.4.mlp.down_proj",
    "model.language_model.layers.5.self_attn.q_proj",
    "model.language_model.layers.5.self_attn.k_proj",
    "model.language_model.layers.5.self_attn.v_proj",
    "model.language_model.layers.5.self_attn.o_proj",
    "model.language_model.layers.5.mlp.gate_proj"
  ]
}
```

| Metric | Value |
|---|---:|
| Total parameters | 2136248320 |
| Trainable parameters | 8716288 |
| Trainable ratio | 0.408% |

## VRAM

| Point | GiB |
|---|---:|
| Before load allocated | 0.0 |
| After model load allocated | 3.964 |
| After LoRA allocated | 3.996 |
| Peak allocated | 3.996 |

## Issues

- none

## Formal Training Risk

High on RTX 5060 Laptop 8GB. Model+LoRA load fits, but real training needs BF16, batch_size=1, gradient checkpointing, gradient accumulation, low image token budget, and frequent OOM checks.
