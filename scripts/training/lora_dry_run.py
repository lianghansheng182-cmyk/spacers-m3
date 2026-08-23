#!/usr/bin/env python3
"""Dry-run Qwen3-VL LoRA data/model plumbing without forward or training."""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def gb(value: int | float) -> float:
    return round(float(value) / 1024**3, 3)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pick_mixed_samples(rows: list[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {"caption": [], "vqa": [], "grounding": []}
    for row in rows:
        task = row.get("task")
        if task in buckets and len(buckets[task]) < total:
            buckets[task].append(row)
    picked = buckets["caption"][:4] + buckets["vqa"][:3] + buckets["grounding"][:3]
    if len(picked) < total:
        picked.extend(row for row in rows if row not in picked)
    return picked[:total]


def validate_rows(rows: list[dict[str, Any]], dataset_dir: Path) -> list[str]:
    errors: list[str] = []
    for i, row in enumerate(rows):
        loc = row.get("sample_id", f"row_{i}")
        for field in ["sample_id", "task", "image", "question", "answer"]:
            if row.get(field) in (None, ""):
                errors.append(f"{loc}: missing {field}")
        image = dataset_dir / str(row.get("image", ""))
        if not image.exists():
            errors.append(f"{loc}: image missing: {image}")
        if row.get("task") == "grounding":
            for field in ["bbox", "bbox_format", "bbox_space"]:
                if row.get(field) in (None, ""):
                    errors.append(f"{loc}: missing {field}")
            bbox = row.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"{loc}: invalid bbox shape")
            elif row.get("bbox_format") != "xyxy" or row.get("bbox_space") != "normalized_0_100":
                errors.append(f"{loc}: invalid bbox metadata")
    return errors


def make_messages(row: dict[str, Any], image_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": row["question"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]},
    ]


def dependency_versions() -> dict[str, str]:
    versions = {}
    for package in ["torch", "transformers", "peft", "accelerate", "pillow", "qwen-vl-utils"]:
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = "MISSING"
    return versions


def count_matching_modules(model: nn.Module) -> dict[str, Any]:
    matches = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in TARGET_MODULES and isinstance(module, nn.Linear):
            matches.append(name)
    return {
        "target_modules": TARGET_MODULES,
        "matched_linear_modules": len(matches),
        "examples": matches[:40],
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Qwen3-VL-2B LoRA Dry-Run Report

Generated: {datetime.now().isoformat(timespec="seconds")}

## Verdict

Dry-run status: **{report['status']}**

Can proceed to smoke training: **{report['can_proceed']}**

No forward pass, backward pass, optimizer step, or checkpoint save was executed in this dry-run.

## Dependencies

```json
{json.dumps(report['dependencies'], indent=2)}
```

## Data

| Item | Value |
|---|---:|
| Dataset dir | `{report['dataset_dir']}` |
| Train samples loaded | {report['sample_count']} |
| Task mix | `{report['task_counts']}` |
| Field errors | {len(report['field_errors'])} |
| Batch tensor keys | `{report['batch_keys']}` |
| input_ids shape | `{report['input_ids_shape']}` |
| pixel_values shape | `{report['pixel_values_shape']}` |

## Model / Processor

| Item | Value |
|---|---|
| Base model | `{report['model_path']}` |
| Processor class | `{report['processor_class']}` |
| Image processor class | `{report['image_processor_class']}` |
| Tokenizer class | `{report['tokenizer_class']}` |
| Chat template present | {report['chat_template_present']} |

## LoRA

Target modules:

```json
{json.dumps(report['module_match'], indent=2)}
```

| Metric | Value |
|---|---:|
| Total parameters | {report['total_parameters']} |
| Trainable parameters | {report['trainable_parameters']} |
| Trainable ratio | {report['trainable_ratio_percent']}% |

## VRAM

| Point | GiB |
|---|---:|
| Before load allocated | {report['vram']['before_load_allocated_gib']} |
| After model load allocated | {report['vram']['after_model_load_allocated_gib']} |
| After LoRA allocated | {report['vram']['after_lora_allocated_gib']} |
| Peak allocated | {report['vram']['peak_allocated_gib']} |

## Issues

{chr(10).join(f'- {item}' for item in report['issues']) if report['issues'] else '- none'}

## Formal Training Risk

{report['training_risk']}
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/vrsbench_lora_v1"))
    parser.add_argument("--model-path", type=Path, default=Path("models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--report", type=Path, default=Path("results/lora/lora_dry_run_report.md"))
    parser.add_argument("--sample-count", type=int, default=10)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_rows = load_jsonl(args.dataset_dir / "train.jsonl")
    samples = pick_mixed_samples(train_rows, args.sample_count)
    field_errors = validate_rows(samples, args.dataset_dir)

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    texts = []
    images = []
    for row in samples:
        image_path = args.dataset_dir / row["image"]
        texts.append(processor.apply_chat_template(make_messages(row, image_path), tokenize=False, add_generation_prompt=False))
        images.append(Image.open(image_path).convert("RGB"))
    batch = processor(text=texts, images=images, padding=True, return_tensors="pt")

    vram = {
        "before_load_allocated_gib": gb(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
    }
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else None,
        local_files_only=True,
    )
    model.eval()
    vram["after_model_load_allocated_gib"] = gb(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0

    module_match = count_matching_modules(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    vram["after_lora_allocated_gib"] = gb(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
    vram["peak_allocated_gib"] = gb(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    ratio = round(trainable_params / total_params * 100, 4)

    issues = []
    if field_errors:
        issues.extend(field_errors)
    if module_match["matched_linear_modules"] == 0:
        issues.append("No LoRA target modules matched the loaded Qwen3-VL model.")
    if not torch.cuda.is_available():
        issues.append("CUDA is not available.")

    training_risk = (
        "High on RTX 5060 Laptop 8GB. Model+LoRA load fits, but real training needs BF16, batch_size=1, "
        "gradient checkpointing, gradient accumulation, low image token budget, and frequent OOM checks."
    )

    report = {
        "status": "PASS" if not issues else "FAIL",
        "can_proceed": not issues,
        "dependencies": dependency_versions(),
        "dataset_dir": str(args.dataset_dir),
        "sample_count": len(samples),
        "task_counts": dict(Counter(row["task"] for row in samples)),
        "field_errors": field_errors,
        "batch_keys": sorted(batch.keys()),
        "input_ids_shape": list(batch["input_ids"].shape),
        "pixel_values_shape": list(batch["pixel_values"].shape) if "pixel_values" in batch else None,
        "model_path": str(args.model_path),
        "processor_class": processor.__class__.__name__,
        "image_processor_class": processor.image_processor.__class__.__name__,
        "tokenizer_class": processor.tokenizer.__class__.__name__,
        "chat_template_present": bool(getattr(processor.tokenizer, "chat_template", None)),
        "module_match": module_match,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio_percent": ratio,
        "vram": vram,
        "issues": issues,
        "training_risk": training_risk,
    }
    write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
