#!/usr/bin/env python3
"""Minimal Qwen3-VL LoRA trainer for smoke and full adapter runs."""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


def dependency_versions() -> dict[str, str]:
    packages = ["torch", "transformers", "peft", "accelerate", "pillow", "qwen-vl-utils", "pyyaml"]
    out = {}
    for package in packages:
        try:
            out[package] = md.version(package)
        except md.PackageNotFoundError:
            out[package] = "MISSING"
    return out


def gb(value: int | float) -> float:
    return round(float(value) / 1024**3, 3)


def make_messages(row: dict[str, Any], image_path: Path, include_answer: bool) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": row["question"]},
            ],
        }
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]})
    return messages


def encode_sample(
    processor: Any,
    row: dict[str, Any],
    image_root: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    image_path = image_root / row["image"]
    image = Image.open(image_path).convert("RGB")
    full_text = processor.apply_chat_template(
        make_messages(row, image_path, include_answer=True),
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = processor.apply_chat_template(
        make_messages(row, image_path, include_answer=False),
        tokenize=False,
        add_generation_prompt=True,
    )
    full = processor(text=[full_text], images=[image], return_tensors="pt")
    prompt = processor(text=[prompt_text], images=[image], return_tensors="pt")
    labels = full["input_ids"].clone()
    prompt_len = prompt["input_ids"].shape[1]
    labels[:, :prompt_len] = -100
    if "attention_mask" in full:
        labels[full["attention_mask"] == 0] = -100
    full["labels"] = labels
    return {key: value.to(device) for key, value in full.items()}


def prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_model_and_processor(config: dict[str, Any], device: torch.device) -> tuple[Any, Any]:
    model_config = config["model"]
    dtype = torch.bfloat16 if model_config.get("dtype") == "bf16" and device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(
        model_config["path"],
        local_files_only=bool(model_config.get("local_files_only", True)),
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_config["path"],
        dtype=dtype,
        local_files_only=bool(model_config.get("local_files_only", True)),
    )
    model.config.use_cache = bool(model_config.get("use_cache", False))
    model.to(device)
    if model_config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_config = LoraConfig(
        r=int(config["lora"]["r"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        bias=str(config["lora"]["bias"]),
        target_modules=list(config["lora"]["target_modules"]),
    )
    model = get_peft_model(model, lora_config)
    model.train()
    return model, processor


def train(config: dict[str, Any], repo_root: Path, mode: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this LoRA training smoke/full run.")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    random.seed(int(config["run"]["seed"]))
    torch.manual_seed(int(config["run"]["seed"]))

    output_dir = repo_root / (config["run"]["smoke_output_dir"] if mode == "smoke" else config["run"]["output_dir"])
    prepare_output_dir(output_dir)
    shutil.copy2(repo_root / "configs/lora/qwen3_vl_2b_lora_v1.yaml", output_dir / "training_config.yaml")
    write_json(
        output_dir / "environment.json",
        {
            "created": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "git_commit": git_commit(repo_root),
            "dependencies": dependency_versions(),
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_total_gib": gb(torch.cuda.get_device_properties(0).total_memory),
        },
    )

    rows = load_jsonl(repo_root / config["data"]["train"])
    if config["training"].get("shuffle", True):
        random.shuffle(rows)
    image_root = repo_root / config["data"]["image_root"]
    max_steps = int(config["training"]["smoke_max_steps"] if mode == "smoke" else config["training"]["max_steps"])
    grad_accum = int(config["training"]["gradient_accumulation_steps"])
    lr = float(config["training"]["learning_rate"])

    model, processor = load_model_and_processor(config, device)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=float(config["training"]["weight_decay"]))

    log_path = output_dir / "training_log.jsonl"
    recent_losses: deque[float] = deque(maxlen=20)
    raw_losses: list[float] = []
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    optimizer_step = 0
    row_index = 0
    oom_count = 0

    with log_path.open("w", encoding="utf-8") as log_handle:
        while optimizer_step < max_steps:
            row = rows[row_index % len(rows)]
            row_index += 1
            try:
                batch = encode_sample(processor, row, image_root, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs.loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at optimizer step {optimizer_step}: {loss.item()}")
                (loss / grad_accum).backward()
                micro_step += 1
                raw_losses.append(float(loss.detach().cpu()))
                recent_losses.append(float(loss.detach().cpu()))
                if micro_step % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["max_grad_norm"]))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    record = {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "mode": mode,
                        "step": optimizer_step,
                        "micro_step": micro_step,
                        "loss": round(sum(recent_losses) / len(recent_losses), 6),
                        "last_loss": round(raw_losses[-1], 6),
                        "learning_rate": lr,
                        "epoch_estimate": round(row_index / len(rows), 6),
                        "gpu_allocated_gib": gb(torch.cuda.memory_allocated()),
                        "gpu_peak_gib": gb(torch.cuda.max_memory_allocated()),
                        "sample_id": row.get("sample_id"),
                        "task": row.get("task"),
                    }
                    log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_handle.flush()
                    if optimizer_step % int(config["training"]["save_every"]) == 0:
                        ckpt = output_dir / f"checkpoint-step-{optimizer_step}"
                        model.save_pretrained(ckpt)
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if oom_count > 3:
                    raise
            finally:
                del row

    final_dir = output_dir / "adapter_final"
    model.save_pretrained(final_dir)
    elapsed = time.time() - start
    first_window = raw_losses[: min(20, len(raw_losses))]
    last_window = raw_losses[-min(20, len(raw_losses)) :] if raw_losses else []
    summary = {
        "status": "PASS",
        "mode": mode,
        "output_dir": str(output_dir),
        "optimizer_steps": optimizer_step,
        "micro_steps": micro_step,
        "loss_first_window": sum(first_window) / len(first_window) if first_window else None,
        "loss_last_window": sum(last_window) / len(last_window) if last_window else None,
        "loss_min": min(raw_losses) if raw_losses else None,
        "loss_max": max(raw_losses) if raw_losses else None,
        "loss_nan": any(math.isnan(x) for x in raw_losses),
        "oom_count": oom_count,
        "elapsed_seconds": round(elapsed, 3),
        "peak_vram_gib": gb(torch.cuda.max_memory_allocated()),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio_percent": round(trainable_params / total_params * 100, 4),
        "checkpoints": sorted(path.name for path in output_dir.glob("checkpoint-step-*")),
        "final_adapter": str(final_dir),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lora/qwen3_vl_2b_lora_v1.yaml"))
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    summary = train(config, repo_root, args.mode)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
