#!/usr/bin/env python3
"""Evaluate a Qwen3-VL LoRA adapter without touching baseline outputs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from transformers import StoppingCriteria, StoppingCriteriaList

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluate_mini_eval import evaluate_files  # noqa: E402
from src.inference.run_batch import build_prompt  # noqa: E402
from src.utils.resource_monitor import ResourceMonitor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA adapter on VRSBench JSONL.")
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--precision", choices=("bf16", "int8", "int4"), default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--grounding-max-new-tokens", type=int, default=None)
    parser.add_argument("--stop-grounding-after-brace", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke runs.")
    return parser.parse_args()


class StopAfterGeneratedBrace(StoppingCriteria):
    def __init__(self, processor: AutoProcessor, prompt_len: int) -> None:
        self.processor = processor
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        generated = input_ids[0, self.prompt_len :]
        if generated.numel() == 0:
            return False
        text = self.processor.batch_decode(
            [generated],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return "}" in text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def pick_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def quantization_config(precision: str) -> BitsAndBytesConfig | None:
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return None


def resolve_image_path(image: str) -> Path:
    path = Path(image)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def build_messages(sample: dict[str, Any], image_path: Path, max_pixels: int, prompt_version: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path), "max_pixels": max_pixels},
                {"type": "text", "text": build_prompt(sample, prompt_version)},
            ],
        }
    ]


def run_one(
    sample: dict[str, Any],
    processor: AutoProcessor,
    model: PeftModel,
    device: torch.device,
    max_new_tokens: int,
    max_pixels: int,
    prompt_version: str,
    stop_after_brace: bool,
) -> str:
    image_path = resolve_image_path(sample["image"])
    messages = build_messages(sample, image_path, max_pixels, prompt_version)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    stopping_criteria = None
    if stop_after_brace:
        stopping_criteria = StoppingCriteriaList([StopAfterGeneratedBrace(processor, inputs.input_ids.shape[1])])
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
        )
    generated_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.json"
    error_cases_path = output_dir / "error_cases.jsonl"
    run_config_path = output_dir / "run_config.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text("", encoding="utf-8")
    error_cases_path.write_text("", encoding="utf-8")

    samples = load_jsonl(args.input.resolve())
    if args.limit is not None:
        samples = samples[: args.limit]
    device = pick_device()
    dtype = pick_dtype(device)
    q_config = quantization_config(args.precision)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    model_kwargs: dict[str, Any] = {
        "device_map": "auto" if device.type == "cuda" else None,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if q_config is None:
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["quantization_config"] = q_config
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        **model_kwargs,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    load_time_s = time.perf_counter() - load_start

    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model_path),
        "adapter_path": str(args.adapter_path),
        "input_path": str(args.input),
        "output_dir": str(output_dir),
        "prompt_version": args.prompt_version,
        "precision": args.precision,
        "max_new_tokens": args.max_new_tokens,
        "grounding_max_new_tokens": args.grounding_max_new_tokens,
        "stop_grounding_after_brace": args.stop_grounding_after_brace,
        "max_pixels": args.max_pixels,
        "device": str(device),
        "dtype": str(dtype),
        "quantization_config": str(q_config) if q_config is not None else None,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A",
        "sample_count": len(samples),
        "model_load_time_s": round(load_time_s, 6),
    }
    write_json(run_config_path, run_config)

    latencies: list[float] = []
    peak_vram_values: list[float] = []
    success_count = 0
    failure_count = 0
    total_start = time.perf_counter()
    for index, sample in enumerate(samples, 1):
        monitor = ResourceMonitor()
        prediction = ""
        error = None
        try:
            monitor.start()
            prediction = run_one(
                sample,
                processor,
                model,
                device,
                args.grounding_max_new_tokens if sample.get("task") == "grounding" and args.grounding_max_new_tokens else args.max_new_tokens,
                args.max_pixels,
                args.prompt_version,
                sample.get("task") == "grounding" and args.stop_grounding_after_brace,
            )
            monitor.stop()
            success = True
            success_count += 1
        except Exception as exc:
            monitor.stop()
            success = False
            failure_count += 1
            error = f"{type(exc).__name__}: {exc}"

        report = monitor.get_report()
        latencies.append(report["latency_s"])
        peak_vram_values.append(report["peak_vram_mb"])
        row = {
            "index": index,
            "id": sample.get("id"),
            "task": sample.get("task"),
            "image": sample.get("image"),
            "question": sample.get("question"),
            "ground_truth": sample.get("answer"),
            "prediction": prediction,
            "success": success,
            "error": error,
            "latency_s": report["latency_s"],
            "peak_vram_mb": report["peak_vram_mb"],
            "max_new_tokens": args.max_new_tokens,
            "effective_max_new_tokens": args.grounding_max_new_tokens if sample.get("task") == "grounding" and args.grounding_max_new_tokens else args.max_new_tokens,
            "stop_after_brace": sample.get("task") == "grounding" and args.stop_grounding_after_brace,
            "sample": sample,
        }
        append_jsonl(predictions_path, row)
        if not success:
            append_jsonl(error_cases_path, row)
        print(f"[{index:03d}/{len(samples):03d}] {sample.get('id')} {sample.get('task')} {'ok' if success else 'error'} latency={report['latency_s']:.3f}s")

    total_runtime_s = time.perf_counter() - total_start
    metrics = evaluate_files(predictions_path=predictions_path, dataset_path=args.input.resolve(), output_path=metrics_path, error_cases_path=error_cases_path)
    adapter_size_bytes = sum(path.stat().st_size for path in args.adapter_path.rglob("*") if path.is_file())
    summary = {
        "input_path": str(args.input),
        "output_dir": str(output_dir),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "error_cases_path": str(error_cases_path),
        "run_config_path": str(run_config_path),
        "sample_count": len(samples),
        "success_count": success_count,
        "failure_count": failure_count,
        "total_runtime_s": round(total_runtime_s, 6),
        "model_load_time_s": round(load_time_s, 6),
        "latency_mean_s": round(statistics.fmean(latencies), 6) if latencies else 0.0,
        "latency_p95_s": round(percentile(latencies, 0.95), 6),
        "peak_vram_mb": round(max(peak_vram_values), 2) if peak_vram_values else 0.0,
        "adapter_size_mb": round(adapter_size_bytes / 1024 / 1024, 3),
        "prompt_version": args.prompt_version,
        "precision": args.precision,
        "max_new_tokens": args.max_new_tokens,
        "grounding_max_new_tokens": args.grounding_max_new_tokens,
        "stop_grounding_after_brace": args.stop_grounding_after_brace,
        "metrics": metrics,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failure_count == 0 else 20


if __name__ == "__main__":
    raise SystemExit(main())
