#!/usr/bin/env python3
"""Run zero-shot Qwen3-VL inference on the VRSBench mini_eval JSONL file."""

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
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluate_mini_eval import evaluate_files  # noqa: E402
from src.utils.resource_monitor import ResourceMonitor  # noqa: E402


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen3-VL-2B-Instruct"
DEFAULT_INPUT = REPO_ROOT / "datasets" / "mini_eval" / "mini_eval.jsonl"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "vrsbench_mini" / "qwen3_vl_2b_zero_shot"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch zero-shot inference for VRSBench mini_eval."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: expected JSON object")
            records.append(record)
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_image_path(repo_root: Path, image: str) -> Path:
    image_path = Path(image)
    if not image_path.is_absolute():
        image_path = repo_root / image_path
    return image_path.resolve()


def build_prompt(sample: dict[str, Any]) -> str:
    task = sample["task"]
    question = sample["question"]
    if task == "caption":
        return question
    if task == "vqa":
        return f"{question}\nAnswer with a concise short answer."
    if task == "grounding":
        return (
            f"{question}\nReturn only one bounding box in normalized 0-100 xyxy "
            "format: {<x1><y1><x2><y2>}."
        )
    raise ValueError(f"unsupported task: {task}")


def build_messages(sample: dict[str, Any], image_path: Path, max_pixels: int) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": build_prompt(sample)},
            ],
        }
    ]


def pick_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "BF16"
    if dtype == torch.float16:
        return "FP16"
    if dtype == torch.float32:
        return "FP32"
    return str(dtype)


def to_device(inputs: Any, device: torch.device) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return inputs


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


def run_one_sample(
    sample: dict[str, Any],
    repo_root: Path,
    processor: AutoProcessor,
    model: Qwen3VLForConditionalGeneration,
    device: torch.device,
    max_new_tokens: int,
    max_pixels: int,
) -> str:
    image_path = resolve_image_path(repo_root, sample["image"])
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")

    messages = build_messages(sample, image_path, max_pixels)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = to_device(inputs, device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main() -> int:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("only batch_size=1 is supported for this baseline")

    model_path = args.model_path.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "metrics.json"
    error_cases_path = output_dir / "error_cases.jsonl"
    run_config_path = output_dir / "run_config.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (predictions_path, error_cases_path):
        path.write_text("", encoding="utf-8")

    samples = load_jsonl(input_path)
    device = pick_device()
    dtype = pick_dtype(device)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"

    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "batch_size": args.batch_size,
        "precision": dtype_name(dtype),
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "device": str(device),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "sample_count": len(samples),
    }
    write_json(run_config_path, run_config)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

    print(f"Loading model from: {model_path}")
    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    model_load_time_s = time.perf_counter() - load_start
    run_config["model_load_time_s"] = round(model_load_time_s, 6)
    write_json(run_config_path, run_config)
    print(f"Model loaded in {model_load_time_s:.3f}s")

    total_start = time.perf_counter()
    success_count = 0
    failure_count = 0
    latencies: list[float] = []
    peak_vram_values: list[float] = []

    for index, sample in enumerate(samples, start=1):
        monitor = ResourceMonitor()
        error: str | None = None
        prediction = ""
        try:
            monitor.start()
            prediction = run_one_sample(
                sample=sample,
                repo_root=REPO_ROOT,
                processor=processor,
                model=model,
                device=device,
                max_new_tokens=args.max_new_tokens,
                max_pixels=args.max_pixels,
            )
            monitor.stop()
            success = True
            success_count += 1
        except Exception as exc:  # keep the batch running on individual failures
            monitor.stop()
            success = False
            failure_count += 1
            error = f"{type(exc).__name__}: {exc}"

        report = monitor.get_report()
        latency_s = report["latency_s"]
        peak_vram_mb = report["peak_vram_mb"]
        latencies.append(latency_s)
        peak_vram_values.append(peak_vram_mb)

        result = {
            "index": index,
            "id": sample.get("id"),
            "task": sample.get("task"),
            "image": sample.get("image"),
            "question": sample.get("question"),
            "ground_truth": sample.get("answer"),
            "prediction": prediction,
            "success": success,
            "error": error,
            "latency_s": latency_s,
            "peak_vram_mb": peak_vram_mb,
            "max_new_tokens": args.max_new_tokens,
            "sample": sample,
        }
        append_jsonl(predictions_path, result)
        if not success:
            append_jsonl(error_cases_path, result)

        status = "ok" if success else "error"
        print(
            f"[{index:02d}/{len(samples):02d}] {sample.get('id')} "
            f"{sample.get('task')} {status} latency={latency_s:.3f}s"
        )

    total_runtime_s = time.perf_counter() - total_start
    metrics = evaluate_files(
        predictions_path=predictions_path,
        dataset_path=input_path,
        output_path=metrics_path,
    )

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "error_cases_path": str(error_cases_path),
        "run_config_path": str(run_config_path),
        "sample_count": len(samples),
        "success_count": success_count,
        "failure_count": failure_count,
        "total_runtime_s": round(total_runtime_s, 6),
        "model_load_time_s": round(model_load_time_s, 6),
        "latency_mean_s": round(statistics.fmean(latencies), 6) if latencies else 0.0,
        "latency_p95_s": round(percentile(latencies, 0.95), 6),
        "peak_vram_mb": round(max(peak_vram_values), 2) if peak_vram_values else 0.0,
        "precision": dtype_name(dtype),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "metrics": metrics,
    }
    write_json(summary_path, summary)

    print(f"Saved predictions: {predictions_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved errors: {error_cases_path}")
    print(f"Saved run config: {run_config_path}")
    print(f"Success: {success_count}; Failure: {failure_count}")
    return 0 if failure_count == 0 else 20


if __name__ == "__main__":
    raise SystemExit(main())
