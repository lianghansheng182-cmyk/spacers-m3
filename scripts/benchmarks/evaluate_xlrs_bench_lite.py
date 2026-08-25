#!/usr/bin/env python3
"""Evaluate LoRA v1 on XLRS-Bench-lite VQA."""

from __future__ import annotations

import argparse
import io
import json
import re
import statistics
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.resource_monitor import ResourceMonitor  # noqa: E402


DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen3-VL-2B-Instruct"
DEFAULT_ADAPTER_PATH = REPO_ROOT / "results" / "lora" / "qwen3_vl_2b_lora_v1" / "adapter_final"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw" / "xlrs_bench_lite"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "xlrs_bench"

LETTER_PATTERN = re.compile(r"[ABCD]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XLRS-Bench-lite evaluation.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--precision", choices=("bf16", "int8", "int4"), default="int4")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
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
    return ordered[lower] * (1 - (rank - lower)) + ordered[upper] * (rank - lower)


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


def iter_arrow_rows(data_root: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    yielded = 0
    shard_paths = sorted((data_root / "train").glob("data-*-of-*.arrow"))
    if not shard_paths:
        raise FileNotFoundError(f"no XLRS Arrow shards found under {data_root / 'train'}")
    for shard_path in shard_paths:
        with pa.memory_map(str(shard_path), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                for row in batch.to_pylist():
                    yield row
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return


def is_multi_answer(row: dict[str, Any]) -> bool:
    return "Overall Land Use Classification" in str(row.get("category", ""))


def build_prompt(row: dict[str, Any]) -> str:
    lines = [
        str(row["question"]),
        "The choices are listed below:",
        *[str(choice) for choice in row["multi-choice options"]],
    ]
    if is_multi_answer(row):
        lines.append(
            "Select the best answer(s) for the multiple-choice question based on the image. "
            "There may be more than one correct option. Only respond with the letter(s) "
            "corresponding to the correct answer(s) (A, B, C, D), with multiple choices "
            "separated by spaces."
        )
    else:
        lines.append(
            "Select the best answer for the multiple-choice question based on the image. "
            "Only respond with the letter corresponding to the correct answer (A, B, C, D)."
        )
    lines.append("The answer is:")
    return "\n".join(lines)


def parse_letters(text: str, multi: bool) -> str:
    letters = LETTER_PATTERN.findall(text.upper())
    if not letters:
        return ""
    if multi:
        return " ".join(sorted(set(letters)))
    return letters[0]


def normalize_answer(answer: Any, multi: bool) -> str:
    if isinstance(answer, list):
        letters = [str(item).upper().strip() for item in answer]
        return " ".join(sorted(set(item for item in letters if LETTER_PATTERN.fullmatch(item))))
    return parse_letters(str(answer), multi)


def image_from_row(row: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    images = row.get("image") or []
    if not images:
        raise ValueError("row has no image entries")
    item = images[0]
    image_bytes = item.get("bytes")
    if image_bytes is None:
        raise ValueError("row image entry has no bytes")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    metadata = {
        "image_path": item.get("path") or row.get("path"),
        "original_width": image.size[0],
        "original_height": image.size[1],
        "original_pixels": image.size[0] * image.size[1],
        "image_bytes": len(image_bytes),
    }
    return image, metadata


def update_metrics(metrics: dict[str, Any], result: dict[str, Any]) -> None:
    metrics["sample_count"] += 1
    metrics["correct"] += int(result["correct"])
    metrics["parse_failures"] += int(result["parsed_answer"] == "")
    metrics["category_counts"][result["category"]] += 1
    metrics["category_correct"][result["category"]] += int(result["correct"])
    metrics["l2_counts"][result["l2_category"]] += 1
    metrics["l2_correct"][result["l2_category"]] += int(result["correct"])
    metrics["multi_answer_count"] += int(result["is_multi_answer"])


def finalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    sample_count = metrics["sample_count"]
    category_scores = {
        category: round(metrics["category_correct"][category] / count, 6)
        for category, count in sorted(metrics["category_counts"].items())
    }
    l2_scores = {
        category: round(metrics["l2_correct"][category] / count, 6)
        for category, count in sorted(metrics["l2_counts"].items())
    }
    return {
        "sample_count": sample_count,
        "correct": metrics["correct"],
        "accuracy_micro": round(metrics["correct"] / sample_count, 6) if sample_count else 0.0,
        "accuracy_macro_category": round(statistics.fmean(category_scores.values()), 6)
        if category_scores
        else 0.0,
        "parse_failures": metrics["parse_failures"],
        "multi_answer_count": metrics["multi_answer_count"],
        "category_counts": dict(sorted(metrics["category_counts"].items())),
        "category_accuracy": category_scores,
        "l2_counts": dict(sorted(metrics["l2_counts"].items())),
        "l2_accuracy": l2_scores,
    }


def main() -> int:
    Image.MAX_IMAGE_PIXELS = None
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.json"
    run_config_path = output_dir / "run_config.json"
    predictions_path.write_text("", encoding="utf-8")

    data_root = args.data_root.resolve()
    rows_iter = iter_arrow_rows(data_root, args.limit)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
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
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    load_time_s = time.perf_counter() - load_start

    metrics_accumulator: dict[str, Any] = {
        "sample_count": 0,
        "correct": 0,
        "parse_failures": 0,
        "category_counts": Counter(),
        "category_correct": Counter(),
        "l2_counts": Counter(),
        "l2_correct": Counter(),
        "multi_answer_count": 0,
    }
    latencies: list[float] = []
    peak_vram_values: list[float] = []
    failures = 0
    resolution_stats: dict[str, Any] = {
        "min_width": None,
        "max_width": 0,
        "min_height": None,
        "max_height": 0,
        "max_pixels": 0,
    }

    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "XLRS-Bench-lite",
        "split": "train",
        "data_root": str(data_root),
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()),
        "precision": args.precision,
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "limit": args.limit,
    }
    write_json(run_config_path, run_config)

    for index, row in enumerate(rows_iter, start=1):
        prediction = ""
        parsed_answer = ""
        target_answer = ""
        error = None
        image_meta: dict[str, Any] = {}
        image_grid_thw = None
        monitor = ResourceMonitor()
        try:
            image, image_meta = image_from_row(row)
            width = image_meta["original_width"]
            height = image_meta["original_height"]
            resolution_stats["min_width"] = width if resolution_stats["min_width"] is None else min(resolution_stats["min_width"], width)
            resolution_stats["max_width"] = max(resolution_stats["max_width"], width)
            resolution_stats["min_height"] = height if resolution_stats["min_height"] is None else min(resolution_stats["min_height"], height)
            resolution_stats["max_height"] = max(resolution_stats["max_height"], height)
            resolution_stats["max_pixels"] = max(resolution_stats["max_pixels"], image_meta["original_pixels"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image, "max_pixels": args.max_pixels},
                        {"type": "text", "text": build_prompt(row)},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
                image_grid_thw = inputs.image_grid_thw.cpu().tolist()
            inputs = inputs.to(device)
            monitor.start()
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            monitor.stop()
            generated_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            prediction = processor.batch_decode(
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            multi = is_multi_answer(row)
            parsed_answer = parse_letters(prediction, multi)
            target_answer = normalize_answer(row["answer"], multi)
            correct = parsed_answer == target_answer
        except Exception as exc:
            monitor.stop()
            failures += 1
            error = f"{type(exc).__name__}: {exc}"
            multi = is_multi_answer(row)
            target_answer = normalize_answer(row.get("answer", ""), multi)
            correct = False

        report = monitor.get_report()
        latencies.append(report["latency_s"])
        peak_vram_values.append(report["peak_vram_mb"])
        result = {
            "index": int(row.get("index", index)),
            "row_number": index,
            "path": row.get("path"),
            "question": row.get("question"),
            "choices": row.get("multi-choice options"),
            "answer": row.get("answer"),
            "target_answer": target_answer,
            "prediction": prediction,
            "parsed_answer": parsed_answer,
            "correct": correct,
            "category": row.get("category"),
            "l2_category": row.get("l2-category"),
            "is_multi_answer": multi,
            "error": error,
            "latency_s": report["latency_s"],
            "peak_vram_mb": report["peak_vram_mb"],
            "image": image_meta,
            "image_grid_thw": image_grid_thw,
        }
        update_metrics(metrics_accumulator, result)
        append_jsonl(predictions_path, result)
        print(
            f"[{index:04d}] idx={result['index']} gt={target_answer or 'NA'} "
            f"pred={parsed_answer or 'NA'} {'ok' if error is None else 'error'} "
            f"latency={report['latency_s']:.3f}s",
            flush=True,
        )

    metrics = finalize_metrics(metrics_accumulator)
    summary = {
        "run_config_path": str(run_config_path),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "sample_count": metrics["sample_count"],
        "failure_count": failures,
        "model_load_time_s": round(load_time_s, 6),
        "latency_mean_s": round(statistics.fmean(latencies), 6) if latencies else 0.0,
        "latency_p95_s": round(percentile(latencies, 0.95), 6),
        "peak_vram_mb": round(max(peak_vram_values), 2) if peak_vram_values else 0.0,
        "resolution_stats": resolution_stats,
        "metrics": metrics,
    }
    write_json(metrics_path, metrics)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failures == 0 else 20


if __name__ == "__main__":
    raise SystemExit(main())
