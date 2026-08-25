#!/usr/bin/env python3
"""Evaluate LoRA v1 on the Remote Sensing subset of MME-RealWorld-Lite."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.resource_monitor import ResourceMonitor  # noqa: E402


DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen3-VL-2B-Instruct"
DEFAULT_ADAPTER_PATH = REPO_ROOT / "results" / "lora" / "qwen3_vl_2b_lora_v1" / "adapter_final"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw" / "mme_realworld_lite"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "mme_realworld_rs"
DEFAULT_OFFICIAL_EVAL = (
    REPO_ROOT / "data" / "raw" / "mme_realworld_official_eval" / "eval_your_results.py"
)


ANSWER_PATTERN = re.compile(r"[ABCDE]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MME-RealWorld-Lite Remote Sensing evaluation.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--official-eval", type=Path, default=DEFAULT_OFFICIAL_EVAL)
    parser.add_argument("--precision", choices=("bf16", "int8", "int4"), default="int4")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def find_question_json(data_root: Path) -> Path:
    candidates: list[Path] = []
    for path in data_root.rglob("*.json"):
        if path.name == "output_test_template.json":
            continue
        try:
            data = load_json(path)
        except Exception:
            continue
        rows = data if isinstance(data, list) else []
        if rows and isinstance(rows[0], dict) and {"Question_id", "Image", "Answer choices"}.issubset(rows[0]):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"no MME question JSON found under {data_root}")
    return sorted(candidates, key=lambda item: (len(item.parts), str(item)))[0]


def load_rs_rows(data_root: Path) -> tuple[Path, list[dict[str, Any]]]:
    question_json = find_question_json(data_root)
    rows = load_json(question_json)
    rs_rows = [row for row in rows if row.get("Subtask") == "Remote Sensing"]
    if not rs_rows:
        raise ValueError(f"no Remote Sensing rows found in {question_json}")
    return question_json, rs_rows


def resolve_image(data_root: Path, image_value: str) -> Path:
    rel = Path(image_value)
    direct_candidates = [
        data_root / rel,
        data_root / "images" / rel,
        data_root / "data" / rel,
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = list(data_root.rglob(rel.name))
    for match in matches:
        if match.as_posix().endswith(rel.as_posix()):
            return match.resolve()
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(f"image not found for {image_value}")


def build_prompt(row: dict[str, Any]) -> str:
    lines = [
        str(row["Text"]),
        "The choices are listed below:",
        *[str(choice) for choice in row["Answer choices"]],
        "Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.",
        "The best answer is:",
    ]
    return "\n".join(lines)


def extract_answer(text: str, choices: list[str]) -> str:
    value = text.strip()
    for prefix in (
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ):
        value = value.replace(prefix, "")
    if len(value.split()) > 10 and not ANSWER_PATTERN.search(value):
        return ""
    match = ANSWER_PATTERN.search(value)
    if match is not None:
        return match.group(0)
    for choice in choices:
        if value.lower() in choice.lower():
            return choice[1]
    return ""


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    total = 0
    correct = 0
    parse_failures = 0
    for row in rows:
        task = row["Task"]
        subtask = row["Subtask"]
        category = str(row["Category"]).lower()
        if "attribute" in category:
            category = category.split("/")[0] + "/attribute"
        parsed = extract_answer(str(row.get("Output", "")), list(row["Answer choices"]))
        is_correct = parsed == row["Ground truth"]
        parse_failures += int(parsed == "")
        correct += int(is_correct)
        total += 1
        by_task.setdefault(task, {}).setdefault(subtask, {}).setdefault(
            category,
            {"true": 0, "false": 0, "is_E": 0},
        )
        bucket = by_task[task][subtask][category]
        bucket["true"] += int(is_correct)
        bucket["false"] += int(not is_correct)
        bucket["is_E"] += int(parsed == "E")

    return {
        "sample_count": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "parse_failures": parse_failures,
        "by_task": by_task,
    }


def run_official_eval(official_eval: Path, predictions_path: Path, output_path: Path) -> dict[str, Any]:
    if not official_eval.is_file():
        return {"available": False, "error": f"official eval script not found: {official_eval}"}
    proc = subprocess.run(
        [sys.executable, str(official_eval), "--results_file", str(predictions_path)],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output_path.write_text(proc.stdout, encoding="utf-8")
    return {
        "available": True,
        "returncode": proc.returncode,
        "stdout_path": str(output_path),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.json"
    run_config_path = output_dir / "run_config.json"
    official_stdout_path = output_dir / "official_eval_stdout.txt"

    data_root = args.data_root.resolve()
    question_json, rows = load_rs_rows(data_root)
    if args.limit is not None:
        rows = rows[: args.limit]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    q_config = quantization_config(args.precision)

    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "MME-RealWorld-Lite",
        "subset": "Remote Sensing",
        "question_json": str(question_json),
        "data_root": str(data_root),
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()),
        "precision": args.precision,
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "sample_count": len(rows),
        "limit": args.limit,
    }
    write_json(run_config_path, run_config)

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

    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    peak_vram_values: list[float] = []
    failures = 0
    for index, row in enumerate(rows, start=1):
        prediction = ""
        error = None
        monitor = ResourceMonitor()
        try:
            image_path = resolve_image(data_root, row["Image"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path), "max_pixels": args.max_pixels},
                        {"type": "text", "text": build_prompt(row)},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
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
        except Exception as exc:
            monitor.stop()
            failures += 1
            error = f"{type(exc).__name__}: {exc}"

        report = monitor.get_report()
        latencies.append(report["latency_s"])
        peak_vram_values.append(report["peak_vram_mb"])
        pred_row = dict(row)
        pred_row["Output"] = prediction
        pred_row["Parsed output"] = extract_answer(prediction, list(row["Answer choices"]))
        pred_row["Inference error"] = error
        pred_row["Latency s"] = report["latency_s"]
        pred_row["Peak VRAM MB"] = report["peak_vram_mb"]
        predictions.append(pred_row)
        print(
            f"[{index:03d}/{len(rows):03d}] {row['Question_id']} "
            f"gt={row['Ground truth']} pred={pred_row['Parsed output'] or 'NA'} "
            f"{'error' if error else 'ok'} latency={report['latency_s']:.3f}s"
        )

    write_json(predictions_path, predictions)
    metrics = compute_metrics(predictions)
    official_eval = run_official_eval(args.official_eval.resolve(), predictions_path, official_stdout_path)
    summary = {
        "run_config_path": str(run_config_path),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "official_eval": official_eval,
        "sample_count": len(predictions),
        "failure_count": failures,
        "model_load_time_s": round(load_time_s, 6),
        "latency_mean_s": round(statistics.fmean(latencies), 6) if latencies else 0.0,
        "latency_p95_s": round(percentile(latencies, 0.95), 6),
        "peak_vram_mb": round(max(peak_vram_values), 2) if peak_vram_values else 0.0,
        "metrics": metrics,
    }
    write_json(metrics_path, metrics)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failures == 0 and official_eval.get("returncode", 0) == 0 else 20


if __name__ == "__main__":
    raise SystemExit(main())
