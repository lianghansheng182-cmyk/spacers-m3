#!/usr/bin/env python3
"""Evaluate LoRA v1 on the LEVIR-CC test split with paired images."""

from __future__ import annotations

import argparse
import json
import re
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.resource_monitor import ResourceMonitor  # noqa: E402


DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen3-VL-2B-Instruct"
DEFAULT_ADAPTER_PATH = REPO_ROOT / "results" / "lora" / "qwen3_vl_2b_lora_v1" / "adapter_final"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw" / "levir_cc" / "extracted"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "levir_cc"
DEFAULT_OFFICIAL_CODE = REPO_ROOT / "data" / "raw" / "rsicc_official_code"

NOCHANGE_SENTENCES = {
    "the scene is the same as before",
    "there is no difference",
    "the two scenes seem identical",
    "no change has occurred",
    "almost nothing has changed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LEVIR-CC paired-image caption evaluation.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--official-code", type=Path, default=DEFAULT_OFFICIAL_CODE)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--precision", choices=("bf16", "int8", "int4"), default="int4")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
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


def find_dataset_root(root: Path) -> Path:
    for candidate in [root, *root.rglob("*")]:
        if (candidate / "LevirCCcaptions.json").is_file() and (candidate / "images").is_dir():
            return candidate
    raise FileNotFoundError(f"could not find LEVIR-CC dataset root under {root}")


def load_caption_rows(dataset_root: Path, split: str, limit: int | None) -> list[dict[str, Any]]:
    captions_path = dataset_root / "LevirCCcaptions.json"
    data = json.loads(captions_path.read_text(encoding="utf-8"))
    images = data["images"] if isinstance(data, dict) and "images" in data else data
    rows: list[dict[str, Any]] = []
    for item in images:
        if str(item.get("split", "")).lower() != split:
            continue
        filename = item.get("filename") or item.get("file_name") or item.get("image") or item.get("filepath")
        if filename is None:
            raise ValueError(f"caption item has no filename: {item}")
        sentences = item.get("sentences") or item.get("captions") or []
        references: list[str] = []
        for sentence in sentences:
            if isinstance(sentence, dict):
                value = sentence.get("raw") or sentence.get("caption") or sentence.get("tokens")
                if isinstance(value, list):
                    value = " ".join(str(token) for token in value)
            else:
                value = sentence
            if value is not None:
                references.append(str(value).strip())
        if not references:
            continue
        rows.append(
            {
                "id": item.get("imgid", len(rows)),
                "filename": filename,
                "split": split,
                "references": references,
                "image_a": dataset_root / "images" / split / "A" / filename,
                "image_b": dataset_root / "images" / split / "B" / filename,
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"no LEVIR-CC rows found for split={split}")
    return rows


def normalize_caption(text: str) -> str:
    value = str(text).lower().strip()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def caption_tokens(text: str) -> list[str]:
    value = normalize_caption(text)
    return value.split() if value else []


def detect_nochange(text: str) -> bool:
    return normalize_caption(text) in NOCHANGE_SENTENCES


def compute_official_metrics(
    official_code: Path,
    references_text: list[list[str]],
    predictions_text: list[str],
) -> tuple[dict[str, Any], str | None]:
    sys.path.insert(0, str(official_code))
    ref = [[" ".join(caption_tokens(ref_caption)) for ref_caption in refs] for refs in references_text]
    hypo = [[" ".join(caption_tokens(prediction))] for prediction in predictions_text]
    metrics: dict[str, Any] = {}
    errors: dict[str, str] = {}
    try:
        from eval_func.bleu.bleu import Bleu  # type: ignore

        score, _ = Bleu(4).compute_score(ref, hypo)
        for name, value in zip(["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"], score):
            metrics[name] = float(value)
    except Exception as exc:
        errors["BLEU"] = f"{type(exc).__name__}: {exc}"
    try:
        from eval_func.meteor.meteor import Meteor  # type: ignore

        score, _ = Meteor().compute_score(ref, hypo)
        metrics["METEOR"] = float(score)
    except Exception as exc:
        errors["METEOR"] = f"{type(exc).__name__}: {exc}"
    try:
        from eval_func.rouge.rouge import Rouge  # type: ignore

        score, _ = Rouge().compute_score(ref, hypo)
        metrics["ROUGE_L"] = float(score)
    except Exception as exc:
        errors["ROUGE_L"] = f"{type(exc).__name__}: {exc}"
    try:
        from eval_func.cider.cider import Cider  # type: ignore

        score, _ = Cider().compute_score(ref, hypo)
        metrics["CIDEr"] = float(score)
    except Exception as exc:
        errors["CIDEr"] = f"{type(exc).__name__}: {exc}"
    return metrics, json.dumps(errors, ensure_ascii=False) if errors else None


def build_prompt() -> str:
    return (
        "You are given two remote-sensing images of the same location at two times. "
        "The first image is time phase A / before. The second image is time phase B / after. "
        "Describe only the actual visible changes between the two images. Do not write two separate captions. "
        "If there is no visible change, say that no change has occurred. Return one concise English change caption."
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.json"
    run_config_path = output_dir / "run_config.json"
    predictions_path.write_text("", encoding="utf-8")

    dataset_root = find_dataset_root(args.data_root.resolve())
    rows = load_caption_rows(dataset_root, args.split, args.limit)
    for row in rows:
        if not row["image_a"].is_file() or not row["image_b"].is_file():
            raise FileNotFoundError(f"missing image pair for {row['filename']}")

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

    latencies: list[float] = []
    peak_vram_values: list[float] = []
    failures = 0
    references_for_metrics: list[list[str]] = []
    predictions_for_metrics: list[str] = []
    nochange_total = 0
    nochange_correct = 0
    change_total = 0
    change_correct = 0

    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "LEVIR-CC",
        "split": args.split,
        "dataset_root": str(dataset_root),
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()),
        "precision": args.precision,
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "limit": args.limit,
        "sample_count": len(rows),
        "prompt": build_prompt(),
    }
    write_json(run_config_path, run_config)

    for index, row in enumerate(rows, start=1):
        prediction = ""
        error = None
        monitor = ResourceMonitor()
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(row["image_a"]), "max_pixels": args.max_pixels},
                        {"type": "image", "image": str(row["image_b"]), "max_pixels": args.max_pixels},
                        {"type": "text", "text": build_prompt()},
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
        references_for_metrics.append(row["references"])
        predictions_for_metrics.append(prediction)

        refs_nochange = any(detect_nochange(ref) for ref in row["references"])
        pred_nochange = detect_nochange(prediction)
        if refs_nochange:
            nochange_total += 1
            nochange_correct += int(pred_nochange)
        else:
            change_total += 1
            change_correct += int(not pred_nochange)

        append_jsonl(
            predictions_path,
            {
                "row_number": index,
                "id": row["id"],
                "filename": row["filename"],
                "image_a": str(row["image_a"]),
                "image_b": str(row["image_b"]),
                "references": row["references"],
                "prediction": prediction,
                "error": error,
                "reference_is_nochange": refs_nochange,
                "prediction_is_nochange": pred_nochange,
                "latency_s": report["latency_s"],
                "peak_vram_mb": report["peak_vram_mb"],
            },
        )
        print(
            f"[{index:04d}/{len(rows):04d}] {row['filename']} "
            f"{'ok' if error is None else 'error'} latency={report['latency_s']:.3f}s",
            flush=True,
        )

    official_metrics, official_error = compute_official_metrics(
        args.official_code.resolve(),
        references_for_metrics,
        predictions_for_metrics,
    )
    metrics = {
        "sample_count": len(rows),
        "failure_count": failures,
        "official_metrics": official_metrics,
        "official_metric_error": official_error,
        "change_total": change_total,
        "change_acc": round(change_correct / change_total, 6) if change_total else None,
        "nochange_total": nochange_total,
        "nochange_acc": round(nochange_correct / nochange_total, 6) if nochange_total else None,
    }
    summary = {
        "run_config_path": str(run_config_path),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "sample_count": len(rows),
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
    return 0 if failures == 0 else 20


if __name__ == "__main__":
    raise SystemExit(main())
