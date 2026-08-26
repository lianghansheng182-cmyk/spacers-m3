#!/usr/bin/env python3
"""Audit real Qwen3-VL visual-token budgets for remote-sensing benchmarks."""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmarks.evaluate_mme_realworld_rs_lite import build_prompt as build_mme_prompt  # noqa: E402
from scripts.benchmarks.evaluate_mme_realworld_rs_lite import load_rs_rows, resolve_image  # noqa: E402
from scripts.benchmarks.evaluate_xlrs_bench_lite import build_prompt as build_xlrs_prompt  # noqa: E402
from src.inference.spacebudget_router import (  # noqa: E402
    SPACEBUDGET_PROFILES,
    language_visual_tokens_from_grid,
    summarize_token_values,
    visual_patch_tokens_from_grid,
)


DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen3-VL-2B-Instruct"
DEFAULT_MME_ROOT = REPO_ROOT / "data" / "raw" / "mme_realworld_lite"
DEFAULT_XLRS_ROOT = REPO_ROOT / "data" / "raw" / "xlrs_bench_lite"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "spacebudget"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit processor visual token counts.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--mme-root", type=Path, default=DEFAULT_MME_ROOT)
    parser.add_argument("--xlrs-root", type=Path, default=DEFAULT_XLRS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--xlrs-limit", type=int, default=300)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def image_token_id(model_path: Path) -> int:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    return int(config["image_token_id"])


def iter_xlrs_rows(data_root: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    yielded = 0
    for shard_path in sorted((data_root / "train").glob("data-*-of-*.arrow")):
        with pa.memory_map(str(shard_path), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                for row in batch.to_pylist():
                    images = row.get("image") or []
                    if not images:
                        continue
                    item = images[0]
                    image_bytes = item.get("bytes")
                    if image_bytes is None:
                        continue
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    yield {
                        "benchmark": "XLRS",
                        "sample_id": int(row.get("index", yielded + 1)),
                        "task_type": row.get("category"),
                        "category": row.get("category"),
                        "question": row.get("question"),
                        "image": image,
                        "width": image.size[0],
                        "height": image.size[1],
                        "prompt": build_xlrs_prompt(row),
                    }
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return


def iter_mme_rows(data_root: Path) -> Iterable[dict[str, Any]]:
    _, rows = load_rs_rows(data_root)
    for row in rows:
        image_path = resolve_image(data_root, row["Image"])
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
        yield {
            "benchmark": "MME-RS",
            "sample_id": row["Question_id"],
            "task_type": row.get("Category"),
            "category": row.get("Category"),
            "question": row.get("Text"),
            "image": image,
            "width": image.size[0],
            "height": image.size[1],
            "prompt": build_mme_prompt(row),
        }


def audit_one(processor: AutoProcessor, image_pad_token_id: int, item: dict[str, Any], budget_name: str, max_pixels: int) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": item["image"], "max_pixels": max_pixels},
                {"type": "text", "text": item["prompt"]},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    grid = inputs.image_grid_thw.cpu().tolist() if getattr(inputs, "image_grid_thw", None) is not None else []
    patch_tokens = visual_patch_tokens_from_grid(grid)
    language_tokens = language_visual_tokens_from_grid(grid)
    image_token_count = int((inputs.input_ids == image_pad_token_id).sum().item())
    patch_size = int(getattr(processor.image_processor, "patch_size", 16))
    processed_resolution = []
    for t, h, w in grid:
        processed_resolution.append(
            {
                "frames": int(t),
                "height": int(h) * patch_size,
                "width": int(w) * patch_size,
            }
        )
    return {
        "benchmark": item["benchmark"],
        "sample_id": item["sample_id"],
        "task_type": item["task_type"],
        "category": item["category"],
        "budget": budget_name,
        "max_pixels": max_pixels,
        "original_width": item["width"],
        "original_height": item["height"],
        "megapixels": round(item["width"] * item["height"] / 1_000_000, 6),
        "processed_resolution": processed_resolution,
        "image_grid_thw": grid,
        "patch_visual_tokens": patch_tokens,
        "language_visual_tokens": language_tokens,
        "image_token_count": image_token_count,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["benchmark"], {}).setdefault(row["budget"], []).append(row)
    result: dict[str, Any] = {}
    for benchmark, by_budget in grouped.items():
        result[benchmark] = {}
        for budget, items in by_budget.items():
            language = [int(row["language_visual_tokens"]) for row in items]
            patch = [int(row["patch_visual_tokens"]) for row in items]
            image_tokens = [int(row["image_token_count"]) for row in items]
            grids = Counter(json.dumps(row["image_grid_thw"]) for row in items)
            result[benchmark][budget] = {
                "sample_count": len(items),
                "language_visual_tokens": summarize_token_values(language),
                "patch_visual_tokens": summarize_token_values(patch),
                "image_token_count": summarize_token_values(image_tokens),
                "most_common_grids": [{"grid": json.loads(key), "count": count} for key, count in grids.most_common(8)],
            }
    return result


def write_report(path: Path, summary: dict[str, Any], rows_path: Path) -> None:
    lines = [
        "# SpaceBudget-RS Visual Token Audit",
        "",
        "This audit uses the actual Qwen3-VL processor output, not raw image-size estimates.",
        "",
        f"Per-sample rows: `{rows_path}`",
        "",
        "Token calculation:",
        "",
        "- `image_grid_thw` is read directly from processor output.",
        "- Patch visual tokens = sum(t * h * w).",
        "- Language-side visual tokens = patch visual tokens / spatial_merge_size^2. The local model uses spatial_merge_size=2.",
        "- `image_token_count` is counted directly from processor `input_ids == image_token_id`.",
        "",
        "| Benchmark | Budget | max_pixels | Samples | Mean language visual tokens | P50 | P95 | Mean image token count |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for benchmark, by_budget in summary.items():
        for budget, item in by_budget.items():
            token_stats = item["language_visual_tokens"]
            image_stats = item["image_token_count"]
            max_pixels = SPACEBUDGET_PROFILES[budget.lower()].max_pixels
            lines.append(
                f"| {benchmark} | {budget} | {max_pixels} | {item['sample_count']} | "
                f"{token_stats['mean']} | {token_stats['p50']} | {token_stats['p95']} | {image_stats['mean']} |"
            )
    lines.append("")
    lines.append("## Most Common Processor Grids")
    for benchmark, by_budget in summary.items():
        lines.append("")
        lines.append(f"### {benchmark}")
        lines.append("")
        lines.append("| Budget | Grid | Count |")
        lines.append("| --- | --- | ---: |")
        for budget, item in by_budget.items():
            for grid in item["most_common_grids"][:4]:
                lines.append(f"| {budget} | `{grid['grid']}` | {grid['count']} |")
    write_text = "\n".join(lines) + "\n"
    path.write_text(write_text, encoding="utf-8")


def main() -> int:
    Image.MAX_IMAGE_PIXELS = None
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    token_id = image_token_id(args.model_path)
    rows: list[dict[str, Any]] = []
    budgets = [(profile.name, profile.max_pixels) for profile in SPACEBUDGET_PROFILES.values()]
    for item in iter_mme_rows(args.mme_root):
        for budget_name, max_pixels in budgets:
            rows.append(audit_one(processor, token_id, item, budget_name, max_pixels))
    for item in iter_xlrs_rows(args.xlrs_root, args.xlrs_limit):
        for budget_name, max_pixels in budgets:
            rows.append(audit_one(processor, token_id, item, budget_name, max_pixels))
    rows_path = args.output_dir / "visual_token_audit_rows.jsonl"
    summary_path = args.output_dir / "visual_token_audit_summary.json"
    report_path = args.output_dir / "visual_token_audit.md"
    write_jsonl(rows_path, rows)
    summary = summarize(rows)
    write_json(summary_path, summary)
    write_report(report_path, summary, rows_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
