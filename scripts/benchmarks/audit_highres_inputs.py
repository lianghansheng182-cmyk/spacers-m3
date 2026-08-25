#!/usr/bin/env python3
"""Audit high-resolution benchmark inputs and processor image grids."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmarks.evaluate_mme_realworld_rs_lite import load_rs_rows, resolve_image  # noqa: E402


XLRS_PREDICTIONS = REPO_ROOT / "results" / "xlrs_bench" / "partial_44shards_int4" / "predictions.jsonl"
MME_DATA_ROOT = REPO_ROOT / "data" / "raw" / "mme_realworld_lite"
OUTPUT_PATH = REPO_ROOT / "results" / "highres" / "highres_input_audit.md"
DEFAULT_MAX_PIXELS = 1024 * 1024


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


def summarize_sizes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    widths = [row["width"] for row in rows]
    heights = [row["height"] for row in rows]
    megapixels = [row["megapixels"] for row in rows]
    compression = [row["compression_ratio_to_1mp"] for row in rows]
    return {
        "count": len(rows),
        "width_min": min(widths),
        "width_median": statistics.median(widths),
        "width_mean": statistics.fmean(widths),
        "width_max": max(widths),
        "height_min": min(heights),
        "height_median": statistics.median(heights),
        "height_mean": statistics.fmean(heights),
        "height_max": max(heights),
        "mp_min": min(megapixels),
        "mp_median": statistics.median(megapixels),
        "mp_mean": statistics.fmean(megapixels),
        "mp_max": max(megapixels),
        "compression_median": statistics.median(compression),
        "compression_mean": statistics.fmean(compression),
        "compression_max": max(compression),
    }


def load_xlrs_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with XLRS_PREDICTIONS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            image = item["image"]
            width = int(image["original_width"])
            height = int(image["original_height"])
            pixels = width * height
            grid = item.get("image_grid_thw") or []
            visual_tokens = None
            if grid:
                visual_tokens = sum(int(t) * int(h) * int(w) for t, h, w in grid)
            rows.append(
                {
                    "id": item.get("index"),
                    "question": item.get("question"),
                    "category": item.get("category"),
                    "path": item.get("path"),
                    "width": width,
                    "height": height,
                    "megapixels": pixels / 1_000_000,
                    "compression_ratio_to_1mp": pixels / DEFAULT_MAX_PIXELS,
                    "image_grid_thw": grid,
                    "visual_tokens": visual_tokens,
                    "correct": bool(item.get("correct")),
                }
            )
    return rows


def load_mme_rows() -> list[dict[str, Any]]:
    _, rows = load_rs_rows(MME_DATA_ROOT)
    audited: list[dict[str, Any]] = []
    for row in rows:
        image_path = resolve_image(MME_DATA_ROOT, row["Image"])
        with Image.open(image_path) as image:
            width, height = image.size
        pixels = width * height
        audited.append(
            {
                "id": row["Question_id"],
                "question": row["Text"],
                "category": row["Category"],
                "path": row["Image"],
                "width": width,
                "height": height,
                "megapixels": pixels / 1_000_000,
                "compression_ratio_to_1mp": pixels / DEFAULT_MAX_PIXELS,
            }
        )
    return audited


def bucket(mp: float) -> str:
    if mp < 4:
        return "<4MP"
    if mp <= 16:
        return "4-16MP"
    return ">16MP"


def bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(bucket(row["megapixels"]) for row in rows))


def format_summary_table(name: str, summary: dict[str, Any]) -> list[str]:
    return [
        f"## {name}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Samples | {summary['count']} |",
        f"| Width min / median / mean / max | {summary['width_min']:.0f} / {summary['width_median']:.0f} / {summary['width_mean']:.1f} / {summary['width_max']:.0f} |",
        f"| Height min / median / mean / max | {summary['height_min']:.0f} / {summary['height_median']:.0f} / {summary['height_mean']:.1f} / {summary['height_max']:.0f} |",
        f"| Megapixels min / median / mean / max | {summary['mp_min']:.3f} / {summary['mp_median']:.3f} / {summary['mp_mean']:.3f} / {summary['mp_max']:.3f} |",
        f"| Compression to 1MP median / mean / max | {summary['compression_median']:.2f}x / {summary['compression_mean']:.2f}x / {summary['compression_max']:.2f}x |",
        "",
    ]


def risky_rows(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: row["megapixels"], reverse=True)[:limit]


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    xlrs = load_xlrs_rows()
    mme = load_mme_rows()
    xlrs_summary = summarize_sizes(xlrs)
    mme_summary = summarize_sizes(mme)
    grid_counter = Counter(
        json.dumps(row["image_grid_thw"], sort_keys=True)
        for row in xlrs
        if row.get("image_grid_thw")
    )

    lines: list[str] = [
        "# High-Resolution Input Audit",
        "",
        "Created: 2026-08-25",
        "",
        "Scope: XLRS-Bench-lite local 44-shard partial run and MME-RealWorld-Lite Remote Sensing subset.",
        "",
        f"Processor `max_pixels` used by current benchmark scripts: `{DEFAULT_MAX_PIXELS}`.",
        "The existing XLRS single-view predictions record `image_grid_thw`; MME-RS dimensions are audited from image files.",
        "",
    ]
    lines.extend(format_summary_table("XLRS-Bench-lite Local 44-Shard Partial", xlrs_summary))
    lines.append("Resolution buckets:")
    lines.append("")
    lines.append("| Bucket | Samples |")
    lines.append("| --- | ---: |")
    for name, count in sorted(bucket_counts(xlrs).items()):
        lines.append(f"| {name} | {count} |")
    lines.append("")
    lines.append("Most common XLRS processor `image_grid_thw` values:")
    lines.append("")
    lines.append("| image_grid_thw | Samples | Approx visual tokens |")
    lines.append("| --- | ---: | ---: |")
    for key, count in grid_counter.most_common(8):
        grid = json.loads(key)
        tokens = sum(int(t) * int(h) * int(w) for t, h, w in grid)
        lines.append(f"| `{grid}` | {count} | {tokens} |")
    lines.append("")
    lines.extend(format_summary_table("MME-RealWorld-Lite Remote Sensing", mme_summary))
    lines.append("Resolution buckets:")
    lines.append("")
    lines.append("| Bucket | Samples |")
    lines.append("| --- | ---: |")
    for name, count in sorted(bucket_counts(mme).items()):
        lines.append(f"| {name} | {count} |")
    lines.append("")
    lines.append("## Samples Most Exposed To Downscaling Detail Loss")
    lines.append("")
    lines.append("XLRS largest images:")
    lines.append("")
    lines.append("| Row | Category | Size | MP | Grid | Single correct | Question |")
    lines.append("| --- | --- | ---: | ---: | --- | --- | --- |")
    for row in risky_rows(xlrs):
        question = str(row["question"]).replace("|", "/")[:120]
        lines.append(
            f"| {row['id']} | {row['category']} | {row['width']}x{row['height']} | "
            f"{row['megapixels']:.2f} | `{row.get('image_grid_thw')}` | {row['correct']} | {question} |"
        )
    lines.append("")
    lines.append("MME-RS largest images:")
    lines.append("")
    lines.append("| Question ID | Category | Size | MP | Question |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for row in risky_rows(mme):
        question = str(row["question"]).replace("|", "/")[:120]
        lines.append(
            f"| {row['id']} | {row['category']} | {row['width']}x{row['height']} | "
            f"{row['megapixels']:.2f} | {question} |"
        )
    lines.append("")
    lines.append("## Audit Conclusion")
    lines.append("")
    lines.append("- XLRS contains many true ultra-high-resolution samples. In the existing single-view run, 100MP images are commonly represented as `image_grid_thw=[[1, 64, 64]]`, which is clear evidence of strong downscaling into the model input budget.")
    lines.append("- MME-RS also contains large remote-sensing images up to 11500x7500 pixels, so it is a useful cross-benchmark check for the same tiling idea.")
    lines.append("- The first tiling ablation should focus on >16MP images and categories involving counting, object properties, and spatial relationships, because these are most likely to lose fine details after global resizing.")
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
