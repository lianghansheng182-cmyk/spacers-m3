#!/usr/bin/env python3
"""Write per-sample Grounding error analysis for LoRA v1 vs baseline."""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def area(box: list[float] | None) -> float | None:
    if box is None:
        return None
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def center(box: list[float] | None) -> tuple[float, float] | None:
    if box is None:
        return None
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def classify(pred_box: list[float] | None, gt_box: list[float], parse_status: str, iou: float) -> str:
    if parse_status == "failed" or pred_box is None:
        return "parse失败"
    if any(value < 0 or value > 100 for value in pred_box):
        return "坐标格式错误"
    pred_area = area(pred_box)
    gt_area = area(gt_box)
    center_distance = distance(center(pred_box), center(gt_box))
    if iou < 0.1 and center_distance is not None and center_distance > 20:
        return "目标理解错误/多目标选错"
    if pred_area is not None and gt_area and pred_area > gt_area * 2.0:
        return "bbox过大"
    if pred_area is not None and gt_area and pred_area < gt_area * 0.5:
        return "bbox过小"
    if center_distance is not None and center_distance > 8:
        return "bbox整体偏移"
    if iou < 0.5:
        return "bbox边界不准"
    return "基本正确"


def escape_cell(value: Any) -> str:
    text = str(value).replace("\n", " ").replace("|", "/")
    return f"<code>{text}</code>"


def format_box(box: list[float] | None) -> str:
    if box is None:
        return "-"
    values = []
    for value in box:
        if float(value).is_integer():
            values.append(str(int(value)))
        else:
            values.append(f"{value:.3f}")
    return "[" + ",".join(values) + "]"


def table_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['sample_id']} | {escape_cell(row['base_prediction'])} | "
        f"{escape_cell(row['lora_prediction'])} | {format_box(row['gt'])} | "
        f"{row['base_iou']:.6f} | {row['lora_iou']:.6f} | {row['delta_iou']:+.6f} | "
        f"{'yes' if row['lora_parse_success'] else 'no'} | {row['category']} |"
    )


def short_item(row: dict[str, Any]) -> str:
    return (
        f"- {row['sample_id']}: delta={row['delta_iou']:+.6f}, "
        f"base={row['base_iou']:.6f}, lora={row['lora_iou']:.6f}, "
        f"category={row['category']}, pred={escape_cell(row['lora_prediction'])}"
    )


def main() -> int:
    base_metrics = load_json(REPO_ROOT / "results/vrsbench_eval_v1/qwen3_vl_2b_prompt_v1/metrics.json")
    lora_metrics = load_json(REPO_ROOT / "results/lora/qwen3_vl_2b_lora_v1_eval/vrsbench_eval_v1_prompt_v1/metrics.json")
    base_preds = load_jsonl(REPO_ROOT / "results/vrsbench_eval_v1/qwen3_vl_2b_prompt_v1/predictions.jsonl")
    lora_preds = load_jsonl(REPO_ROOT / "results/lora/qwen3_vl_2b_lora_v1_eval/vrsbench_eval_v1_prompt_v1/predictions.jsonl")

    base_by_id = {row["id"]: row for row in base_preds if row.get("task") == "grounding"}
    lora_by_id = {row["id"]: row for row in lora_preds if row.get("task") == "grounding"}
    base_pm = {row["id"]: row for row in base_metrics["per_sample"] if row["task"] == "grounding"}
    lora_pm = {row["id"]: row for row in lora_metrics["per_sample"] if row["task"] == "grounding"}

    rows: list[dict[str, Any]] = []
    for sample_id in sorted(lora_by_id):
        base_metric = base_pm[sample_id]
        lora_metric = lora_pm[sample_id]
        gt_box = [float(value) for value in lora_metric["target_bbox"]]
        base_iou = float(base_metric["iou"])
        lora_iou = float(lora_metric["iou"])
        rows.append(
            {
                "sample_id": sample_id,
                "question": lora_by_id[sample_id].get("question"),
                "base_prediction": base_by_id[sample_id].get("prediction"),
                "lora_prediction": lora_by_id[sample_id].get("prediction"),
                "gt": gt_box,
                "base_iou": base_iou,
                "lora_iou": lora_iou,
                "delta_iou": lora_iou - base_iou,
                "base_parse_success": base_metric.get("parsed_bbox") is not None,
                "lora_parse_success": lora_metric.get("parsed_bbox") is not None,
                "lora_parse_status": lora_metric.get("parse_status"),
                "lora_parse_error": lora_metric.get("parse_error"),
                "category": classify(
                    lora_metric.get("parsed_bbox"),
                    gt_box,
                    str(lora_metric.get("parse_status")),
                    lora_iou,
                ),
            }
        )

    improved = [row for row in rows if row["delta_iou"] > 0.02]
    declined = [row for row in rows if row["delta_iou"] < -0.02]
    unchanged = [row for row in rows if abs(row["delta_iou"]) <= 0.02]
    parse_failures = [row for row in rows if not row["lora_parse_success"]]
    category_counts = Counter(row["category"] for row in rows)

    lines = [
        "# LoRA v1 Grounding Error Analysis",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Grounding samples | {len(rows)} |",
        f"| Improved, delta IoU > 0.02 | {len(improved)} |",
        f"| Declined, delta IoU < -0.02 | {len(declined)} |",
        f"| Basically unchanged, abs delta <= 0.02 | {len(unchanged)} |",
        f"| LoRA parse failures | {len(parse_failures)} |",
        "",
        "Category counts use geometry-only heuristics from prediction bbox vs GT bbox; multi-target and target-understanding labels are inferred, not manually image-inspected.",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"| {category} | {count} |")

    lines.extend(
        [
            "",
            "## All Grounding Samples",
            "",
            "| sample_id | base prediction | LoRA prediction | GT | base IoU | LoRA IoU | delta IoU | LoRA parse | category |",
            "|---|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    lines.extend(table_row(row) for row in rows)

    lines.extend(["", "## Top 10 IoU Improvements", ""])
    lines.extend(short_item(row) for row in sorted(rows, key=lambda row: row["delta_iou"], reverse=True)[:10])
    lines.extend(["", "## Top 10 IoU Declines", ""])
    lines.extend(short_item(row) for row in sorted(rows, key=lambda row: row["delta_iou"])[:10])

    lines.extend(["", "## Parse Failure", ""])
    if parse_failures:
        for row in parse_failures:
            lines.append(
                f"- {row['sample_id']}: prediction={escape_cell(row['lora_prediction'])}, "
                f"reason={row['lora_parse_error']}, question={row['question']}"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Protocol Notes",
            "",
            "- LoRA v1 training assistant target for grounding is <code>{&lt;x1&gt;&lt;y1&gt;&lt;x2&gt;&lt;y2&gt;}</code> in normalized 0-100 xyxy space.",
            "- Prompt v1 asks for one bounding box in normalized 0-100 xyxy format: <code>{&lt;x1&gt;&lt;y1&gt;&lt;x2&gt;&lt;y2&gt;}</code>.",
            "- Evaluator parser accepts JSON arrays/objects and free-form strings containing four numbers, with 0-1, 0-100, or 0-1000 coordinate spaces normalized to 0-100.",
            "- Training, inference prompt, and evaluator are broadly compatible, but evaluator is more permissive than the training/prompt protocol.",
            "",
        ]
    )

    output_path = REPO_ROOT / "results/lora/grounding_v1_error_analysis.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "improved": len(improved),
                "declined": len(declined),
                "unchanged": len(unchanged),
                "parse_failures": len(parse_failures),
                "category_counts": dict(category_counts),
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
