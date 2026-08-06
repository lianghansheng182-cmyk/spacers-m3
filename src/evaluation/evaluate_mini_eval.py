#!/usr/bin/env python3
"""Evaluate VRSBench mini_eval predictions."""

from __future__ import annotations

import argparse
import json
import re
import string
import unicodedata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "datasets" / "mini_eval" / "mini_eval.jsonl"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "vrsbench_mini" / "qwen3_vl_2b_zero_shot"
)
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIR / "predictions.jsonl"
DEFAULT_METRICS = DEFAULT_OUTPUT_DIR / "metrics.json"

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mini_eval predictions.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_METRICS)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_answer(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    value = value.replace("\n", " ").replace("\t", " ")
    value = value.translate(str.maketrans({char: " " for char in string.punctuation}))
    tokens = [NUMBER_WORDS.get(token, token) for token in value.split()]
    return " ".join(tokens)


def parse_bbox(text: Any) -> list[float] | None:
    value = unicodedata.normalize("NFKC", str(text or ""))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
    if len(numbers) < 4:
        return None

    coords = [float(item) for item in numbers[:4]]
    if all(0.0 <= coord <= 1.0 for coord in coords):
        coords = [coord * 100.0 for coord in coords]

    x1, y1, x2, y2 = coords
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    coords = [max(0.0, min(100.0, coord)) for coord in (x1, y1, x2, y2)]
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        return None
    return coords


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def evaluate_rows(
    dataset_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_by_id = {row["id"]: row for row in dataset_rows}
    predictions_by_id = {row.get("id"): row for row in prediction_rows}
    missing_ids = sorted(set(dataset_by_id) - set(predictions_by_id))
    extra_ids = sorted(set(predictions_by_id) - set(dataset_by_id))
    if missing_ids or extra_ids:
        raise ValueError(f"prediction ids mismatch: missing={missing_ids}, extra={extra_ids}")

    caption_count = 0
    vqa_total = 0
    vqa_correct = 0
    grounding_total = 0
    grounding_parsed = 0
    ious: list[float] = []

    per_sample: list[dict[str, Any]] = []
    for sample in dataset_rows:
        prediction_row = predictions_by_id[sample["id"]]
        prediction = prediction_row.get("prediction", "")
        task = sample["task"]

        sample_metric: dict[str, Any] = {
            "id": sample["id"],
            "task": task,
            "success": bool(prediction_row.get("success")),
        }
        if task == "caption":
            caption_count += 1
        elif task == "vqa":
            vqa_total += 1
            normalized_prediction = normalize_answer(prediction)
            normalized_answer = normalize_answer(sample.get("answer"))
            exact_match = normalized_prediction == normalized_answer
            vqa_correct += int(exact_match)
            sample_metric.update(
                {
                    "normalized_prediction": normalized_prediction,
                    "normalized_answer": normalized_answer,
                    "exact_match": exact_match,
                }
            )
        elif task == "grounding":
            grounding_total += 1
            parsed_bbox = parse_bbox(prediction)
            iou = 0.0
            if parsed_bbox is not None:
                grounding_parsed += 1
                iou = box_iou(parsed_bbox, [float(value) for value in sample["bbox"]])
            ious.append(iou)
            sample_metric.update(
                {
                    "parsed_bbox": parsed_bbox,
                    "target_bbox": sample["bbox"],
                    "iou": round(iou, 6),
                    "acc_at_0_5": iou >= 0.5,
                    "acc_at_0_7": iou >= 0.7,
                }
            )
        per_sample.append(sample_metric)

    return {
        "sample_count": len(dataset_rows),
        "prediction_count": len(prediction_rows),
        "success_count": sum(1 for row in prediction_rows if row.get("success")),
        "failure_count": sum(1 for row in prediction_rows if not row.get("success")),
        "caption": {
            "count": caption_count,
            "metric": None,
            "note": "Caption predictions are saved only; no automatic caption metric is reported for this run.",
        },
        "vqa": {
            "count": vqa_total,
            "correct": vqa_correct,
            "accuracy": round(vqa_correct / vqa_total, 6) if vqa_total else 0.0,
            "normalization": "NFKC, lowercase, punctuation-to-space, whitespace collapse, number words 0-10",
        },
        "grounding": {
            "count": grounding_total,
            "parsed_count": grounding_parsed,
            "mean_iou": round(sum(ious) / grounding_total, 6) if grounding_total else 0.0,
            "acc_at_0_5": round(
                sum(1 for iou in ious if iou >= 0.5) / grounding_total,
                6,
            )
            if grounding_total
            else 0.0,
            "acc_at_0_7": round(
                sum(1 for iou in ious if iou >= 0.7) / grounding_total,
                6,
            )
            if grounding_total
            else 0.0,
        },
        "per_sample": per_sample,
    }


def evaluate_files(
    predictions_path: Path,
    dataset_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    dataset_rows = load_jsonl(dataset_path)
    prediction_rows = load_jsonl(predictions_path)
    metrics = evaluate_rows(dataset_rows, prediction_rows)
    if output_path is not None:
        write_json(output_path, metrics)
    return metrics


def main() -> int:
    args = parse_args()
    metrics = evaluate_files(
        predictions_path=args.predictions.expanduser().resolve(),
        dataset_path=args.dataset.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
