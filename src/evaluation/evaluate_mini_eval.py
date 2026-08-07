#!/usr/bin/env python3
"""Evaluate VRSBench mini_eval predictions."""

from __future__ import annotations

import argparse
import json
import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass
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

BBOX_VALUE_KEYS = ("bbox", "bbox_2d", "bbox_3d", "box")
COORD_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
CODE_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class BBoxParseResult:
    raw_parsed_bbox: list[float] | None
    normalized_bbox_0_100: list[float] | None
    detected_bbox_space: str | None
    parse_status: str
    parse_error: str | None

    @property
    def parsed_bbox(self) -> list[float] | None:
        return self.normalized_bbox_0_100

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_parsed_bbox": self.raw_parsed_bbox,
            "normalized_bbox_0_100": self.normalized_bbox_0_100,
            "detected_bbox_space": self.detected_bbox_space,
            "parse_status": self.parse_status,
            "parse_error": self.parse_error,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mini_eval predictions.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument(
        "--error-cases",
        type=Path,
        default=None,
        help="JSONL output path for categorized error cases. Defaults to output-dir/error_cases.jsonl.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Summary JSON to synchronize with the new metrics. Defaults to output-dir/summary.json when present.",
    )
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    value = value.replace("\n", " ").replace("\t", " ")
    value = value.translate(str.maketrans({char: " " for char in string.punctuation}))
    tokens = [NUMBER_WORDS.get(token, token) for token in value.split()]
    return " ".join(tokens)


def _parse_json_candidates(text: str) -> list[Any]:
    candidates: list[Any] = []
    decoder = json.JSONDecoder()

    code_blocks = [match.group(1).strip() for match in CODE_BLOCK_PATTERN.finditer(text)]
    for block in code_blocks:
        try:
            candidates.append(json.loads(block))
        except json.JSONDecodeError:
            pass
    if candidates:
        return candidates

    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(value)
    return candidates


def _numbers_from_string(value: str) -> list[float] | None:
    numbers = [float(item) for item in COORD_PATTERN.findall(value)]
    if len(numbers) == 4:
        return numbers
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coords_from_value(value: Any) -> list[float] | None:
    if isinstance(value, str):
        return _numbers_from_string(value)

    if isinstance(value, dict):
        if all(key in value and _is_number(value[key]) for key in ("x1", "y1", "x2", "y2")):
            return [float(value[key]) for key in ("x1", "y1", "x2", "y2")]

        for key in BBOX_VALUE_KEYS:
            if key in value:
                coords = _coords_from_value(value[key])
                if coords is not None:
                    return coords

        for key, nested in value.items():
            if re.fullmatch(r"(?:bbox|box)_\d+", key) or re.fullmatch(
                r"(?:bbox|box)_\dd", key
            ):
                coords = _coords_from_value(nested)
                if coords is not None:
                    return coords

        for nested in value.values():
            coords = _coords_from_value(nested)
            if coords is not None:
                return coords
        return None

    if isinstance(value, list):
        if len(value) == 4 and all(_is_number(item) for item in value):
            return [float(item) for item in value]
        for nested in value:
            coords = _coords_from_value(nested)
            if coords is not None:
                return coords
    return None


def _normalize_bbox(raw_bbox: list[float]) -> BBoxParseResult:
    if len(raw_bbox) != 4:
        return BBoxParseResult(raw_bbox, None, None, "failed", "bbox must contain 4 values")
    if not all(_is_number(value) for value in raw_bbox):
        return BBoxParseResult(raw_bbox, None, None, "failed", "bbox values must be numeric")

    values = [float(value) for value in raw_bbox]
    if any(value < 0 for value in values):
        return BBoxParseResult(values, None, None, "failed", "bbox contains negative values")

    max_value = max(values)
    if max_value <= 1.0:
        normalized = [value * 100.0 for value in values]
        space = "normalized_0_1"
    elif max_value <= 100.0:
        normalized = values
        space = "normalized_0_100"
    elif max_value <= 1000.0:
        normalized = [value / 10.0 for value in values]
        space = "pixel_0_1000"
    else:
        return BBoxParseResult(
            values,
            None,
            None,
            "failed",
            "bbox coordinates exceed supported 0-1000 range",
        )

    x1, y1, x2, y2 = normalized
    reordered = False
    if x2 < x1:
        x1, x2 = x2, x1
        reordered = True
    if y2 < y1:
        y1, y2 = y2, y1
        reordered = True
    normalized = [round(value, 6) for value in (x1, y1, x2, y2)]

    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        return BBoxParseResult(values, None, space, "failed", "bbox has zero area")

    status = "ok_reordered" if reordered else "ok"
    return BBoxParseResult(values, normalized, space, status, None)


def parse_grounding_bbox(text: Any) -> BBoxParseResult:
    value = unicodedata.normalize("NFKC", str(text or ""))
    if not value.strip():
        return BBoxParseResult(None, None, None, "failed", "empty prediction")

    for candidate in _parse_json_candidates(value):
        coords = _coords_from_value(candidate)
        if coords is not None:
            return _normalize_bbox(coords)

    coords = _numbers_from_string(value)
    if coords is None:
        return BBoxParseResult(None, None, None, "failed", "no supported bbox found")
    return _normalize_bbox(coords)


def parse_bbox(text: Any) -> list[float] | None:
    """Backward-compatible helper returning the normalized box only."""
    return parse_grounding_bbox(text).parsed_bbox


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


def _error_case(
    prediction_row: dict[str, Any],
    sample: dict[str, Any],
    error_type: str,
    comment: str,
    parsed_bbox: list[float] | None = None,
    target_bbox: list[float] | None = None,
    iou: float | None = None,
) -> dict[str, Any]:
    return {
        "id": sample.get("id"),
        "task": sample.get("task"),
        "error_type": error_type,
        "prediction": prediction_row.get("prediction"),
        "ground_truth": sample.get("answer"),
        "parsed_bbox": parsed_bbox,
        "target_bbox": target_bbox,
        "iou": round(iou, 6) if iou is not None else None,
        "comment": comment,
    }


def evaluate_predictions(
    dataset_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    error_cases: list[dict[str, Any]] = []
    error_case_counts: Counter[str] = Counter()

    per_sample: list[dict[str, Any]] = []
    for sample in dataset_rows:
        prediction_row = predictions_by_id[sample["id"]]
        prediction = prediction_row.get("prediction", "")
        task = sample["task"]
        success = bool(prediction_row.get("success"))

        sample_metric: dict[str, Any] = {
            "id": sample["id"],
            "task": task,
            "success": success,
        }
        if not success:
            error_type = "inference_failure"
            error_case_counts[error_type] += 1
            error_cases.append(
                _error_case(
                    prediction_row,
                    sample,
                    error_type,
                    str(prediction_row.get("error") or "inference failed"),
                )
            )

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
            if success and not exact_match:
                error_type = "vqa_wrong"
                error_case_counts[error_type] += 1
                error_cases.append(
                    _error_case(
                        prediction_row,
                        sample,
                        error_type,
                        "normalized exact match is false",
                    )
                )
        elif task == "grounding":
            grounding_total += 1
            parse_result = parse_grounding_bbox(prediction)
            parsed_bbox = parse_result.normalized_bbox_0_100
            target_bbox = [float(value) for value in sample["bbox"]]
            iou = 0.0
            if parsed_bbox is not None:
                grounding_parsed += 1
                iou = box_iou(parsed_bbox, target_bbox)
            ious.append(iou)
            sample_metric.update(
                {
                    "parsed_bbox": parsed_bbox,
                    "target_bbox": sample["bbox"],
                    "iou": round(iou, 6),
                    "acc_at_0_5": iou >= 0.5,
                    "acc_at_0_7": iou >= 0.7,
                    **parse_result.as_dict(),
                }
            )
            if success and parsed_bbox is None:
                error_type = "grounding_parse_failure"
                error_case_counts[error_type] += 1
                error_cases.append(
                    _error_case(
                        prediction_row,
                        sample,
                        error_type,
                        parse_result.parse_error or "bbox parsing failed",
                        parsed_bbox=None,
                        target_bbox=sample["bbox"],
                        iou=None,
                    )
                )
            elif success and iou < 0.5:
                error_type = "grounding_low_iou"
                error_case_counts[error_type] += 1
                error_cases.append(
                    _error_case(
                        prediction_row,
                        sample,
                        error_type,
                        "grounding IoU is below 0.5",
                        parsed_bbox=parsed_bbox,
                        target_bbox=sample["bbox"],
                        iou=iou,
                    )
                )
        per_sample.append(sample_metric)

    metrics = {
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
        "error_case_count": len(error_cases),
        "error_case_counts": dict(sorted(error_case_counts.items())),
        "per_sample": per_sample,
    }
    return metrics, error_cases


def evaluate_rows(
    dataset_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics, _ = evaluate_predictions(dataset_rows, prediction_rows)
    return metrics


def evaluate_files(
    predictions_path: Path,
    dataset_path: Path,
    output_path: Path | None = None,
    error_cases_path: Path | None = None,
) -> dict[str, Any]:
    dataset_rows = load_jsonl(dataset_path)
    prediction_rows = load_jsonl(predictions_path)
    metrics, error_cases = evaluate_predictions(dataset_rows, prediction_rows)
    if output_path is not None:
        write_json(output_path, metrics)
        if error_cases_path is None:
            error_cases_path = output_path.parent / "error_cases.jsonl"
    if error_cases_path is not None:
        write_jsonl(error_cases_path, error_cases)
    return metrics


def sync_summary(summary_path: Path, metrics: dict[str, Any]) -> None:
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"] = metrics
    summary["sample_count"] = metrics["sample_count"]
    summary["success_count"] = metrics["success_count"]
    summary["failure_count"] = metrics["failure_count"]
    summary["error_case_count"] = metrics["error_case_count"]
    summary["error_case_counts"] = metrics["error_case_counts"]
    write_json(summary_path, summary)


def main() -> int:
    args = parse_args()
    output_path = args.output.expanduser().resolve()
    error_cases_path = (
        args.error_cases.expanduser().resolve()
        if args.error_cases is not None
        else output_path.parent / "error_cases.jsonl"
    )
    metrics = evaluate_files(
        predictions_path=args.predictions.expanduser().resolve(),
        dataset_path=args.dataset.expanduser().resolve(),
        output_path=output_path,
        error_cases_path=error_cases_path,
    )

    summary_path = (
        args.summary.expanduser().resolve()
        if args.summary is not None
        else output_path.parent / "summary.json"
    )
    sync_summary(summary_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
