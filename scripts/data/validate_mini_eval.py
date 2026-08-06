#!/usr/bin/env python3
"""Validate the VRSBench mini_eval JSONL file and referenced images."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_TASKS = {"caption", "vqa", "grounding"}
EXPECTED_COUNTS = {"caption": 10, "vqa": 10, "grounding": 10}

REQUIRED_FIELDS = {
    "caption": {"id", "task", "image", "question", "answer", "source_split"},
    "vqa": {
        "id",
        "task",
        "image",
        "question",
        "answer",
        "question_type",
        "source_split",
    },
    "grounding": {
        "id",
        "task",
        "image",
        "question",
        "answer",
        "target",
        "bbox",
        "bbox_format",
        "bbox_space",
        "source_split",
    },
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Validate datasets/mini_eval/mini_eval.jsonl."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root used to resolve relative image paths.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=repo_root / "datasets" / "mini_eval" / "mini_eval.jsonl",
        help="Path to the mini_eval JSONL file.",
    )
    return parser.parse_args()


def load_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        text = jsonl_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{jsonl_path} is not valid UTF-8: {exc}") from exc

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"line {line_number}: empty lines are not allowed")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number}: JSON value must be an object")
        records.append(record)
    return records


def validate_bbox(record: dict[str, Any], line_number: int) -> None:
    if record.get("bbox_format") != "xyxy":
        raise ValueError(f"line {line_number}: bbox_format must be xyxy")
    if record.get("bbox_space") != "normalized_0_100":
        raise ValueError(f"line {line_number}: bbox_space must be normalized_0_100")

    bbox = record.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"line {line_number}: bbox must be a list of 4 numbers")
    if not all(isinstance(value, (int, float)) for value in bbox):
        raise ValueError(f"line {line_number}: bbox values must be numeric")

    x1, y1, x2, y2 = bbox
    if not (0 <= x1 <= x2 <= 100 and 0 <= y1 <= y2 <= 100):
        raise ValueError(
            f"line {line_number}: normalized_0_100 bbox must satisfy "
            "0 <= x1 <= x2 <= 100 and 0 <= y1 <= y2 <= 100"
        )


def validate_records(records: list[dict[str, Any]], repo_root: Path) -> Counter[str]:
    seen_ids: set[str] = set()
    counts: Counter[str] = Counter()

    for line_number, record in enumerate(records, start=1):
        task = record.get("task")
        if task not in ALLOWED_TASKS:
            raise ValueError(f"line {line_number}: unsupported task {task!r}")

        missing = sorted(REQUIRED_FIELDS[task] - record.keys())
        if missing:
            raise ValueError(f"line {line_number}: missing fields {missing}")
        for field in sorted(REQUIRED_FIELDS[task] - {"bbox"}):
            if record[field] in ("", None):
                raise ValueError(f"line {line_number}: field {field!r} must be non-empty")

        sample_id = record["id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"line {line_number}: id must be a non-empty string")
        if sample_id in seen_ids:
            raise ValueError(f"line {line_number}: duplicate id {sample_id!r}")
        seen_ids.add(sample_id)

        image = record["image"]
        if not isinstance(image, str) or not image:
            raise ValueError(f"line {line_number}: image must be a non-empty string")
        image_path = Path(image)
        if not image_path.is_absolute():
            image_path = repo_root / image_path
        if not image_path.is_file():
            raise ValueError(f"line {line_number}: image not found: {image}")

        if task == "grounding":
            validate_bbox(record, line_number)

        counts[task] += 1

    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(
            f"expected counts {EXPECTED_COUNTS}, got {dict(sorted(counts.items()))}"
        )
    return counts


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    jsonl_path = args.jsonl.resolve()
    records = load_jsonl(jsonl_path)
    counts = validate_records(records, repo_root)

    print(f"jsonl: {jsonl_path}")
    print(f"total: {len(records)}")
    for task in EXPECTED_COUNTS:
        print(f"{task}: {counts[task]}")
    print("status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
