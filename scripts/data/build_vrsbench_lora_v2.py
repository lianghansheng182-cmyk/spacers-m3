#!/usr/bin/env python3
"""Build the VRSBench LoRA v2 dataset from the validated v1 materialization."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_bbox_answer(bbox: list[Any]) -> str:
    values = [int(round(float(value))) for value in bbox]
    return "{" + "".join(f"<{value}>" for value in values) + "}"


def extract_target(question: str) -> str:
    match = re.search(r"<p>(.*?)</p>", question)
    if match:
        return " ".join(match.group(1).split())
    text = " ".join(question.replace("\n", " ").split())
    text = re.sub(r"^the location of\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+is$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^请找出图像中的目标[:：]\s*", "", text)
    text = re.sub(r"[，,]\s*并输出.*$", "", text)
    return text.strip() or question


def prompt_v1_style_question(source: dict[str, Any]) -> str:
    target = extract_target(str(source["question"]))
    return (
        f"请根据目标描述定位图像中的唯一目标：{target}。"
        "必须利用目标描述中的位置、属性和相对关系，只选择一个目标。"
        "只返回一个归一化0-100的xyxy边界框，格式严格为{<x1><y1><x2><y2>}。"
        "不要输出类别名、解释、Markdown或多余数字。"
    )


def with_source(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("source_sample_id", row.get("source_sample_id") or row["sample_id"])
    if out.get("task") == "grounding":
        out["answer"] = canonical_bbox_answer(out["bbox"])
        out["bbox_format"] = "xyxy"
        out["bbox_space"] = "normalized_0_100"
    return out


def augment_grounding(row: dict[str, Any]) -> dict[str, Any]:
    out = with_source(row)
    out["sample_id"] = f"{row['sample_id']}_promptv1_zh"
    out["source_sample_id"] = row["sample_id"]
    out["question"] = prompt_v1_style_question(row)
    out["augmentation"] = "grounding_prompt_v1_zh"
    out["answer"] = canonical_bbox_answer(out["bbox"])
    return out


def copy_referenced_images(rows: list[dict[str, Any]], source_dir: Path, target_dir: Path) -> int:
    image_values = sorted({str(row["image"]) for row in rows})
    for image_value in image_values:
        src = source_dir / image_value
        dst = target_dir / image_value
        if not src.is_file():
            raise FileNotFoundError(f"missing v1 image: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
    return len(image_values)


def build(source_dir: Path, target_dir: Path) -> dict[str, Any]:
    train_v1 = load_jsonl(source_dir / "train.jsonl")
    dev_v1 = load_jsonl(source_dir / "train_dev.jsonl")

    train_rows: list[dict[str, Any]] = []
    augmented_count = 0
    for row in train_v1:
        normalized = with_source(row)
        train_rows.append(normalized)
        if row.get("task") == "grounding":
            train_rows.append(augment_grounding(row))
            augmented_count += 1

    dev_rows = [with_source(row) for row in dev_v1]
    write_jsonl(target_dir / "train.jsonl", train_rows)
    write_jsonl(target_dir / "train_dev.jsonl", dev_rows)

    image_count = copy_referenced_images(train_rows + dev_rows, source_dir, target_dir)
    if (source_dir / "leakage_audit_report.json").exists():
        shutil.copy2(source_dir / "leakage_audit_report.json", target_dir / "leakage_audit_report.json")
    else:
        write_json(target_dir / "leakage_audit_report.json", {"source": str(source_dir), "status": "copied_from_v1_missing"})

    readme = f"""# VRSBench LoRA v2

Generated: {datetime.now().isoformat(timespec="seconds")}

Source dataset: `{source_dir}`

This dataset keeps v1 caption and VQA samples unchanged and adds one Prompt-v1-style
Chinese grounding question for each v1 train grounding sample. Grounding assistant
answers are canonicalized to `{{<x1><y1><x2><y2>}}` in normalized 0-100 xyxy space.

Evaluation datasets were not sampled or copied into this dataset.
"""
    (target_dir / "README.md").write_text(readme, encoding="utf-8")

    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "train_task_counts": dict(Counter(row["task"] for row in train_rows)),
        "dev_task_counts": dict(Counter(row["task"] for row in dev_rows)),
        "augmented_grounding_train_samples": augmented_count,
        "unique_referenced_images": image_count,
        "augmentation": {
            "grounding_prompt_v1_zh_per_source": 1,
            "assistant_format": "{<x1><y1><x2><y2>}",
            "bbox_format": "xyxy",
            "bbox_space": "normalized_0_100",
        },
    }
    write_json(target_dir / "build_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("datasets/vrsbench_lora_v1"))
    parser.add_argument("--target-dir", type=Path, default=Path("datasets/vrsbench_lora_v2"))
    args = parser.parse_args()

    summary = build(args.source_dir, args.target_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
