#!/usr/bin/env python3
"""Validate the 90-sample VRSBench internal evaluation package."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets" / "vrsbench_eval_v1" / "candidate_manifest.jsonl"
DEFAULT_REPAIRS = PROJECT_ROOT / "datasets" / "vrsbench_eval_v1" / "bbox_repairs.json"
EXPECTED_COUNTS = {"caption": 30, "vqa": 30, "grounding": 30}
VALID_SPLITS = {"train", "val", "test"}

REQUIRED_FIELDS = {
    "caption": {"id", "task", "image", "question", "answer", "source_split"},
    "vqa": {"id", "task", "image", "question", "answer", "question_type", "source_split"},
    "grounding": {
        "id", "task", "image", "question", "target", "bbox",
        "bbox_format", "bbox_space", "source_split",
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON格式错误: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} 必须是JSON对象")
            rows.append(value)
    return rows


def load_repairs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须是JSON对象")
    repairs = value.get("bbox_repairs", {})
    if not isinstance(repairs, dict):
        raise ValueError("bbox_repairs 必须是对象")
    return repairs


def bbox_error(bbox: Any) -> str | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return "bbox必须是4个数字组成的列表"
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bbox):
        return "bbox必须全部为数字"
    x1, y1, x2, y2 = map(float, bbox)
    if any(v < 0 or v > 100 for v in (x1, y1, x2, y2)):
        return "bbox坐标必须位于0-100"
    if not x1 < x2:
        return f"必须满足x1<x2，当前{x1}>={x2}"
    if not y1 < y2:
        return f"必须满足y1<y2，当前{y1}>={y2}"
    return None


def validate(
    manifest: Path,
    repairs_path: Path,
    project_root: Path,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not manifest.exists():
        return [f"找不到manifest: {manifest}"], warnings

    try:
        samples = load_jsonl(manifest)
        repairs = load_repairs(repairs_path)
    except ValueError as exc:
        return [str(exc)], warnings

    if len(samples) != 90:
        errors.append(f"总样本应为90，实际为{len(samples)}")

    ids = [str(row.get("id") or "") for row in samples]
    if any(not sample_id for sample_id in ids):
        errors.append("存在空id")
    duplicates = [k for k, v in Counter(ids).items() if k and v > 1]
    if duplicates:
        errors.append(f"存在重复id: {duplicates}")

    counts = Counter(str(row.get("task") or "") for row in samples)
    for task, expected in EXPECTED_COUNTS.items():
        if counts.get(task, 0) != expected:
            errors.append(f"{task}应为{expected}条，实际为{counts.get(task, 0)}条")
    unknown_tasks = sorted(set(counts) - set(EXPECTED_COUNTS))
    if unknown_tasks:
        errors.append(f"存在未知任务类型: {unknown_tasks}")

    sample_ids = set(ids)
    unused_repairs = sorted(set(repairs) - sample_ids)
    if unused_repairs:
        errors.append(f"bbox_repairs包含不存在的样本: {unused_repairs}")

    repaired_count = 0
    for row in samples:
        sample_id = str(row.get("id") or "<unknown>")
        task = str(row.get("task") or "")

        required = REQUIRED_FIELDS.get(task)
        if required:
            missing = sorted(field for field in required if field not in row)
            if missing:
                errors.append(f"{sample_id} 缺少字段: {missing}")

        split = row.get("source_split")
        if split not in VALID_SPLITS:
            errors.append(f"{sample_id} source_split无效: {split!r}")

        image_value = row.get("image")
        if not isinstance(image_value, str) or not image_value.strip():
            errors.append(f"{sample_id} image字段为空")
        else:
            image_path = Path(image_value)
            if image_path.is_absolute():
                errors.append(f"{sample_id} image必须使用项目相对路径: {image_value}")
            else:
                resolved = (project_root / image_path).resolve()
                try:
                    resolved.relative_to(project_root.resolve())
                except ValueError:
                    errors.append(f"{sample_id} image路径越出项目目录: {image_value}")
                else:
                    if not resolved.is_file():
                        errors.append(f"{sample_id} 图片不存在: {image_value}")

        if task != "grounding":
            continue

        if row.get("bbox_format") != "xyxy":
            errors.append(f"{sample_id} bbox_format必须为xyxy")
        if row.get("bbox_space") != "normalized_0_100":
            errors.append(f"{sample_id} bbox_space必须为normalized_0_100")

        raw_bbox = row.get("bbox")
        issue = bbox_error(raw_bbox)
        if issue is None:
            continue

        repair = repairs.get(sample_id)
        if not repair:
            errors.append(f"{sample_id} {issue}，且没有显式bbox修复记录")
            continue

        if repair.get("source_bbox") != raw_bbox:
            errors.append(f"{sample_id} bbox修复记录的source_bbox与manifest不一致")
            continue

        effective_bbox = repair.get("effective_bbox")
        effective_issue = bbox_error(effective_bbox)
        if effective_issue:
            errors.append(f"{sample_id} effective_bbox仍无效: {effective_issue}")
            continue

        repaired_count += 1
        warnings.append(
            f"{sample_id} 原始官方归一化bbox退化为{raw_bbox}；"
            f"内部评测使用显式修复{effective_bbox}"
        )

    if repaired_count:
        warnings.append(f"共{repaired_count}条Grounding使用显式bbox修复记录")

    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repairs", type=Path, default=DEFAULT_REPAIRS)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)

    errors, warnings = validate(
        args.manifest.resolve(),
        args.repairs.resolve(),
        args.project_root.resolve(),
    )

    print("=" * 60)
    print("VRSBench eval_v1 validation")
    print("=" * 60)
    for warning in warnings:
        print(f"[WARN] {warning}")

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        print(f"[FAIL] validation failed: {len(errors)} error(s)")
        return 1

    print("[PASS] 90 samples")
    print("[PASS] caption/vqa/grounding = 30/30/30")
    print("[PASS] IDs, fields, source_split and image paths")
    print("[PASS] Grounding bbox format/range/non-zero area")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
