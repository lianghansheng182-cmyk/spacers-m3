#!/usr/bin/env python3
"""Validate the materialized VRSBench LoRA v1 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[str]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_no}: JSON parse error: {exc}")
                continue
            if not isinstance(obj, dict):
                errors.append(f"{path.name}:{line_no}: row is not an object")
                continue
            rows.append((line_no, obj))
    return rows, errors


def image_basename(value: Any) -> str:
    return Path(str(value).replace("\\", "/")).name


def collect_eval_blacklist(eval_dirs: list[Path]) -> tuple[set[str], dict[str, list[str]]]:
    basenames: set[str] = set()
    hashes: dict[str, list[str]] = defaultdict(list)
    for eval_dir in eval_dirs:
        if not eval_dir.exists():
            continue
        for manifest in eval_dir.rglob("*.jsonl"):
            rows, _ = load_jsonl(manifest)
            for _, obj in rows:
                if obj.get("image"):
                    basenames.add(image_basename(obj["image"]))
        image_dir = eval_dir / "images"
        if image_dir.exists():
            for image in image_dir.rglob("*"):
                if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
                    basenames.add(image.name)
                    hashes[sha256(image)].append(str(image))
    return basenames, hashes


def check_bbox(obj: dict[str, Any], loc: str, errors: list[str]) -> None:
    if obj.get("bbox_format") != "xyxy":
        errors.append(f"{loc}: bbox_format must be xyxy")
    if obj.get("bbox_space") != "normalized_0_100":
        errors.append(f"{loc}: bbox_space must be normalized_0_100")
    bbox = obj.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        errors.append(f"{loc}: bbox must be a 4-number list")
        return
    if not all(isinstance(v, (int, float)) for v in bbox):
        errors.append(f"{loc}: bbox contains non-numeric values")
        return
    x1, y1, x2, y2 = bbox
    if x1 >= x2:
        errors.append(f"{loc}: x1 >= x2")
    if y1 >= y2:
        errors.append(f"{loc}: y1 >= y2")
    if any(v < 0 or v > 100 for v in bbox):
        errors.append(f"{loc}: bbox out of normalized 0..100 range")


def validate(dataset_dir: Path, repo_root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    train_path = dataset_dir / "train.jsonl"
    dev_path = dataset_dir / "train_dev.jsonl"
    image_dir = dataset_dir / "images"
    for required in [train_path, dev_path, image_dir, dataset_dir / "README.md", dataset_dir / "leakage_audit_report.json"]:
        if not required.exists():
            errors.append(f"missing required path: {required}")

    split_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    parse_errors: list[str] = []
    for split, path in [("train", train_path), ("dev", dev_path)]:
        if path.exists():
            rows, errs = load_jsonl(path)
            split_rows[split] = rows
            parse_errors.extend(errs)
        else:
            split_rows[split] = []
    errors.extend(parse_errors)

    task_counts: dict[str, Counter[str]] = {"train": Counter(), "dev": Counter()}
    sample_ids: set[str] = set()
    split_basenames: dict[str, set[str]] = {"train": set(), "dev": set()}
    referenced_images: set[Path] = set()
    required_fields = ["sample_id", "task", "image", "question", "answer"]

    for split, rows in split_rows.items():
        for line_no, obj in rows:
            loc = f"{split}.jsonl:{line_no}"
            for field in required_fields:
                if obj.get(field) in (None, ""):
                    errors.append(f"{loc}: missing {field}")
            sample_id = str(obj.get("sample_id", ""))
            if sample_id:
                if sample_id in sample_ids:
                    errors.append(f"{loc}: duplicate sample_id {sample_id}")
                sample_ids.add(sample_id)
            task = str(obj.get("task", ""))
            task_counts[split][task] += 1
            image_value = str(obj.get("image", ""))
            if image_value.startswith("/") or ":" in image_value:
                errors.append(f"{loc}: image path must be relative")
            if not image_value.startswith("images/"):
                errors.append(f"{loc}: image path must start with images/")
            image_path = dataset_dir / image_value
            if not image_path.exists():
                errors.append(f"{loc}: referenced image missing: {image_value}")
            else:
                referenced_images.add(image_path)
                split_basenames[split].add(image_path.name)
            if task == "grounding":
                for field in ["bbox", "bbox_format", "bbox_space"]:
                    if obj.get(field) in (None, ""):
                        errors.append(f"{loc}: missing {field}")
                check_bbox(obj, loc, errors)

    train_dev_name_overlap = sorted(split_basenames["train"] & split_basenames["dev"])
    if train_dev_name_overlap:
        errors.append(f"train/dev image basename overlap: {len(train_dev_name_overlap)}")

    image_hashes: dict[str, list[str]] = defaultdict(list)
    for image_path in sorted(referenced_images):
        image_hashes[sha256(image_path)].append(image_path.name)
    duplicate_hashes = {h: names for h, names in image_hashes.items() if len(names) > 1}
    if duplicate_hashes:
        errors.append(f"duplicate image hashes in LoRA dataset: {len(duplicate_hashes)}")

    eval_dirs = [repo_root / "datasets" / "mini_eval", repo_root / "datasets" / "vrsbench_eval_v1"]
    eval_basenames, eval_hashes = collect_eval_blacklist(eval_dirs)
    basename_leaks = sorted((split_basenames["train"] | split_basenames["dev"]) & eval_basenames)
    hash_leaks = sorted(set(image_hashes) & set(eval_hashes))
    if basename_leaks:
        errors.append(f"evaluation basename leakage: {len(basename_leaks)}")
    if hash_leaks:
        errors.append(f"evaluation hash leakage: {len(hash_leaks)}")

    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "train_count": len(split_rows["train"]),
        "dev_count": len(split_rows["dev"]),
        "task_counts": {split: dict(counts) for split, counts in task_counts.items()},
        "referenced_images": len(referenced_images),
        "image_hashes": len(image_hashes),
        "duplicate_image_hashes": len(duplicate_hashes),
        "train_dev_image_basename_overlap": len(train_dev_name_overlap),
        "eval_image_basenames": len(eval_basenames),
        "eval_image_hashes": len(eval_hashes),
        "evaluation_basename_leakage": len(basename_leaks),
        "evaluation_hash_leakage": len(hash_leaks),
        "basename_leak_examples": basename_leaks[:20],
        "hash_leak_examples": [
            {"hash": h, "lora": image_hashes[h], "eval": eval_hashes[h][:5]} for h in hash_leaks[:20]
        ],
        "status": "PASS" if not errors else "FAIL",
    }
    return summary, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/vrsbench_lora_v1"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true", help="Print JSON summary only.")
    args = parser.parse_args()

    summary, errors = validate(args.dataset_dir, args.repo_root)
    if args.json:
        print(json.dumps({"summary": summary, "errors": errors}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if errors:
            print("Errors:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
