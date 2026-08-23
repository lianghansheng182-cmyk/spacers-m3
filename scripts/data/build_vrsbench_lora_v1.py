#!/usr/bin/env python3
"""Build a leak-free VRSBench LoRA v1 dataset from official train data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TASK_TARGETS = {"caption": 3000, "vqa": 3000, "grounding": 3000}
DEV_TARGETS = {"caption": 200, "vqa": 200, "grounding": 200}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def image_basename(value: Any) -> str:
    return Path(str(value).replace("\\", "/")).name


def clean_question(value: str, task_tag: str) -> str:
    text = value.replace("<image>", "").replace("\n", " ").strip()
    return text.replace(task_tag, "").strip()


def parse_bbox(value: str) -> list[int] | None:
    nums = [int(x) for x in re.findall(r"-?\d+", value)]
    if len(nums) != 4:
        return None
    x1, y1, x2, y2 = nums
    if x1 >= x2 or y1 >= y2:
        return None
    if any(v < 0 or v > 100 for v in nums):
        return None
    return nums


def parse_official_train(path: Path) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    field_keys = Counter()
    conversation_shapes = Counter()
    bbox_values: list[int] = []
    invalid_grounding = 0

    for row_index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        field_keys.update(item.keys())
        image = image_basename(item.get("image", ""))
        conversations = item.get("conversations", [])
        if not image or not isinstance(conversations, list):
            continue
        human = ""
        assistant = ""
        for conv in conversations:
            if not isinstance(conv, dict):
                continue
            conversation_shapes.update(conv.keys())
            if conv.get("from") == "human":
                human = str(conv.get("value", ""))
            elif conv.get("from") == "gpt":
                assistant = str(conv.get("value", ""))
        if "[caption]" in human:
            sample = {
                "source_index": row_index,
                "task": "caption",
                "image": image,
                "question": clean_question(human, "[caption]"),
                "answer": assistant.strip(),
            }
            if sample["question"] and sample["answer"]:
                groups[image]["caption"].append(sample)
        elif "[vqa]" in human:
            sample = {
                "source_index": row_index,
                "task": "vqa",
                "image": image,
                "question": clean_question(human, "[vqa]"),
                "answer": assistant.strip(),
            }
            if sample["question"] and sample["answer"]:
                groups[image]["vqa"].append(sample)
        elif "[refer]" in human:
            bbox = parse_bbox(assistant)
            if bbox is None:
                invalid_grounding += 1
                continue
            bbox_values.extend(bbox)
            sample = {
                "source_index": row_index,
                "task": "grounding",
                "image": image,
                "question": clean_question(human, "[refer]"),
                "answer": assistant.strip(),
                "bbox": bbox,
                "bbox_format": "xyxy",
                "bbox_space": "normalized_0_100",
            }
            if sample["question"] and sample["answer"]:
                groups[image]["grounding"].append(sample)

    task_counts = Counter()
    for task_map in groups.values():
        for task, rows in task_map.items():
            task_counts[task] += len(rows)
    structure = {
        "official_rows": len(data),
        "unique_images": len(groups),
        "task_counts": dict(task_counts),
        "top_level_keys": dict(field_keys),
        "conversation_keys": dict(conversation_shapes),
        "invalid_grounding_rows": invalid_grounding,
        "bbox_min": min(bbox_values) if bbox_values else None,
        "bbox_max": max(bbox_values) if bbox_values else None,
    }
    return groups, structure


def analyze_annotations_zip(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False}
    key_counts: Counter[str] = Counter()
    object_key_counts: Counter[str] = Counter()
    qa_key_counts: Counter[str] = Counter()
    obj_coord_values: list[float] = []
    obj_corner_values: list[float] = []
    json_count = 0
    examples: list[dict[str, Any]] = []
    with ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.startswith("Annotations_train/") or not info.filename.endswith(".json"):
                continue
            json_count += 1
            obj = json.loads(archive.read(info).decode("utf-8"))
            key_counts.update(obj.keys())
            if len(examples) < 2:
                examples.append({"file": info.filename, "keys": list(obj.keys())})
            for item in obj.get("objects", []):
                if isinstance(item, dict):
                    object_key_counts.update(item.keys())
                    for value in item.get("obj_coord", []):
                        if isinstance(value, (int, float)):
                            obj_coord_values.append(float(value))
                    for value in item.get("obj_corner", []):
                        if isinstance(value, (int, float)):
                            obj_corner_values.append(float(value))
            for qa in obj.get("qa_pairs", []):
                if isinstance(qa, dict):
                    qa_key_counts.update(qa.keys())
    return {
        "present": True,
        "json_files": json_count,
        "top_level_keys": dict(key_counts),
        "object_keys": dict(object_key_counts),
        "qa_pair_keys": dict(qa_key_counts),
        "obj_coord_min": min(obj_coord_values) if obj_coord_values else None,
        "obj_coord_max": max(obj_coord_values) if obj_coord_values else None,
        "obj_corner_min": min(obj_corner_values) if obj_corner_values else None,
        "obj_corner_max": max(obj_corner_values) if obj_corner_values else None,
        "examples": examples,
    }


def collect_eval_blacklist(repo_root: Path) -> dict[str, Any]:
    eval_dirs = [repo_root / "datasets" / "mini_eval", repo_root / "datasets" / "vrsbench_eval_v1"]
    ids: set[str] = set()
    basenames: set[str] = set()
    hashes: dict[str, list[str]] = defaultdict(list)
    manifest_rows = 0
    for eval_dir in eval_dirs:
        if not eval_dir.exists():
            continue
        for manifest in eval_dir.rglob("*.jsonl"):
            rows = load_jsonl(manifest)
            manifest_rows += len(rows)
            for row in rows:
                sample_id = row.get("sample_id") or row.get("id")
                if sample_id:
                    ids.add(str(sample_id))
                if row.get("image"):
                    basenames.add(image_basename(row["image"]))
        image_dir = eval_dir / "images"
        if image_dir.exists():
            for image in image_dir.rglob("*"):
                if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
                    basenames.add(image.name)
                    hashes[sha256(image)].append(str(image))
    return {
        "ids": sorted(ids),
        "basenames": sorted(basenames),
        "hashes": dict(sorted(hashes.items())),
        "manifest_rows": manifest_rows,
    }


def index_images(image_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicate_names: dict[str, list[str]] = defaultdict(list)
    for image in image_root.rglob("*"):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
            continue
        if image.name in index:
            duplicate_names[image.name].append(str(image))
        else:
            index[image.name] = image
    if duplicate_names:
        raise RuntimeError(f"duplicate image basenames in source image root: {len(duplicate_names)}")
    return index


def sample_from_images(
    image_names: list[str],
    groups: dict[str, dict[str, list[dict[str, Any]]]],
    targets: dict[str, int],
    split: str,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    per_task_seen: Counter[str] = Counter()
    for task, target in targets.items():
        task_rows: list[dict[str, Any]] = []
        for image_name in image_names:
            task_rows.extend(groups[image_name].get(task, []))
        if len(task_rows) < target:
            raise RuntimeError(f"{split} has only {len(task_rows)} {task} samples, target is {target}")
        for task_seq, sample in enumerate(task_rows[:target], 1):
            out = {k: v for k, v in sample.items() if k != "source_index"}
            out["sample_id"] = f"{split}_{task}_{task_seq:06d}"
            out["image"] = f"images/{out['image']}"
            samples.append(out)
            per_task_seen[task] += 1
    samples.sort(key=lambda obj: (obj["task"], obj["sample_id"]))
    return samples


def choose_image_splits(
    groups: dict[str, dict[str, list[dict[str, Any]]]],
    image_index: dict[str, Path],
    eval_basenames: set[str],
    eval_hashes: set[str],
    seed: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    rng = random.Random(seed)
    candidates = [
        name
        for name, task_map in groups.items()
        if name in image_index
        and name not in eval_basenames
        and all(task_map.get(task) for task in TASK_TARGETS)
    ]
    rng.shuffle(candidates)

    selected_hashes: set[str] = set()
    rejected = Counter()
    train_images: list[str] = []
    dev_images: list[str] = []

    def safe_add(name: str) -> bool:
        image_hash = sha256(image_index[name])
        if image_hash in eval_hashes:
            rejected["eval_hash"] += 1
            return False
        if image_hash in selected_hashes:
            rejected["duplicate_hash"] += 1
            return False
        selected_hashes.add(image_hash)
        return True

    for name in candidates:
        if len(train_images) >= TASK_TARGETS["caption"]:
            break
        if safe_add(name):
            train_images.append(name)
    for name in candidates:
        if name in set(train_images):
            continue
        if len(dev_images) >= DEV_TARGETS["caption"]:
            break
        if safe_add(name):
            dev_images.append(name)

    if len(train_images) < TASK_TARGETS["caption"]:
        raise RuntimeError(f"only selected {len(train_images)} train images")
    if len(dev_images) < DEV_TARGETS["caption"]:
        raise RuntimeError(f"only selected {len(dev_images)} dev images")

    stats = {
        "candidate_images_with_all_tasks": len(candidates),
        "selected_train_images": len(train_images),
        "selected_dev_images": len(dev_images),
        "rejected": dict(rejected),
    }
    return train_images, dev_images, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_structure_report(path: Path, source_info: dict[str, Any], structure: dict[str, Any], annotations: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# VRSBench Official Train Structure

Generated: {datetime.now().isoformat(timespec="seconds")}

## Official Data Paths

| Item | Path | Count/Size |
|---|---|---:|
| Images_train | `{source_info['image_root']}` | {source_info['source_image_count']} images |
| VRSBench_train.json | `{source_info['train_json']}` | {structure['official_rows']} rows |
| Annotations_train | `{source_info['annotations_zip']}` | {annotations.get('json_files', 0)} json files |

## VRSBench_train.json

| Metric | Count |
|---|---:|
| Unique images | {structure['unique_images']} |
| Caption samples | {structure['task_counts'].get('caption', 0)} |
| VQA samples | {structure['task_counts'].get('vqa', 0)} |
| Grounding samples | {structure['task_counts'].get('grounding', 0)} |
| Invalid grounding rows skipped | {structure['invalid_grounding_rows']} |

Top-level keys: `{structure['top_level_keys']}`

Conversation keys: `{structure['conversation_keys']}`

Grounding bbox is parsed from the assistant value in `[refer]` rows. Parsed range: `{structure['bbox_min']}` to `{structure['bbox_max']}`.

## Annotations_train.zip

| Metric | Value |
|---|---:|
| Present | {annotations.get('present')} |
| JSON files | {annotations.get('json_files', 0)} |
| obj_coord range | {annotations.get('obj_coord_min')} to {annotations.get('obj_coord_max')} |
| obj_corner range | {annotations.get('obj_corner_min')} to {annotations.get('obj_corner_max')} |

Annotation top-level keys: `{annotations.get('top_level_keys')}`

Object keys: `{annotations.get('object_keys')}`

QA pair keys: `{annotations.get('qa_pair_keys')}`

Notes:

- `VRSBench_train.json` is the final source for LoRA manifest construction.
- `[caption]`, `[vqa]`, and `[refer]` task tags are read from the human conversation text.
- `[refer]` assistant values are normalized 0-100 xyxy boxes such as `{{<45><45><59><59>}}`.
"""
    path.write_text(text, encoding="utf-8")


def write_readme(path: Path, audit: dict[str, Any]) -> None:
    text = f"""# VRSBench LoRA v1

Leak-free LoRA training subset rebuilt from the official VRSBench train split.

## Source

- Official manifest: `{audit['official_train_json']}`
- Official image root: `{audit['official_image_root']}`
- Evaluation images from `datasets/mini_eval` and `datasets/vrsbench_eval_v1` were excluded at image level.

## Counts

| Split | Caption | VQA | Grounding | Total |
|---|---:|---:|---:|---:|
| train | {audit['task_counts']['train'].get('caption', 0)} | {audit['task_counts']['train'].get('vqa', 0)} | {audit['task_counts']['train'].get('grounding', 0)} | {audit['train_count']} |
| train_dev | {audit['task_counts']['dev'].get('caption', 0)} | {audit['task_counts']['dev'].get('vqa', 0)} | {audit['task_counts']['dev'].get('grounding', 0)} | {audit['dev_count']} |

## Format

All rows contain `sample_id`, `task`, `image`, `question`, and `answer`.

Grounding rows also contain `bbox`, `bbox_format=\"xyxy\"`, and `bbox_space=\"normalized_0_100\"`.

The `image` field is relative and starts with `images/`.
"""
    path.write_text(text, encoding="utf-8")


def write_final_audit(path: Path, audit: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# VRSBench LoRA v1 Final Audit

Generated: {datetime.now().isoformat(timespec="seconds")}

## Verdict

Ready for training: **{'YES' if audit['ready'] else 'NO'}**

No LoRA training was started.

## Counts

| Metric | Count |
|---|---:|
| Official train images | {audit['official_unique_images']} |
| Source image files | {audit['source_image_count']} |
| Evaluation blacklist images | {audit['evaluation_blacklist_images']} |
| Train samples | {audit['train_count']} |
| Train-dev samples | {audit['dev_count']} |
| Final copied images | {audit['final_image_count']} |

| Split | Caption | VQA | Grounding |
|---|---:|---:|---:|
| train | {audit['task_counts']['train'].get('caption', 0)} | {audit['task_counts']['train'].get('vqa', 0)} | {audit['task_counts']['train'].get('grounding', 0)} |
| train-dev | {audit['task_counts']['dev'].get('caption', 0)} | {audit['task_counts']['dev'].get('vqa', 0)} | {audit['task_counts']['dev'].get('grounding', 0)} |

## Leakage

| Check | Count |
|---|---:|
| basename overlap with eval | {audit['validator_summary'].get('evaluation_basename_leakage')} |
| hash overlap with eval | {audit['validator_summary'].get('evaluation_hash_leakage')} |
| train/dev image overlap | {audit['validator_summary'].get('train_dev_image_basename_overlap')} |
| duplicate image hashes in final set | {audit['validator_summary'].get('duplicate_image_hashes')} |

## Validator

Validator command:

```bash
python scripts/data/validate_vrsbench_lora_v1.py --dataset-dir datasets/vrsbench_lora_v1 --repo-root . --json
```

Validator status: **{audit['validator_summary'].get('status')}**

## Build Notes

- Old `lora_train_v1` was not used as a manifest source.
- Official train data was split at image level before sampling rows.
- Images present in `mini_eval` or `vrsbench_eval_v1` were blacklisted by basename and SHA-256 hash.
- Final image paths are relative `images/*.png` paths.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--official-train-json", type=Path, default=Path("data/raw/vrsbench_official/VRSBench_train.json"))
    parser.add_argument("--annotations-zip", type=Path, default=Path("data/raw/vrsbench_official/Annotations_train.zip"))
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/vrsbench_lora_v1"))
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    train_json = args.official_train_json if args.official_train_json.is_absolute() else repo_root / args.official_train_json
    annotations_zip = args.annotations_zip if args.annotations_zip.is_absolute() else repo_root / args.annotations_zip
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    image_root = args.image_root

    groups, structure = parse_official_train(train_json)
    annotations = analyze_annotations_zip(annotations_zip)
    image_index = index_images(image_root)

    results_dir = repo_root / "results" / "data_audit"
    source_info = {
        "image_root": str(image_root),
        "train_json": str(train_json),
        "annotations_zip": str(annotations_zip),
        "source_image_count": len(image_index),
    }
    write_structure_report(results_dir / "vrsbench_official_train_structure.md", source_info, structure, annotations)

    blacklist = collect_eval_blacklist(repo_root)
    blacklist_path = results_dir / "evaluation_image_blacklist.json"
    blacklist_path.write_text(json.dumps(blacklist, ensure_ascii=False, indent=2), encoding="utf-8")

    train_images, dev_images, split_stats = choose_image_splits(
        groups,
        image_index,
        set(blacklist["basenames"]),
        set(blacklist["hashes"]),
        args.seed,
    )
    train_rows = sample_from_images(train_images, groups, TASK_TARGETS, "train")
    dev_rows = sample_from_images(dev_images, groups, DEV_TARGETS, "dev")

    final_names = sorted({image_basename(row["image"]) for row in train_rows + dev_rows})
    if len(final_names) != len(set(final_names)):
        raise RuntimeError("internal duplicate image name error")
    if set(train_images) & set(dev_images):
        raise RuntimeError("train/dev image split overlap")
    if any(name in set(blacklist["basenames"]) for name in final_names):
        raise RuntimeError("selected eval blacklisted basename")

    tmp_dir = output_dir.with_name(output_dir.name + "_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    (tmp_dir / "images").mkdir()

    for name in final_names:
        shutil.copy2(image_index[name], tmp_dir / "images" / name)
    write_jsonl(tmp_dir / "train.jsonl", train_rows)
    write_jsonl(tmp_dir / "train_dev.jsonl", dev_rows)

    task_counts = {
        "train": dict(Counter(row["task"] for row in train_rows)),
        "dev": dict(Counter(row["task"] for row in dev_rows)),
    }
    preliminary_audit = {
        "official_train_json": str(train_json),
        "official_image_root": str(image_root),
        "official_unique_images": structure["unique_images"],
        "source_image_count": len(image_index),
        "evaluation_blacklist_images": len(blacklist["basenames"]),
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "task_counts": task_counts,
        "final_image_count": len(final_names),
        "split_stats": split_stats,
        "ready": False,
    }
    write_readme(tmp_dir / "README.md", preliminary_audit)
    (tmp_dir / "leakage_audit_report.json").write_text(
        json.dumps(preliminary_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    wrapper = """#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.argv = [
    "validate_vrsbench_lora_v1.py",
    "--dataset-dir",
    str(Path(__file__).resolve().parent),
    "--repo-root",
    str(repo_root),
    "--json",
]
runpy.run_path(str(repo_root / "scripts" / "data" / "validate_vrsbench_lora_v1.py"), run_name="__main__")
"""
    (tmp_dir / "validate_lora_data.py").write_text(wrapper, encoding="utf-8")

    validator = repo_root / "scripts" / "data" / "validate_vrsbench_lora_v1.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(validator),
            "--dataset-dir",
            str(tmp_dir),
            "--repo-root",
            str(repo_root),
            "--json",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError("validator failed on staged LoRA dataset")
    validator_result = json.loads(completed.stdout)
    validator_summary = validator_result["summary"]

    final_audit = dict(preliminary_audit)
    final_audit["validator_summary"] = validator_summary
    final_audit["ready"] = validator_summary.get("status") == "PASS"
    (tmp_dir / "leakage_audit_report.json").write_text(
        json.dumps(final_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(tmp_dir / "README.md", final_audit)

    if not final_audit["ready"]:
        raise RuntimeError("staged dataset is not ready")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    tmp_dir.rename(output_dir)

    write_final_audit(results_dir / "lora_final_audit.md", final_audit)
    print(json.dumps(final_audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
