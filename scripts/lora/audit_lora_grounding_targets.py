#!/usr/bin/env python3
"""Audit real supervised assistant targets for LoRA grounding samples."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from transformers import AutoProcessor


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_messages(row: dict[str, Any], image_path: Path, include_answer: bool) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": row["question"]},
            ],
        }
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]})
    return messages


def target_kind(text: str) -> str:
    if re.fullmatch(r"\{(?:<\d+>){4}\}(?:<\|im_end\|>\n?)?", text):
        return "bbox_brace_angle"
    if re.search(r"\d", text):
        return "numeric_other"
    return "text"


def audit(config_path: Path, output_path: Path, sample_count: int, example_count: int) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = Path(".").resolve()
    train_path = repo_root / config["data"]["train"]
    image_root = repo_root / config["data"]["image_root"]
    rows = [row for row in load_jsonl(train_path) if row.get("task") == "grounding"]
    rng = random.Random(int(config["run"]["seed"]))
    sampled = rng.sample(rows, min(sample_count, len(rows)))

    processor = AutoProcessor.from_pretrained(repo_root / config["model"]["path"], local_files_only=True)
    examples: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in sampled:
        image_path = image_root / row["image"]
        image = Image.open(image_path).convert("RGB")
        full_text = processor.apply_chat_template(
            make_messages(row, image_path, include_answer=True),
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = processor.apply_chat_template(
            make_messages(row, image_path, include_answer=False),
            tokenize=False,
            add_generation_prompt=True,
        )
        full = processor(text=[full_text], images=[image], return_tensors="pt")
        prompt = processor(text=[prompt_text], images=[image], return_tensors="pt")
        target_ids = full["input_ids"][0, prompt["input_ids"].shape[1] :]
        target_text = processor.batch_decode(
            [target_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        kind = target_kind(target_text)
        counts[kind] += 1
        if len(examples) < example_count:
            examples.append(
                {
                    "sample_id": row["sample_id"],
                    "source_sample_id": row.get("source_sample_id"),
                    "question": row["question"],
                    "manifest_answer": row["answer"],
                    "bbox": row.get("bbox"),
                    "full_train_text": full_text,
                    "prompt_text": prompt_text,
                    "supervised_assistant_target": target_text,
                    "target_kind": kind,
                }
            )

    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": str(config_path),
        "train_path": str(train_path),
        "grounding_rows": len(rows),
        "sampled_rows": len(sampled),
        "target_kind_counts": dict(counts),
        "examples": examples,
        "status": "PASS" if counts and set(counts) == {"bbox_brace_angle"} else "FAIL",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--example-count", type=int, default=5)
    args = parser.parse_args()
    report = audit(args.config, args.output, args.sample_count, args.example_count)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
