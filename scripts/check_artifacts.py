#!/usr/bin/env python3
"""Check whether external artifacts are placed at the documented paths."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_ADAPTER_SHA256 = {
    "adapter_config.json": "d5be846a7ee1e164dee38cf8a526fcb8de60c2dcd1b96a97e877d38e55b20ede",
    "adapter_model.safetensors": "a5d4949df5715cd04cc9ce18e7b9625934e95393703c80196bbf7c5115fe81d6",
    "README.md": "e34d4cbe24b794b76844a32cc674672e9e6613a5c5f53b924ba72f74e4c17e0f",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_path(label: str, path: Path, *, required: bool = True) -> bool:
    ok = path.exists()
    prefix = "[PASS]" if ok else ("[MISSING]" if required else "[OPTIONAL-MISSING]")
    print(f"{prefix} {label}: {path}")
    return ok or not required


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.glob(pattern)) if path.exists() else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SPACERS-M3 local artifacts.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root.")
    parser.add_argument("--skip-raw", action="store_true", help="Do not fail on missing external benchmark raw data.")
    args = parser.parse_args()

    root = args.root.resolve()
    required_ok = True

    print(f"Repository root: {root}")
    required_ok &= check_path("mini_eval manifest", root / "datasets/mini_eval/mini_eval.jsonl")
    required_ok &= check_path("mini_eval images", root / "datasets/mini_eval/images")
    required_ok &= check_path("VRSBench 90 manifest", root / "datasets/vrsbench_eval_v1/resolved_manifest.jsonl")
    required_ok &= check_path("VRSBench 90 images", root / "datasets/vrsbench_eval_v1/images")

    model_dir = root / "models/Qwen3-VL-2B-Instruct"
    required_ok &= check_path("base model directory", model_dir)
    for name in ("config.json", "generation_config.json", "tokenizer_config.json", "preprocessor_config.json"):
        required_ok &= check_path(f"base model {name}", model_dir / name)
    if model_dir.exists():
        has_weight = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
        print(("[PASS]" if has_weight else "[MISSING]") + " base model weight file: *.safetensors or *.bin")
        required_ok &= has_weight

    adapter_dir = root / "results/lora/qwen3_vl_2b_lora_v1/adapter_final"
    required_ok &= check_path("LoRA v1 adapter directory", adapter_dir)
    for name, expected_sha in EXPECTED_ADAPTER_SHA256.items():
        path = adapter_dir / name
        exists = check_path(f"LoRA v1 {name}", path)
        required_ok &= exists
        if exists:
            actual_sha = sha256_file(path)
            status = "[PASS]" if actual_sha == expected_sha else "[WARNING]"
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"{status} LoRA v1 {name} sha256={actual_sha} size_mb={size_mb:.3f}")

    raw_required = not args.skip_raw
    required_ok &= check_path("MME-RealWorld-Lite raw data", root / "data/raw/mme_realworld_lite", required=raw_required)
    required_ok &= check_path("XLRS-Bench-lite raw data", root / "data/raw/xlrs_bench_lite", required=raw_required)
    required_ok &= check_path("LEVIR-CC raw data", root / "data/raw/levir_cc", required=raw_required)

    xlrs_train = root / "data/raw/xlrs_bench_lite/train"
    shard_count = count_files(xlrs_train, "data-*-of-00074.arrow")
    if xlrs_train.exists():
        status = "[PASS]" if shard_count == 74 else "[WARNING]"
        print(f"{status} XLRS arrow shards present: {shard_count}/74")

    if required_ok:
        print("[PASS] Artifact precheck completed.")
        return 0
    print("[FAIL] Required artifacts are missing. Place them at the paths shown above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
