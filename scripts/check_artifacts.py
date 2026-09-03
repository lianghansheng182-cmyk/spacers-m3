#!/usr/bin/env python3
"""Check whether SPACERS-M3 artifacts are placed at the documented paths."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER_REL = Path("results/lora/qwen3_vl_2b_lora_v1/adapter_final")
PACKAGED_ADAPTER_REL = Path("03_model/qwen3_vl_2b_lora_v1_adapter")
EXPECTED_ADAPTER_SHA256 = {
    "adapter_config.json": "d5be846a7ee1e164dee38cf8a526fcb8de60c2dcd1b96a97e877d38e55b20ede",
    "adapter_model.safetensors": "a5d4949df5715cd04cc9ce18e7b9625934e95393703c80196bbf7c5115fe81d6",
    "README.md": "e34d4cbe24b794b76844a32cc674672e9e6613a5c5f53b924ba72f74e4c17e0f",
}
BAIDU_LINK = "https://pan.baidu.com/s/1GrAYgwE0YkL8rRalwX4g4Q?pwd=r0k4"
BAIDU_CODE = "r0k4"
BAIDU_FOLDER = "external_uploads"
BASE_MODEL_ZIP = "Qwen3-VL-2B-Instruct__SpaceRS-M3__base_model.zip"
BASE_MODEL_SHA256 = "f97c7387001f1d371a0aa12e5232abaf42cb1c84f73fd9c7b4883bba56af70a7"
RAW_DATA_ZIP = "raw_benchmark_data__SpaceRS-M3.zip"
RAW_DATA_SHA256 = "9fe29f2dad8455de88dee725122523e50d9537cf64d1db5ccf17ea5ecd651208"


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


def check_external_path(label: str, path: Path, artifact: str, sha256: str, placement: str) -> bool:
    if path.exists():
        print(f"[PASS] {label}: {path}")
        return True
    print(f"[EXPECTED-EXTERNAL] {label}: {path}")
    print(f"  Baidu link: {BAIDU_LINK}")
    print(f"  Extraction code: {BAIDU_CODE}")
    print(f"  Shared folder: {BAIDU_FOLDER}")
    print(f"  File: {artifact}")
    print(f"  SHA256: {sha256}")
    print(f"  Placement after extraction: {placement}")
    return True


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.glob(pattern)) if path.exists() else 0


def resolve_roots(root: Path) -> tuple[Path, Path]:
    """Return (source_root, package_root).

    The checker can run either inside a Git checkout/source tree or from a
    fresh submission package. In the package layout, the source tree is
    `submission_final/02_source_code` and the LoRA adapter is stored beside it
    under `submission_final/03_model`.
    """

    root = root.resolve()
    if (root / "02_source_code/scripts/check_artifacts.py").exists():
        return root / "02_source_code", root
    package_root = root.parent if root.name == "02_source_code" else root.parent
    return root, package_root


def adapter_candidates(source_root: Path, package_root: Path) -> list[Path]:
    candidates = [
        source_root / DEFAULT_ADAPTER_REL,
        package_root / PACKAGED_ADAPTER_REL,
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve(strict=False)
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def find_adapter_dir(source_root: Path, package_root: Path) -> Path | None:
    for path in adapter_candidates(source_root, package_root):
        if path.exists():
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SPACERS-M3 local artifacts.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root.")
    parser.add_argument("--skip-raw", action="store_true", help="Skip raw benchmark data checks.")
    parser.add_argument(
        "--require-external",
        action="store_true",
        help="Fail if externally supplied base model or raw benchmark data have not been placed locally.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    source_root, package_root = resolve_roots(root)
    required_ok = True

    print(f"Input root: {root}")
    print(f"Source root: {source_root}")
    print(f"Package root: {package_root}")
    required_ok &= check_path("mini_eval manifest", source_root / "datasets/mini_eval/mini_eval.jsonl")
    required_ok &= check_path("mini_eval images", source_root / "datasets/mini_eval/images")
    required_ok &= check_path("VRSBench 90 manifest", source_root / "datasets/vrsbench_eval_v1/resolved_manifest.jsonl")
    required_ok &= check_path("VRSBench 90 images", source_root / "datasets/vrsbench_eval_v1/images")

    model_dir = source_root / "models/Qwen3-VL-2B-Instruct"
    if args.require_external:
        required_ok &= check_path("base model directory", model_dir)
    else:
        required_ok &= check_external_path(
            "base model directory",
            model_dir,
            BASE_MODEL_ZIP,
            BASE_MODEL_SHA256,
            "submission_final/02_source_code/models/Qwen3-VL-2B-Instruct/",
        )
    for name in ("config.json", "generation_config.json", "tokenizer_config.json", "preprocessor_config.json"):
        if model_dir.exists():
            required_ok &= check_path(f"base model {name}", model_dir / name, required=args.require_external)
    if model_dir.exists():
        has_weight = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
        print(("[PASS]" if has_weight else "[MISSING]") + " base model weight file: *.safetensors or *.bin")
        required_ok &= has_weight or not args.require_external

    print("LoRA v1 adapter candidates:")
    for candidate in adapter_candidates(source_root, package_root):
        print(f"  - {candidate}")
    adapter_dir = find_adapter_dir(source_root, package_root)
    if adapter_dir is None:
        required_ok &= check_path("LoRA v1 adapter directory", source_root / DEFAULT_ADAPTER_REL)
        adapter_dir = source_root / DEFAULT_ADAPTER_REL
    else:
        print(f"[PASS] Detected LoRA v1 adapter directory: {adapter_dir.resolve()}")
    for name, expected_sha in EXPECTED_ADAPTER_SHA256.items():
        path = adapter_dir / name
        exists = check_path(f"LoRA v1 {name}", path)
        required_ok &= exists
        if exists:
            actual_sha = sha256_file(path)
            status = "[PASS]" if actual_sha == expected_sha else "[WARNING]"
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"{status} LoRA v1 {name} sha256={actual_sha} size_mb={size_mb:.3f}")

    if args.skip_raw:
        raw_required = False
        required_ok &= check_path("MME-RealWorld-Lite raw data", source_root / "data/raw/mme_realworld_lite", required=False)
        required_ok &= check_path("XLRS-Bench-lite raw data", source_root / "data/raw/xlrs_bench_lite", required=False)
        required_ok &= check_path("LEVIR-CC raw data", source_root / "data/raw/levir_cc", required=False)
    elif args.require_external:
        raw_required = True
        required_ok &= check_path("MME-RealWorld-Lite raw data", source_root / "data/raw/mme_realworld_lite", required=True)
        required_ok &= check_path("XLRS-Bench-lite raw data", source_root / "data/raw/xlrs_bench_lite", required=True)
        required_ok &= check_path("LEVIR-CC raw data", source_root / "data/raw/levir_cc", required=True)
    else:
        raw_required = False
        for label, path in (
            ("MME-RealWorld-Lite raw data", source_root / "data/raw/mme_realworld_lite"),
            ("XLRS-Bench-lite raw data", source_root / "data/raw/xlrs_bench_lite"),
            ("LEVIR-CC raw data", source_root / "data/raw/levir_cc"),
        ):
            if path.exists():
                required_ok &= check_path(label, path, required=False)
            else:
                required_ok &= check_external_path(
                    label,
                    path,
                    RAW_DATA_ZIP,
                    RAW_DATA_SHA256,
                    "submission_final/02_source_code/data/raw/",
                )

    xlrs_train = source_root / "data/raw/xlrs_bench_lite/train"
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
