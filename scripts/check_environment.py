#!/usr/bin/env python3
"""Print the runtime environment and highlight reproduction risks."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "environment_info.txt"
RECOMMENDED_PYTHON = (3, 12)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def torch_info() -> dict[str, str]:
    info = {
        "torch": "MISSING",
        "cuda_available": "False",
        "cuda_runtime": "N/A",
        "gpu": "N/A",
        "compute_capability": "N/A",
        "gpu_total_memory_mb": "N/A",
    }
    try:
        import torch
    except Exception as exc:
        info["torch_import_error"] = f"{type(exc).__name__}: {exc}"
        return info

    info["torch"] = torch.__version__
    info["cuda_runtime"] = str(torch.version.cuda)
    info["cuda_available"] = str(torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = torch.cuda.get_device_name(0)
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["gpu_total_memory_mb"] = f"{props.total_memory / 1024**2:.2f}"
    return info


def nvidia_driver() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip().splitlines()[0] if result.stdout.strip() else "N/A"
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
        return "N/A"


def collect_environment_info() -> tuple[list[str], list[str]]:
    py = sys.version_info
    torch = torch_info()
    packages = {
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "bitsandbytes": package_version("bitsandbytes"),
        "PyYAML": package_version("PyYAML"),
        "accelerate": package_version("accelerate"),
        "qwen-vl-utils": package_version("qwen-vl-utils"),
    }
    warnings: list[str] = []

    if platform.system().lower() == "windows":
        warnings.append(
            "Windows native Python is not the verified submission environment. "
            "Use WSL2 Ubuntu for the documented path."
        )
    if (py.major, py.minor) != RECOMMENDED_PYTHON:
        warnings.append(f"Recommended Python is 3.12; current Python is {py.major}.{py.minor}.")
    if torch["torch"] == "MISSING":
        warnings.append("PyTorch is missing or failed to import.")
    elif torch["cuda_available"] != "True":
        warnings.append("CUDA is not available to PyTorch; GPU INT4/BF16 smoke will not run.")
    if torch.get("compute_capability") == "12.0" and "cu126" in torch.get("torch", ""):
        warnings.append(
            "This GPU reports sm_120, but the installed PyTorch build appears to be cu126; "
            "use the verified Linux/WSL CUDA 13 environment if kernels fail."
        )
    for name in ("peft", "PyYAML"):
        if packages[name] == "MISSING":
            warnings.append(f"{name} is missing; install the verified requirements before running benchmarks.")

    lines: list[str] = [
        "=" * 40,
        "Environment Information",
        f"Generated at: {datetime.now()}",
        "=" * 40,
        "",
        "--- System ---",
        f"OS: {platform.system()} {platform.release()}",
        f"Platform: {platform.platform()}",
        f"Python: {platform.python_version()}",
        f"Executable: {sys.executable}",
        "",
        "--- PyTorch / CUDA ---",
        f"torch: {torch['torch']}",
        f"CUDA runtime: {torch['cuda_runtime']}",
        f"CUDA available: {torch['cuda_available']}",
        f"GPU: {torch['gpu']}",
        f"Compute capability: {torch['compute_capability']}",
        f"GPU total memory MB: {torch['gpu_total_memory_mb']}",
        f"NVIDIA driver: {nvidia_driver()}",
        "",
        "--- Python Packages ---",
    ]
    lines.extend(f"{name}: {version}" for name, version in packages.items())
    try:
        free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
        lines.extend(["", "--- Disk ---", f"Project disk free GB: {free_gb:.2f}"])
    except OSError:
        pass
    if "torch_import_error" in torch:
        lines.extend(["", "--- Torch Import Error ---", torch["torch_import_error"]])
    lines.append("")
    lines.append("--- Warnings ---")
    if warnings:
        lines.extend(f"[WARNING] {warning}" for warning in warnings)
    else:
        lines.append("[PASS] Environment matches the verified reproduction profile closely enough.")
    return lines, warnings


def main() -> int:
    output_lines, _warnings = collect_environment_info()
    output_text = "\n".join(output_lines) + "\n"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(output_text, encoding="utf-8")
    print(output_text, end="")
    print(f"[PASS] Environment info saved to {LOG_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
