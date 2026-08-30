#!/usr/bin/env python3
"""Print the runtime environment and highlight reproduction risks."""

from __future__ import annotations

import importlib.metadata
import io
import platform
import shutil
import subprocess
import sys
import warnings
from contextlib import redirect_stderr
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
        "suppressed_torch_warnings": "0",
        "suppressed_torch_stderr": "",
    }
    warning_records: list[warnings.WarningMessage] = []
    stderr_buffer = io.StringIO()
    try:
        with warnings.catch_warnings(record=True) as caught, redirect_stderr(stderr_buffer):
            warnings.simplefilter("always", UserWarning)
            import torch

            info["torch"] = torch.__version__
            info["cuda_runtime"] = str(torch.version.cuda)
            info["cuda_available"] = str(torch.cuda.is_available())
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                info["gpu"] = torch.cuda.get_device_name(0)
                info["compute_capability"] = f"{props.major}.{props.minor}"
                info["gpu_total_memory_mb"] = f"{props.total_memory / 1024**2:.2f}"
            warning_records = list(caught)
    except Exception as exc:
        info["torch_import_error"] = f"{type(exc).__name__}: {exc}"
        captured = stderr_buffer.getvalue().strip()
        if captured:
            info["suppressed_torch_stderr"] = captured
        return info

    captured = stderr_buffer.getvalue().strip()
    info["suppressed_torch_warnings"] = str(len(warning_records))
    if captured:
        info["suppressed_torch_stderr"] = captured
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
            "Native Windows Python is not the verified submission environment. "
            "The verified path is Ubuntu/Linux; Windows users may use WSL2 Ubuntu."
        )
    if (py.major, py.minor) != RECOMMENDED_PYTHON:
        warnings.append(f"Recommended Python is 3.12; current Python is {py.major}.{py.minor}.")
    if torch["torch"] == "MISSING":
        warnings.append("[ERROR] PyTorch is missing or failed to import.")
    elif torch["cuda_available"] != "True":
        warnings.append("CUDA is not available to PyTorch; GPU INT4/BF16 smoke will not run.")
    suppressed_torch_text = torch.get("suppressed_torch_stderr", "")
    if (
        torch.get("compute_capability") == "12.0"
        and (not str(torch.get("cuda_runtime", "")).startswith("13") or "cu126" in torch.get("torch", ""))
    ) or any(token in suppressed_torch_text for token in ("sm_120", "no kernel image", "not compatible")):
        warnings.append(
            "GPU reports sm_120, while the current PyTorch build may not include matching kernels. "
            "Use the verified Linux/CUDA environment for inference."
        )
    for name in ("peft", "PyYAML"):
        if packages[name] == "MISSING":
            warnings.append(f"[MISSING] {name} is missing; install the verified requirements before running benchmarks.")

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
    if torch.get("suppressed_torch_warnings") != "0" or torch.get("suppressed_torch_stderr"):
        lines.extend(
            [
                "",
                "--- Suppressed Torch Import Diagnostics ---",
                f"UserWarning count: {torch.get('suppressed_torch_warnings', '0')}",
                "Long PyTorch import/CUDA diagnostics were captured locally so the structured check remains readable.",
            ]
        )
    lines.append("")
    lines.append("--- Warnings ---")
    if warnings:
        lines.extend(warning if warning.startswith("[") else f"[WARNING] {warning}" for warning in warnings)
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
