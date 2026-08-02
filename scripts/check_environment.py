import gc
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "environment_info.txt"


def _append_section(lines: list[str], title: str, value: str) -> None:
    lines.append("")
    lines.append(f"--- {title} ---")
    lines.append(value)


def collect_environment_info() -> list[str]:
    lines: list[str] = [
        "=" * 40,
        "Environment Information",
        f"Generated at: {datetime.now()}",
        "=" * 40,
    ]

    _append_section(lines, "Python Version", platform.python_version())
    _append_section(lines, "PyTorch Version", torch.__version__)
    _append_section(lines, "CUDA Available", str(torch.cuda.is_available()))

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        gpu_mem = f"{total_mem_mb:.2f}"
    else:
        gpu_name = "N/A"
        gpu_mem = "N/A"

    _append_section(lines, "GPU Full Name", gpu_name)
    _append_section(lines, "GPU Total Memory (MB)", gpu_mem)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        driver_version = result.stdout.strip().splitlines()[0] if result.stdout.strip() else "N/A"
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
        driver_version = "N/A"

    _append_section(lines, "NVIDIA Driver Version", driver_version)

    try:
        free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
        disk_free = f"{free_gb:.2f}"
    except OSError:
        disk_free = "N/A"

    _append_section(lines, "Disk Free Space (GB)", disk_free)

    lines.append("")
    lines.append("=" * 40)
    lines.append("Done.")
    return lines


def main() -> int:
    output_lines = collect_environment_info()
    output_text = "\n".join(output_lines) + "\n"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(output_text, encoding="utf-8")

    print(output_text, end="")
    print(f"[PASS] Environment info saved to {LOG_FILE.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
