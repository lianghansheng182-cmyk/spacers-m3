#!/usr/bin/env python3
"""Build the XH-202603 preliminary submission precheck package."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "submission_final"
ZIP_PATH = ROOT / "submission_final_PRECHECK_v3.zip"
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

DIR_APPLICATION = "00_application_form"
DIR_REPORT = "01_technical_report"
DIR_SOURCE = "02_source_code"
DIR_MODEL = "03_model"
DIR_RESULTS = "04_experiment_results"
DIR_REPRO = "05_reproduction"
DIR_INFO = "06_submission_info"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def ignore_runtime(_dir: str, names: list[str]) -> set[str]:
    ignored = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "logs",
        "raw",
        "models",
        "submission_final",
    }
    ignored.update(name for name in names if name.endswith((".pyc", ".pyo", ".tmp", ".temp", ".part")))
    return ignored.intersection(names)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst, ignore=ignore_runtime, dirs_exist_ok=True)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def wrap_line(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if not text:
        return [""]
    result: list[str] = []
    current = ""
    for char in text:
        probe = current + char
        width = draw.textbbox((0, 0), probe, font=font)[2]
        if width <= max_width or not current:
            current = probe
        else:
            result.append(current)
            current = char
    if current:
        result.append(current)
    return result


def draw_architecture_page(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(42)
    font = load_font(28)
    small = load_font(22)
    draw.text((90, 70), "SPACERS-M3 Final System Architecture", fill=(20, 40, 65), font=title_font)
    boxes = [
        ("Remote Sensing Input", 130),
        ("SpaceBudget-RS\nECO / BALANCED / PRECISION\nHiRes fallback", 270),
        ("HiRes-View\nsingle-view or global+2x2", 455),
        ("RemoteLoRA\nQwen3-VL-2B + LoRA v1", 625),
        ("OrbitQuant\nINT4 NF4 default, BF16 fallback", 795),
        ("OrbitGuard-RS\nIntegrity check / fault detection / recovery", 965),
        ("Caption / VQA / Grounding / Change Captioning", 1160),
    ]
    x, box_w = 170, width - 340
    for label, y in boxes:
        h = 92 if "\n" not in label else 130
        draw.rounded_rectangle((x, y, x + box_w, y + h), radius=20, outline=(42, 86, 130), width=4, fill=(236, 245, 252))
        lines = label.splitlines()
        total_h = len(lines) * 32
        for i, line in enumerate(lines):
            draw.text((x + 36, y + (h - total_h) // 2 + i * 34), line, fill=(15, 35, 55), font=font if i == 0 else small)
        if y != boxes[-1][1]:
            draw.line((width // 2, y + h + 8, width // 2, y + h + 38), fill=(42, 86, 130), width=4)
            draw.polygon(
                [(width // 2 - 10, y + h + 38), (width // 2 + 10, y + h + 38), (width // 2, y + h + 55)],
                fill=(42, 86, 130),
            )
    side_x = width - 360
    draw.rounded_rectangle((side_x, 455, side_x + 230, 760), radius=18, outline=(122, 92, 22), width=3, fill=(255, 250, 230))
    draw.text((side_x + 22, 480), "Resource Monitor", fill=(70, 55, 10), font=small)
    for i, item in enumerate(["visual tokens", "latency / P95", "peak VRAM", "failures"]):
        draw.text((side_x + 34, 535 + i * 44), f"- {item}", fill=(70, 55, 10), font=small)
    return image


def markdown_to_pdf(markdown_path: Path, pdf_path: Path, title: str) -> None:
    width, height = 1240, 1754
    margin_x, margin_y = 85, 70
    body_font = load_font(25)
    title_font = load_font(38)
    mono_font = load_font(22)
    pages: list[Image.Image] = [draw_architecture_page(width, height)]

    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    y = margin_y
    page_no = 1

    def finish_page() -> None:
        nonlocal page, draw, y, page_no
        draw.text((width - 170, height - 58), f"Page {page_no}", fill=(90, 90, 90), font=mono_font)
        pages.append(page)
        page_no += 1
        page = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(page)
        y = margin_y

    draw.text((margin_x, y), title, fill=(20, 40, 65), font=title_font)
    y += 62
    draw.text((margin_x, y), f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  Commit: {COMMIT[:12]}", fill=(90, 90, 90), font=mono_font)
    y += 52

    in_code = False
    for raw in markdown_path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if not line:
            y += 18
            continue
        font = mono_font if in_code or line.startswith("|") else body_font
        indent = 0
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            line = line[level:].strip()
            font = load_font(34 if level <= 2 else 29)
            y += 14
        elif line.startswith("- "):
            line = "• " + line[2:]
            indent = 24
        wrapped = wrap_line(draw, line, font, width - margin_x * 2 - indent)
        for part in wrapped:
            if y > height - 95:
                finish_page()
            draw.text((margin_x + indent, y), part, fill=(25, 25, 25), font=font)
            y += 34 if font == mono_font else 39
    finish_page()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    first, rest = pages[0], pages[1:]
    first.save(pdf_path, "PDF", resolution=150.0, save_all=True, append_images=rest)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def build_docs() -> None:
    write_text(
        PACKAGE / "README.md",
        f"""
        # XH-202603 初赛提交包预检版

        本目录为 SPACERS-M3 初赛提交包的本地预检版，生成自 Git commit `{COMMIT}`。

        状态说明：

        - `submission_final_PRECHECK_v3.zip` 可用于内部核验。
        - 正式上传前仍需人工加入盖章版参赛申报表，并按赛事要求将压缩包重命名为“单位+姓名+作品名称+联系方式”。
        - 本包不包含 Qwen3-VL-2B-Instruct 基座模型和公开 benchmark raw data；LoRA v1 adapter 已放入 `03_model/`。
        - 外部 artifact 仅包括基座模型、公开 benchmark raw data 和后续可能单独分发的大容量文件；LoRA v1 adapter 已随包提供。
        - 正式复现环境为 WSL2 Ubuntu / Linux。Windows native 仅作为静态检查或有限单图替代流程，不作为正式 full smoke 验证环境。

        ## Directory Layout

        | Directory | 含义 |
        | --- | --- |
        | `00_application_form/` | 参赛申报表 |
        | `01_technical_report/` | 技术报告 |
        | `02_source_code/` | 完整源代码 |
        | `03_model/` | LoRA 模型权重及模型说明 |
        | `04_experiment_results/` | 实验结果 |
        | `05_reproduction/` | 复现说明 |
        | `06_submission_info/` | 提交清单及其他说明 |

        推荐验证环境：Ubuntu 24.04 / Python 3.12 / CUDA 13.0 / NVIDIA GPU。

        快速复现：

        ```bash
        cd 02_source_code
        bash scripts/setup_linux.sh
        bash scripts/smoke_test.sh
        ```

        外部文件放置方式见 `05_reproduction/artifact_placement.md` 和 `05_reproduction/benchmark_data.md`。
        """,
    )
    write_text(
        PACKAGE / DIR_APPLICATION / "README.md",
        """
        # 参赛申报表

        EXTERNAL ARTIFACT。

        当前预检包未伪造盖章申报表。正式提交前请将赛事要求的盖章版参赛申报表放入本目录，并按赛事命名规范重命名最终压缩包。
        """,
    )
    write_text(
        PACKAGE / DIR_MODEL / "base_model.md",
        """
        # Qwen3-VL-2B-Instruct 基座模型说明

        - Base model: Qwen3-VL-2B-Instruct
        - 代码默认路径：`models/Qwen3-VL-2B-Instruct/`
        - 最终推理使用本地模型目录，不依赖在线 API 或隐藏 cache。
        - 本预检包不内置完整基座模型权重。正式复现时请将基座模型完整目录放置到 `02_source_code/models/Qwen3-VL-2B-Instruct/`。

        目录内应至少包含：

        ```text
        config.json
        generation_config.json
        tokenizer_config.json
        preprocessor_config.json
        model.safetensors
        ```

        如后续使用网盘或模型仓库分发，请在正式提交前补充真实下载地址、版本号和校验信息；不要填写占位或虚假链接。
        """,
    )
    write_text(
        PACKAGE / DIR_REPRO / "artifact_placement.md",
        """
        # 外部 artifact 放置说明

        在提交包解压后，以 `02_source_code/` 为工作目录。

        期望结构：

        ```text
        02_source_code/
        ├── models/Qwen3-VL-2B-Instruct/
        ├── results/lora/qwen3_vl_2b_lora_v1/adapter_final/
        └── data/raw/
            ├── mme_realworld_lite/
            ├── xlrs_bench_lite/
            └── levir_cc/
        ```

        LoRA v1 adapter 已随包放置于 `../03_model/qwen3_vl_2b_lora_v1_adapter/`。`scripts/check_artifacts.py` 会直接识别该路径，不需要先运行 `scripts/smoke_test.sh`。`scripts/smoke_test.sh` 会在需要时将它链接到代码默认 adapter 路径。基座模型和公开 benchmark raw data 需要按说明单独放置。
        """,
    )
    write_text(
        PACKAGE / DIR_REPRO / "benchmark_data.md",
        """
        # Benchmark 数据说明

        不随包分发的公开 raw data：

        - MME-RealWorld-Lite：使用 Remote Sensing 相关子集，目录为 `data/raw/mme_realworld_lite/`。
        - XLRS-Bench Lite：目录为 `data/raw/xlrs_bench_lite/`。当前实验结果明确标记为 PARTIAL，使用本地 44 / 74 shards。
        - LEVIR-CC：目录为 `data/raw/levir_cc/`，使用官方 test split。

        随代码保留的小型内部验证数据：

        - `datasets/mini_eval/`
        - `datasets/vrsbench_eval_v1/`

        协议标记必须保持：

        - VRSBench：INTERNAL VALIDATION
        - MME-RealWorld-RS：OFFICIAL LITE / RS subset
        - XLRS-Bench：PARTIAL
        - LEVIR-CC：OFFICIAL TEST SPLIT
        - OrbitGuard-RS：software-level SEU-equivalent / soft-error injection
        """,
    )
    write_text(
        PACKAGE / DIR_INFO / "FINAL_SUBMISSION_CHECKLIST.md",
        """
        # Final Submission Checklist

        | Item | Status | Notes |
        | --- | --- | --- |
        | Algorithm model | PASS | LoRA v1 adapter included in `03_model`; base model documented as external artifact. |
        | Complete source code | PASS | Runtime source, configs, scripts, docs, and small eval datasets included. |
        | Technical report | PASS | Markdown source and generated PDF included. |
        | Environment config | PASS | requirements files and setup script included. |
        | One-click scripts | PASS | `scripts/setup_linux.sh` and `scripts/smoke_test.sh`. |
        | Running instructions | PASS | Package README and reproduction docs included. |
        | Experiment results | PASS | Selected final summaries, metrics, and figures included. |
        | Necessary reproduction material | PASS | LoRA adapter and small VRSBench eval data included. |
        | Stamped registration form | EXTERNAL ARTIFACT | Must be added manually before official upload. |
        | Base model weights | EXTERNAL ARTIFACT | Place under `02_source_code/models/Qwen3-VL-2B-Instruct/`. |
        | Public benchmark raw data | EXTERNAL ARTIFACT | Place under `02_source_code/data/raw/`. |
        """,
    )
    write_text(
        PACKAGE / DIR_INFO / "submission_manifest.md",
        f"""
        # Submission Manifest

        Generated from commit `{COMMIT}`.

        ## Package Layout

        - `00_application_form/`: manual insertion point for the stamped registration form.
        - `01_technical_report/`: final technical report Markdown and PDF.
        - `02_source_code/`: source code, configs, scripts, docs, requirements, and small validation datasets.
        - `03_model/`: LoRA v1 adapter and base-model placement instructions.
        - `04_experiment_results/`: selected final result summaries, metrics, and figures.
        - `05_reproduction/`: artifact placement, benchmark data, and reproduction guide.
        - `06_submission_info/`: checklist and manifest.

        ## External Large Files

        - Qwen3-VL-2B-Instruct base model: not included.
        - Public benchmark raw data: not included.
        - Stamped registration form: not included in precheck package.
        - LoRA v1 adapter: included in this package, not an external artifact.

        ## Included Model Artifact

        - LoRA v1 adapter: `03_model/qwen3_vl_2b_lora_v1_adapter/`

        ## Protocol Labels

        - VRSBench: INTERNAL VALIDATION
        - MME-RealWorld-RS Lite: OFFICIAL LITE / RS subset
        - XLRS-Bench Lite: PARTIAL, 44 / 74 shards
        - LEVIR-CC: OFFICIAL TEST SPLIT
        - OrbitGuard-RS: software-level SEU-equivalent fault injection
        """,
    )


def copy_source() -> None:
    src_root = PACKAGE / DIR_SOURCE
    for name in ("README.md", "requirements.txt", "requirements-verified.txt", "requirements-linux-cu130.txt"):
        copy_file(ROOT / name, src_root / name)
    for name in ("configs", "src", "scripts", "docs"):
        copy_tree(ROOT / name, src_root / name)
    for rel in (
        "datasets/README.md",
        "datasets/mini_eval",
        "datasets/vrsbench_eval_v1",
    ):
        src = ROOT / rel
        dst = src_root / rel
        copy_tree(src, dst) if src.is_dir() else copy_file(src, dst)
    for rel in (
        "results/quantization/lora_v1_int4_eval_vrsbench90/summary.json",
        "results/lora/qwen3_vl_2b_lora_v1_eval/mini_eval_prompt_v1/summary.json",
        "results/lora/qwen3_vl_2b_lora_v1_eval/vrsbench_eval_v1_prompt_v1/summary.json",
    ):
        copy_file(ROOT / rel, src_root / rel)


def copy_results() -> None:
    result_root = PACKAGE / DIR_RESULTS
    files = [
        "results/final_core_results.md",
        "results/final_benchmark_summary.md",
        "results/submission_readiness.md",
        "results/lora/lora_model_selection_report.md",
        "results/lora/lora_v1_vs_base_report.md",
        "results/quantization/lora_v1_quantization_report.md",
        "results/quantization/lora_v1_quantization_audit.json",
        "results/data_audit/remaining_benchmarks_readiness.md",
        "results/mme_realworld_rs/report.md",
        "results/mme_realworld_rs/metrics.json",
        "results/mme_realworld_rs/summary.json",
        "results/xlrs_bench/report.md",
        "results/xlrs_bench/partial_44shards_int4/metrics.json",
        "results/xlrs_bench/partial_44shards_int4/summary.json",
        "results/levir_cc/report.md",
        "results/levir_cc/metrics.json",
        "results/levir_cc/summary.json",
        "results/highres/highres_input_audit.md",
        "results/highres/highres_tiling_ablation.md",
        "results/spacebudget/visual_token_audit.md",
        "results/spacebudget/spacebudget_ablation.md",
        "results/spacebudget/spacebudget_summary.json",
        "results/reliability/satellite_fault_tolerance_ablation.md",
        "results/reliability/orbitguard_summary.json",
        "results/final_acceptance/acceptance_summary.md",
    ]
    for rel in files:
        src = ROOT / rel
        if src.exists():
            copy_file(src, result_root / rel.removeprefix("results/"))
    for rel in ("results/spacebudget/figures", "results/reliability/figures"):
        src = ROOT / rel
        if src.exists():
            copy_tree(src, result_root / rel.removeprefix("results/"))


def copy_smoke_logs() -> None:
    src = ROOT / "docs/smoke_logs"
    if src.exists():
        copy_tree(src, PACKAGE / DIR_REPRO / "logs")


def copy_model() -> None:
    adapter_src = ROOT / "results/lora/qwen3_vl_2b_lora_v1/adapter_final"
    adapter_dst = PACKAGE / DIR_MODEL / "qwen3_vl_2b_lora_v1_adapter"
    copy_tree(adapter_src, adapter_dst)
    lines = [
        "# LoRA v1 Adapter Manifest",
        "",
        f"Generated from commit: {COMMIT}",
        "",
    ]
    total = 0
    for path in sorted(adapter_dst.rglob("*")):
        if path.is_file():
            rel = path.relative_to(adapter_dst)
            size = path.stat().st_size
            total += size
            lines.append(f"{sha256_file(path)}  {rel}  {size} bytes")
    lines.append("")
    lines.append(f"Total size: {total / 1024 / 1024:.3f} MB")
    write_text(PACKAGE / DIR_MODEL / "adapter_manifest.txt", "\n".join(lines) + "\n")


def sanitize_package_text() -> None:
    """Remove machine-local paths and stale wording from copied text artifacts."""
    replacements = {
        str(ROOT): "<REPO_ROOT>",
        "/home/lhs": "<USER_HOME>",
        "/mnt/g": "<MOUNTED_DRIVE>",
        "G:\\\\": "<WINDOWS_DRIVE>\\",
        "Technical report draft exists": "Technical report source exists",
    }
    suffixes = {".md", ".txt", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".py", ".sh", ".svg"}
    for path in PACKAGE.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def build_package() -> None:
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)
    build_docs()
    copy_source()
    copy_model()
    copy_results()
    copy_smoke_logs()
    copy_file(ROOT / "docs/competition_technical_report.md", PACKAGE / DIR_REPORT / "competition_technical_report.md")
    copy_file(ROOT / "docs/figures/spacers_agent_architecture.svg", PACKAGE / DIR_REPORT / "figures/spacers_agent_architecture.svg")
    markdown_to_pdf(ROOT / "docs/competition_technical_report.md", PACKAGE / DIR_REPORT / "technical_report.pdf", "SPACERS-M3 技术报告")
    markdown_to_pdf(PACKAGE / DIR_INFO / "submission_manifest.md", PACKAGE / DIR_INFO / "submission_manifest.pdf", "SPACERS-M3 提交清单")
    copy_file(ROOT / "docs/reproduction_guide.md", PACKAGE / DIR_REPRO / "reproduction_guide.md")
    sanitize_package_text()

    sums: list[str] = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            sums.append(f"{sha256_file(path)}  {path.relative_to(PACKAGE)}")
    write_text(PACKAGE / "SHA256SUMS.txt", "\n".join(sums) + "\n")


def make_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    subprocess.run(["zip", "-qr", str(ZIP_PATH), PACKAGE.name], cwd=ROOT, check=True)
    (ROOT / f"{ZIP_PATH.name}.sha256").write_text(f"{sha256_file(ZIP_PATH)}  {ZIP_PATH.name}\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", action="store_true", help="Also create submission_final_PRECHECK_v3.zip")
    args = parser.parse_args()
    build_package()
    if args.zip:
        make_zip()
    print(f"[PASS] Built {PACKAGE.relative_to(ROOT)} from commit {COMMIT}")
    if args.zip:
        print(f"[PASS] Built {ZIP_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
