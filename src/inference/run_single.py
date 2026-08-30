#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODEL_DIR = REPO_ROOT / "models" / "Qwen3-VL-2B-Instruct"


def build_error(message: str, exc: BaseException | None = None) -> str:
    tips = [
        "确认已激活项目虚拟环境：source .venv/bin/activate",
        "确认 GPU 可见：nvidia-smi && python -c \"import torch; print(torch.cuda.is_available())\"",
        "若模型下载失败，检查网络或 Hugging Face 访问权限，并重试运行脚本。",
        "若出现 CUDA OOM，关闭占用显存的进程，或换更小图片后重试；本脚本不做量化。",
        "若图片路径报错，请确认 --image 指向真实存在的本地图片文件。",
    ]
    detail = f"{type(exc).__name__}: {exc}" if exc else message
    return f"{message}\n错误详情：{detail}\n修复建议：\n- " + "\n- ".join(tips)


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def to_device(inputs: Any, device: torch.device) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Qwen3-VL image inference.")
    parser.add_argument("--model-path", default=None, help="Local Qwen3-VL model directory.")
    parser.add_argument("--image", required=True, help="Path to a local image.")
    parser.add_argument("--question", required=True, help="Question about the image.")
    parser.add_argument("--output", required=True, help="UTF-8 JSON output path.")
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024, help="Per-image visual pixel budget.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not image_path.exists() or not image_path.is_file():
        print(build_error(f"图片不存在或不是文件：{image_path}"), file=sys.stderr)
        return 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype(device)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    model_load_path = args.model_path or os.environ.get("SPACERS_M3_MODEL_PATH", str(DEFAULT_LOCAL_MODEL_DIR))
    if not Path(model_load_path).exists():
        model_load_path = MODEL_ID

    model_loaded = False
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)

        print(f"Loading model: {MODEL_ID}")
        print(f"Model load path: {model_load_path}")
        print(f"Device: {device}; dtype: {dtype}; GPU: {gpu_name or 'N/A'}")
        processor = AutoProcessor.from_pretrained(model_load_path, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_load_path,
            dtype=dtype,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        if device.type == "cpu":
            model = model.to(device)
        model.eval()
        model_loaded = True
        print("Model loaded successfully.")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path),
                        "max_pixels": args.max_pixels,
                    },
                    {"type": "text", "text": args.question},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = to_device(inputs, device)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)

        start = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_s = time.perf_counter() - start

        generated_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        peak_vram_mb = (
            torch.cuda.max_memory_allocated(0) / 1024 / 1024
            if device.type == "cuda"
            else 0.0
        )

        result = {
            "model": MODEL_ID,
            "image": str(image_path),
            "question": args.question,
            "answer": answer,
            "latency_s": round(latency_s, 4),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": gpu_name,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(answer)
        print(f"Saved JSON: {output_path}")
        print(f"Latency seconds: {result['latency_s']}")
        print(f"Peak VRAM MB: {result['peak_vram_mb']}")
        return 0
    except torch.cuda.OutOfMemoryError as exc:
        print(build_error("CUDA 显存不足，推理未完成。", exc), file=sys.stderr)
        return 10
    except Exception as exc:
        state = "模型已加载，推理或保存阶段失败。" if model_loaded else "模型加载失败。"
        print(build_error(state, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
